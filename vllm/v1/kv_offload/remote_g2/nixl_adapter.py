# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw NIXL agent wrapper for Remote G2 transfers.

Mirrors the TRT-LLM ``remote_g2_raw_nixl_adapter.py`` shape:

* Source side: ``build_source_agent`` — construct a ``nixl_agent``,
  register the host pool, return a ``NixlSourceBundle`` (agent +
  metadata bytes + pool base/size) that ``SourceG2RpcServer.get_metadata``
  hands to peers.
* Target side: ``RawNixlRemoteG2Adapter`` — construct an agent, register
  the local GPU pool, add each remote agent, then build aligned descriptor
  lists and issue bounded multi-block ``initialize_xfer`` READs.

When the ``nixl`` python module is not installed, ``NIXL_AVAILABLE`` is
``False`` and the adapter exposes a *mock* transport that copies bytes
through a /dev/shm-backed channel; the upper layers see the same call
shape so the source/target dataflow can be exercised end-to-end in CI
without NIXL or a GPU.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    # The packaged module name is ``nixl`` and the public API is exposed
    # at ``nixl._api``. The CUDA build also installs ``nixl_cu13`` but
    # users always import the version-agnostic ``nixl`` entry point.
    from nixl._api import (  # type: ignore[import-not-found]
        nixl_agent,
        nixl_agent_config,
    )

    NIXL_AVAILABLE = True
except ImportError:
    nixl_agent = None  # type: ignore[assignment]
    nixl_agent_config = None  # type: ignore[assignment]
    NIXL_AVAILABLE = False


@dataclass
class NixlSourceBundle:
    """What the source publishes via ``get_metadata`` so peers can
    reach its host pool.

    vLLM v1 allocates one CPU tensor per transformer layer (e.g. 36 for
    Qwen3-8B). Each layer is a contiguous DRAM region with its own
    base pointer. ``layer_pool_base_ptrs`` carries one base pointer per
    layer; the byte offset for ``block_id`` within layer ``i`` is
    ``block_id * page_size_bytes`` (uniform stride across layers).

    For backward compatibility with the TRT-LLM-shape bundle (which
    assumed a single contiguous pool), ``pool_base_ptr`` is the first
    layer's pointer and ``pool_size_bytes`` is its size. Peers that
    know about multi-layer should consult ``layer_pool_base_ptrs``;
    peers that don't will only see layer 0 (and will produce garbage on
    multi-layer models, which is the bug we're fixing here).
    """

    agent: Any
    source_generation: int
    remote_name: str
    agent_desc: bytes
    pool_base_ptr: int
    pool_size_bytes: int
    layer_pool_base_ptrs: list[int]
    layer_pool_size_bytes: list[int]
    page_size_bytes: int


def build_source_agent(
    agent_name: str,
    layer_pool_base_ptrs: list[int],
    layer_pool_size_bytes: list[int],
    page_size_bytes: int,
    *,
    source_generation: int = 1,
    backends: tuple[str, ...] = ("UCX",),
    mem_type: str = "DRAM",
    device_id: int = 0,
) -> NixlSourceBundle | None:
    """Construct the source NIXL agent and register **every** per-layer
    pool with the agent. One ``register_memory`` call covers all
    layers (NIXL accepts a list of regions).

    Returns ``None`` on failure. When ``backends`` is empty OR NIXL is
    not importable, falls back to the byte-copy mock transport so the
    rest of the pipeline can still be exercised in CI.
    """
    if not layer_pool_base_ptrs:
        raise ValueError("layer_pool_base_ptrs must be non-empty")
    if len(layer_pool_base_ptrs) != len(layer_pool_size_bytes):
        raise ValueError("layer_pool_base_ptrs / layer_pool_size_bytes length mismatch")

    if not NIXL_AVAILABLE or not backends:
        if not NIXL_AVAILABLE:
            logger.warning(
                "RemoteG2: NIXL python module not available; using mock "
                "transport (ctypes.memmove). Wire protocol unchanged so "
                "peers can negotiate without code changes."
            )
        return _build_mock_source_bundle(
            agent_name,
            layer_pool_base_ptrs,
            layer_pool_size_bytes,
            page_size_bytes,
            source_generation,
        )

    try:
        config = nixl_agent_config(True, False, 0, False, 0, list(backends))
        agent = nixl_agent(agent_name, config, instantiate_all=False)
    except Exception:
        logger.exception("RemoteG2: nixl_agent construction failed")
        return None

    try:
        reg_list = [
            (int(p), int(s), device_id, "")
            for p, s in zip(layer_pool_base_ptrs, layer_pool_size_bytes)
        ]
        agent.register_memory(reg_list, mem_type=mem_type)
    except Exception:
        logger.exception(
            "RemoteG2: register_memory failed for %d layer pools",
            len(layer_pool_base_ptrs),
        )
        return None

    try:
        agent_desc = agent.get_agent_metadata()
    except Exception:
        logger.exception("RemoteG2: get_agent_metadata failed")
        return None

    return NixlSourceBundle(
        agent=agent,
        source_generation=source_generation,
        remote_name=agent_name,
        agent_desc=agent_desc,
        pool_base_ptr=int(layer_pool_base_ptrs[0]),
        pool_size_bytes=int(layer_pool_size_bytes[0]),
        layer_pool_base_ptrs=[int(p) for p in layer_pool_base_ptrs],
        layer_pool_size_bytes=[int(s) for s in layer_pool_size_bytes],
        page_size_bytes=int(page_size_bytes),
    )


class RawNixlRemoteG2Adapter:
    """Target-side adapter performing batched per-layer NIXL READs.

    vLLM v1 allocates one CPU tensor per transformer layer (e.g. 36 for
    Qwen3-8B) and similarly one GPU tensor per layer for the target's
    KV cache. The adapter takes the *list* of per-layer base pointers
    on each side, registers all of them with NIXL in a single
    ``register_memory`` call. ``read_blocks`` groups logical blocks into
    bounded chunks; each NIXL transfer contains ``chunk_blocks *
    num_layers`` source/destination descriptors. This amortises the
    transaction setup and completion polling cost while bounding descriptor
    list size. ``read_block`` remains as a compatibility wrapper.

    The mock transport mirrors this by memcpy-ing every layer.
    """

    def __init__(
        self,
        agent_name: str,
        local_layer_pool_base_ptrs: list[int],
        local_layer_pool_size_bytes: list[int],
        *,
        backends: tuple[str, ...] = ("UCX",),
        use_mock: bool = False,
        local_mem_type: str = "VRAM",
        local_device_id: int = 0,
        poll_interval_s: float = 0.0005,
        fault_inject_every: int = 0,
        max_blocks_per_xfer: int = 64,
    ) -> None:
        if not local_layer_pool_base_ptrs:
            raise ValueError("local_layer_pool_base_ptrs must be non-empty")
        if len(local_layer_pool_base_ptrs) != len(local_layer_pool_size_bytes):
            raise ValueError("local_layer_pool_base_ptrs / size_bytes length mismatch")
        self._agent_name = agent_name
        self._local_layer_bases = [int(p) for p in local_layer_pool_base_ptrs]
        self._local_layer_sizes = [int(s) for s in local_layer_pool_size_bytes]
        self._local_mem_type = local_mem_type
        self._local_device_id = int(local_device_id)
        self._poll_interval_s = float(poll_interval_s)
        self._backends = tuple(backends)
        self._use_mock = use_mock or not NIXL_AVAILABLE or not backends
        self._max_blocks_per_xfer = int(max_blocks_per_xfer)
        if self._max_blocks_per_xfer <= 0:
            raise ValueError("max_blocks_per_xfer must be positive")
        self._peers: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        # Fault injection: when >0, every Nth logical block read raises
        # RuntimeError. The count is independent of NIXL transaction
        # batching. Used by the failure-recovery test suite to confirm the
        # handler propagates failures to vLLM's load policy. Defaults to off.
        self._fault_inject_every = int(fault_inject_every)
        self._read_block_calls = 0

        if self._use_mock:
            self._agent: Any = _MockAgent(agent_name, self._local_layer_bases[0])
            self._agent_metadata = _MockAgent.encode_metadata(agent_name)
            return

        config = nixl_agent_config(True, False, 0, False, 0, list(backends))
        self._agent = nixl_agent(agent_name, config, instantiate_all=False)
        reg_list = [
            (p, s, self._local_device_id, "")
            for p, s in zip(self._local_layer_bases, self._local_layer_sizes)
        ]
        self._agent.register_memory(reg_list, mem_type=self._local_mem_type)
        self._agent_metadata = self._agent.get_agent_metadata()

    @property
    def agent_metadata(self) -> bytes:
        return self._agent_metadata

    @property
    def num_layers(self) -> int:
        return len(self._local_layer_bases)

    @property
    def use_mock(self) -> bool:
        """True when transfers are host memcpy, not a real NIXL/UCX READ.

        Set when the caller forced mock, NIXL is unavailable, or no backend
        was requested (see __init__). A fail-closed perf gate must reject a
        run where this is True -- mock memcpy still bumps the completion
        counters but proves nothing about the network path.
        """
        return self._use_mock

    @property
    def transport_backend(self) -> str:
        """Human-readable transport tag: "MOCK" or the NIXL backend list."""
        if self._use_mock:
            return "MOCK"
        return ",".join(self._backends) if self._backends else "MOCK"

    def add_peer(
        self,
        peer_name: str,
        peer_agent_metadata: bytes,
        peer_layer_pool_base_ptrs: list[int],
        peer_layer_pool_size_bytes: list[int],
        *,
        peer_mem_type: str = "DRAM",
        peer_device_id: int = 0,
    ) -> None:
        """Idempotent: register the source peer's per-layer pools."""
        if not peer_layer_pool_base_ptrs:
            raise ValueError("peer_layer_pool_base_ptrs must be non-empty")
        if len(peer_layer_pool_base_ptrs) != len(peer_layer_pool_size_bytes):
            raise ValueError("peer_layer_pool_base_ptrs / size_bytes length mismatch")
        if len(peer_layer_pool_base_ptrs) != self.num_layers:
            raise ValueError(
                f"peer has {len(peer_layer_pool_base_ptrs)} layers but "
                f"local adapter has {self.num_layers} (model mismatch?)"
            )
        with self._lock:
            if peer_name in self._peers:
                return
            if self._use_mock:
                self._peers[peer_name] = {
                    "layer_bases": [int(p) for p in peer_layer_pool_base_ptrs],
                    "layer_sizes": [int(s) for s in peer_layer_pool_size_bytes],
                    "metadata": peer_agent_metadata,
                }
                return

            loaded = self._agent.add_remote_agent(peer_agent_metadata)
            remote_handle = (
                loaded
                if isinstance(loaded, str)
                else loaded.decode("utf-8", errors="ignore")
            )
            self._peers[peer_name] = {
                "handle": remote_handle,
                "layer_bases": [int(p) for p in peer_layer_pool_base_ptrs],
                "layer_sizes": [int(s) for s in peer_layer_pool_size_bytes],
                "peer_mem_type": peer_mem_type,
                "peer_device_id": int(peer_device_id),
            }

    def read_block(
        self,
        peer_name: str,
        peer_byte_offset: int,
        local_byte_offset: int,
        byte_length: int,
    ) -> None:
        """Compatibility wrapper for a one-block synchronous READ."""
        self.read_blocks(
            peer_name,
            peer_byte_offsets=[peer_byte_offset],
            local_byte_offsets=[local_byte_offset],
            byte_lengths=[byte_length],
        )

    def read_blocks(
        self,
        peer_name: str,
        peer_byte_offsets: Sequence[int],
        local_byte_offsets: Sequence[int],
        byte_lengths: Sequence[int],
    ) -> None:
        """Synchronously READ aligned logical blocks across every layer.

        The three input sequences are positionally aligned and retain their
        caller-provided order. Descriptor lists use block-major, layer-minor
        order on both sides. Logical blocks are chunked so a transfer never
        contains more than ``max_blocks_per_xfer * num_layers`` descriptors.

        Fault injection continues to count logical blocks, not NIXL
        transactions. If an injected failure falls inside this call, blocks
        preceding it are transferred and the injected block plus all later
        blocks are skipped, matching the former per-block loop.
        """
        block_count = len(peer_byte_offsets)
        if len(local_byte_offsets) != block_count or len(byte_lengths) != block_count:
            raise ValueError(
                "peer/local offsets and byte_lengths must have equal length"
            )
        if block_count == 0:
            return

        peer_offsets = [int(value) for value in peer_byte_offsets]
        local_offsets = [int(value) for value in local_byte_offsets]
        lengths = [int(value) for value in byte_lengths]

        with self._lock:
            peer = self._peers.get(peer_name)
            if peer is None:
                # The old per-block implementation counted the first attempted
                # read before reporting a missing peer, then stopped the job.
                self._read_block_calls += 1
                raise RuntimeError(f"peer {peer_name!r} not registered; call add_peer")
            peer = dict(peer)

        peer_layer_bases = peer["layer_bases"]
        n_layers = len(peer_layer_bases)
        if n_layers != self.num_layers:
            raise RuntimeError(
                f"peer {peer_name!r} layer count {n_layers} mismatches "
                f"local {self.num_layers}"
            )

        peer_layer_sizes = peer["layer_sizes"]
        peer_limit = min(int(size) for size in peer_layer_sizes)
        local_limit = min(self._local_layer_sizes)
        for index, (peer_offset, local_offset, length) in enumerate(
            zip(peer_offsets, local_offsets, lengths)
        ):
            if peer_offset < 0 or local_offset < 0:
                raise ValueError(f"block {index} has a negative byte offset")
            if length <= 0:
                raise ValueError(f"block {index} byte_length must be positive")
            if peer_offset + length > peer_limit:
                raise ValueError(
                    f"block {index} exceeds peer layer pool: "
                    f"offset={peer_offset} length={length} limit={peer_limit}"
                )
            if local_offset + length > local_limit:
                raise ValueError(
                    f"block {index} exceeds local layer pool: "
                    f"offset={local_offset} length={length} limit={local_limit}"
                )

        inject_index: int | None = None
        inject_call = 0
        with self._lock:
            for index in range(block_count):
                self._read_block_calls += 1
                if (
                    self._fault_inject_every > 0
                    and self._read_block_calls % self._fault_inject_every == 0
                ):
                    inject_index = index
                    inject_call = self._read_block_calls
                    break

        transfer_count = block_count if inject_index is None else inject_index
        if transfer_count:
            self._read_blocks_impl(
                peer_name,
                peer,
                peer_offsets[:transfer_count],
                local_offsets[:transfer_count],
                lengths[:transfer_count],
            )

        if inject_index is not None:
            raise RuntimeError(f"RemoteG2 fault injection: read_block #{inject_call}")

    def _read_blocks_impl(
        self,
        peer_name: str,
        peer: dict[str, Any],
        peer_offsets: Sequence[int],
        local_offsets: Sequence[int],
        lengths: Sequence[int],
    ) -> None:
        peer_layer_bases = peer["layer_bases"]
        n_layers = len(peer_layer_bases)

        if self._use_mock:
            import ctypes

            for peer_offset, local_offset, length in zip(
                peer_offsets, local_offsets, lengths
            ):
                for layer in range(n_layers):
                    ctypes.memmove(
                        self._local_layer_bases[layer] + local_offset,
                        int(peer_layer_bases[layer]) + peer_offset,
                        length,
                    )
            return

        deadline = time.monotonic() + 5.0
        while not self._agent.check_remote_metadata(peer["handle"]):
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"timed out waiting for remote metadata of peer {peer_name!r}"
                )
            time.sleep(self._poll_interval_s)

        for start in range(0, len(peer_offsets), self._max_blocks_per_xfer):
            stop = min(start + self._max_blocks_per_xfer, len(peer_offsets))
            local_regions: list[tuple[int, int, int]] = []
            remote_regions: list[tuple[int, int, int]] = []
            for block in range(start, stop):
                for layer in range(n_layers):
                    local_regions.append(
                        (
                            self._local_layer_bases[layer] + local_offsets[block],
                            lengths[block],
                            self._local_device_id,
                        )
                    )
                    remote_regions.append(
                        (
                            int(peer_layer_bases[layer]) + peer_offsets[block],
                            lengths[block],
                            peer["peer_device_id"],
                        )
                    )

            local_descs = self._agent.get_xfer_descs(
                local_regions, mem_type=self._local_mem_type
            )
            remote_descs = self._agent.get_xfer_descs(
                remote_regions, mem_type=peer["peer_mem_type"]
            )
            self._run_xfer(peer_name, peer["handle"], local_descs, remote_descs)

    def _run_xfer(
        self,
        peer_name: str,
        peer_handle: str,
        local_descs: Any,
        remote_descs: Any,
    ) -> None:
        xfer_handle = self._agent.initialize_xfer(
            "READ", local_descs, remote_descs, peer_handle
        )
        transfer_error: BaseException | None = None
        try:
            state = self._agent.transfer(xfer_handle)
            while state == "PROC":
                time.sleep(self._poll_interval_s)
                state = self._agent.check_xfer_state(xfer_handle)
            if state != "DONE":
                raise RuntimeError(
                    f"NIXL transfer for peer {peer_name!r} ended in state {state!r}"
                )
        except BaseException as exc:
            transfer_error = exc
            raise
        finally:
            try:
                self._agent.release_xfer_handle(xfer_handle)
            except Exception:
                if transfer_error is None:
                    raise
                logger.exception(
                    "RemoteG2: release_xfer_handle failed after transfer error "
                    "for peer %r",
                    peer_name,
                )


# --- mock transport helpers ---


class _MockAgent:
    """Minimal nixl_agent stand-in for the byte-copy transport.

    Only the surface the source REP loop uses is implemented:
    ``add_remote_agent`` (no-op, returns the agent name string).
    """

    def __init__(self, name: str, base_ptr: int) -> None:
        self.name = name
        self.base_ptr = int(base_ptr)

    def add_remote_agent(self, peer_metadata: bytes) -> str:
        # Mock metadata is "mock:<name>" — return the decoded peer name
        # so callers can match the symbolic key in their bookkeeping.
        try:
            return peer_metadata.decode("utf-8").split(":", 1)[1]
        except Exception:
            return ""

    @staticmethod
    def encode_metadata(agent_name: str) -> bytes:
        return f"mock:{agent_name}".encode()


def _build_mock_source_bundle(
    agent_name: str,
    layer_pool_base_ptrs: list[int],
    layer_pool_size_bytes: list[int],
    page_size_bytes: int,
    source_generation: int,
) -> NixlSourceBundle:
    agent = _MockAgent(agent_name, int(layer_pool_base_ptrs[0]))
    return NixlSourceBundle(
        agent=agent,
        source_generation=source_generation,
        remote_name=agent_name,
        agent_desc=_MockAgent.encode_metadata(agent_name),
        pool_base_ptr=int(layer_pool_base_ptrs[0]),
        pool_size_bytes=int(layer_pool_size_bytes[0]),
        layer_pool_base_ptrs=[int(p) for p in layer_pool_base_ptrs],
        layer_pool_size_bytes=[int(s) for s in layer_pool_size_bytes],
        page_size_bytes=int(page_size_bytes),
    )
