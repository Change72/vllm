# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw NIXL agent wrapper for Remote G2 transfers.

Mirrors the TRT-LLM ``remote_g2_raw_nixl_adapter.py`` shape:

* Source side: ``build_source_agent`` — construct a ``nixl_agent``,
  register the host pool, return a ``NixlSourceBundle`` (agent +
  metadata bytes + pool base/size) that ``SourceG2RpcServer.get_metadata``
  hands to peers.
* Target side: ``RawNixlRemoteG2Adapter`` — construct an agent, register
  the local GPU pool, on first contact with each peer call
  ``add_remote_agent`` and ``prep_xfer_dlist`` for the source's pool,
  then ``make_prepped_xfer`` for each READ.

When the ``nixl`` python module is not installed, ``NIXL_AVAILABLE`` is
``False`` and the adapter exposes a *mock* transport that copies bytes
through a /dev/shm-backed channel; the upper layers see the same call
shape so the source/target dataflow can be exercised end-to-end in CI
without NIXL or a GPU.
"""

from __future__ import annotations

import threading
import time
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
    """Target-side adapter performing per-layer NIXL READs.

    vLLM v1 allocates one CPU tensor per transformer layer (e.g. 36 for
    Qwen3-8B) and similarly one GPU tensor per layer for the target's
    KV cache. The adapter takes the *list* of per-layer base pointers
    on each side, registers all of them with NIXL in a single
    ``register_memory`` call, and on each ``read_block`` issues one
    batched NIXL transfer whose source/destination descriptor list has
    ``num_layers`` entries — one per layer at the requested block
    offset. A single ``initialize_xfer`` + ``transfer`` + completion
    poll covers the whole block, including all its layers.

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
        self._peers: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        # Fault injection: when >0, every Nth read_block call raises
        # RuntimeError. Used by the failure-recovery test suite to
        # confirm transfer_handler propagates failures to vLLM's
        # kv_load_failure_policy. Defaults to 0 (off).
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
        """Synchronous block-sized READ across **all** layers.

        Builds a transfer descriptor list with one entry per layer (the
        peer's layer pool plus its block byte offset, paired with the
        local layer pool plus the local block byte offset). Issues one
        ``initialize_xfer`` / ``transfer`` / completion poll covering
        the full block.
        """
        with self._lock:
            peer = self._peers.get(peer_name)
            self._read_block_calls += 1
            inject = (
                self._fault_inject_every > 0
                and self._read_block_calls % self._fault_inject_every == 0
            )
        if peer is None:
            raise RuntimeError(f"peer {peer_name!r} not registered; call add_peer")

        peer_layer_bases = peer["layer_bases"]
        n_layers = len(peer_layer_bases)
        if n_layers != self.num_layers:
            raise RuntimeError(
                f"peer {peer_name!r} layer count {n_layers} mismatches "
                f"local {self.num_layers}"
            )

        if inject:
            raise RuntimeError(
                f"RemoteG2 fault injection: read_block #{self._read_block_calls}"
            )

        if self._use_mock:
            import ctypes

            for i in range(n_layers):
                ctypes.memmove(
                    self._local_layer_bases[i] + local_byte_offset,
                    int(peer_layer_bases[i]) + peer_byte_offset,
                    byte_length,
                )
            return

        local_descs = self._agent.get_xfer_descs(
            [
                (
                    self._local_layer_bases[i] + local_byte_offset,
                    byte_length,
                    self._local_device_id,
                )
                for i in range(n_layers)
            ],
            mem_type=self._local_mem_type,
        )
        remote_descs = self._agent.get_xfer_descs(
            [
                (
                    int(peer_layer_bases[i]) + peer_byte_offset,
                    byte_length,
                    peer["peer_device_id"],
                )
                for i in range(n_layers)
            ],
            mem_type=peer["peer_mem_type"],
        )

        deadline = time.monotonic() + 5.0
        while not self._agent.check_remote_metadata(peer["handle"]):
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"timed out waiting for remote metadata of peer {peer_name!r}"
                )
            time.sleep(self._poll_interval_s)

        xfer_handle = self._agent.initialize_xfer(
            "READ", local_descs, remote_descs, peer["handle"]
        )
        state = self._agent.transfer(xfer_handle)
        if state == "ERR":
            self._agent.release_xfer_handle(xfer_handle)
            raise RuntimeError(f"NIXL transfer post failed for peer {peer_name!r}")
        while True:
            state = self._agent.check_xfer_state(xfer_handle)
            if state == "ERR":
                self._agent.release_xfer_handle(xfer_handle)
                raise RuntimeError(
                    f"NIXL transfer entered ERR state for peer {peer_name!r}"
                )
            if state == "DONE":
                break
            time.sleep(self._poll_interval_s)
        self._agent.release_xfer_handle(xfer_handle)


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
