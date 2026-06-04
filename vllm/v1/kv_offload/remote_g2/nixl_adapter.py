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
    reach its host pool. Same shape as TRT-LLM POC's bundle.

    ``agent`` is the raw nixl_agent (or a mock-transport handle) — used
    by the source REP loop to call ``add_remote_agent`` when peers send
    their metadata in the ``peer_metadata_b64`` payload.
    """

    agent: Any
    source_generation: int
    remote_name: str
    agent_desc: bytes
    pool_base_ptr: int
    pool_size_bytes: int


def build_source_agent(
    agent_name: str,
    pool_base_ptr: int,
    pool_size_bytes: int,
    *,
    source_generation: int = 1,
    backends: tuple[str, ...] = ("UCX",),
    mem_type: str = "DRAM",
    device_id: int = 0,
) -> NixlSourceBundle | None:
    """Construct the source NIXL agent and register the host pool.

    Returns ``None`` on failure. When ``backends`` is empty OR NIXL is
    not importable, falls back to the byte-copy mock transport so the
    rest of the pipeline can still be exercised in CI.
    """
    if not NIXL_AVAILABLE or not backends:
        if not NIXL_AVAILABLE:
            logger.warning(
                "RemoteG2: NIXL python module not available; using mock "
                "transport (ctypes.memmove). Wire protocol unchanged so "
                "peers can negotiate without code changes."
            )
        return _build_mock_source_bundle(
            agent_name, pool_base_ptr, pool_size_bytes, source_generation
        )

    try:
        # Positional args mirror the nixl examples: (enable_prog_thread,
        # enable_listen_thread, listen_port, capture_telemetry,
        # num_threads, backends).
        # Mirror the NIXL basic_two_peers example: enable_prog_thread=True,
        # enable_listen_thread=True, listen_port=0 (auto-assigned).
        config = nixl_agent_config(True, True, 0, False, 0, list(backends))
        agent = nixl_agent(agent_name, config, instantiate_all=False)
    except Exception:
        logger.exception("RemoteG2: nixl_agent construction failed")
        return None

    try:
        # 4-tuple registration list: (addr, size, device_id, "extra").
        agent.register_memory(
            [(pool_base_ptr, pool_size_bytes, device_id, "")], mem_type=mem_type
        )
    except Exception:
        logger.exception(
            "RemoteG2: register_memory failed for pool 0x%x size=%d",
            pool_base_ptr,
            pool_size_bytes,
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
        pool_base_ptr=pool_base_ptr,
        pool_size_bytes=pool_size_bytes,
    )


class RawNixlRemoteG2Adapter:
    """Target-side adapter performing block-granular NIXL READs.

    Idempotent per-peer setup: the first time a peer is seen, the
    adapter calls ``add_remote_agent`` and ``prep_xfer_dlist`` against
    the peer's whole pool (so block-indexed READs can subset cheaply).
    ``read_block`` then performs the actual transfer for one block.

    For the mock transport, transfers are bytes-level memcpy via a
    shared ``MockPeerRegistry`` keyed by ``remote_name``.
    """

    def __init__(
        self,
        agent_name: str,
        local_pool_base_ptr: int,
        local_pool_size_bytes: int,
        *,
        backends: tuple[str, ...] = ("UCX",),
        use_mock: bool = False,
        local_mem_type: str = "VRAM",
        local_device_id: int = 0,
        poll_interval_s: float = 0.0005,
    ) -> None:
        self._agent_name = agent_name
        self._local_base = int(local_pool_base_ptr)
        self._local_size = int(local_pool_size_bytes)
        self._local_mem_type = local_mem_type
        self._local_device_id = int(local_device_id)
        self._poll_interval_s = float(poll_interval_s)
        self._use_mock = use_mock or not NIXL_AVAILABLE or not backends
        self._peers: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

        if self._use_mock:
            self._agent: Any = _MockAgent(agent_name, local_pool_base_ptr)
            self._agent_metadata = _MockAgent.encode_metadata(agent_name)
            return

        # Mirror the NIXL basic_two_peers example: enable_prog_thread=True,
        # enable_listen_thread=True, listen_port=0 (auto-assigned).
        config = nixl_agent_config(True, True, 0, False, 0, list(backends))
        self._agent = nixl_agent(agent_name, config, instantiate_all=False)
        self._agent.register_memory(
            [(self._local_base, self._local_size, self._local_device_id, "")],
            mem_type=self._local_mem_type,
        )
        self._agent_metadata = self._agent.get_agent_metadata()

    @property
    def agent_metadata(self) -> bytes:
        return self._agent_metadata

    def add_peer(
        self,
        peer_name: str,
        peer_agent_metadata: bytes,
        peer_pool_base_ptr: int,
        peer_pool_size_bytes: int,
        *,
        peer_mem_type: str = "DRAM",
        peer_device_id: int = 0,
    ) -> None:
        """Idempotent: register the source peer's pool for later READs."""
        with self._lock:
            if peer_name in self._peers:
                return
            if self._use_mock:
                self._peers[peer_name] = {
                    "base_ptr": int(peer_pool_base_ptr),
                    "size": int(peer_pool_size_bytes),
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
                "peer_base": int(peer_pool_base_ptr),
                "peer_size": int(peer_pool_size_bytes),
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
        """Synchronous block-sized READ from peer's pool.

        For real NIXL: descriptors are built per-call via
        ``get_xfer_descs`` and the transfer is initialised with
        ``initialize_xfer``. M4 can switch to ``prep_xfer_dlist`` +
        ``make_prepped_xfer`` once the per-block dlist layout is fixed.
        """
        with self._lock:
            peer = self._peers.get(peer_name)
        if peer is None:
            raise RuntimeError(f"peer {peer_name!r} not registered; call add_peer")

        if self._use_mock:
            import ctypes

            ctypes.memmove(
                self._local_base + local_byte_offset,
                int(peer["base_ptr"]) + peer_byte_offset,
                byte_length,
            )
            return

        local_descs = self._agent.get_xfer_descs(
            [(self._local_base + local_byte_offset, byte_length, self._local_device_id)],
            mem_type=self._local_mem_type,
        )
        remote_descs = self._agent.get_xfer_descs(
            [
                (
                    peer["peer_base"] + peer_byte_offset,
                    byte_length,
                    peer["peer_device_id"],
                )
            ],
            mem_type=peer["peer_mem_type"],
        )

        # Wait until remote metadata is loaded on the agent (loaded
        # synchronously by add_remote_agent under the hood, but
        # check_remote_metadata polls UCX readiness).
        deadline = time.monotonic() + 5.0
        while not self._agent.check_remote_metadata(peer["handle"]):
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"timed out waiting for remote metadata of peer "
                    f"{peer_name!r}"
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
        return f"mock:{agent_name}".encode("utf-8")


def _build_mock_source_bundle(
    agent_name: str,
    pool_base_ptr: int,
    pool_size_bytes: int,
    source_generation: int,
) -> NixlSourceBundle:
    agent = _MockAgent(agent_name, pool_base_ptr)
    return NixlSourceBundle(
        agent=agent,
        source_generation=source_generation,
        remote_name=agent_name,
        agent_desc=_MockAgent.encode_metadata(agent_name),
        pool_base_ptr=int(pool_base_ptr),
        pool_size_bytes=int(pool_size_bytes),
    )
