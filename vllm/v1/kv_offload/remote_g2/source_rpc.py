# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-side ZMQ REP loop for Remote G2 RPCs.

Wire protocol matches the TRT-LLM POC's ``remote_g2_source_setup.py``
(``start_remote_g2_rep_loop``) **byte for byte** so the dynamo parent's
``source_rpc_server.py`` bridge works against both engines without
backend-specific branches.

* Encoding: ``pickle`` (the two ends are colocated on the same host;
  pickle is safe here per the TRT-LLM POC's reasoning).
* Socket: Unix domain socket at
  ``/tmp/dynamo_remote_g2_ipc_<dynamo_pid>.sock`` (PID = the dynamo
  parent that spawned the engine).
* Request frame: ``{"method": <name>, "payload": <dict>}``.
* Response frame: ``{"ok": bool, "result": <obj>}`` or
  ``{"ok": False, "error": <str>}``.
* Methods: ``resolve_and_lease``, ``release_lease``, ``get_metadata``.

``get_metadata`` returns ``"nixl_source_bundle_not_ready"`` until M3
plugs the NIXL adapter in; both bridge sides interpret that as "no
bundle yet, retry on next plan" so the gating is graceful.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pickle
import threading
from collections.abc import Mapping
from typing import Any, Protocol

import pybase64 as base64
import zmq

from vllm.logger import init_logger
from vllm.v1.kv_offload.remote_g2.data_model import (
    RemoteG2Descriptor,
    RemoteG2ResolveResult,
    SourceG2DescriptorRegistry,
)

logger = init_logger(__name__)

# TRT-LLM-compatible RPC method names.
_METHOD_RESOLVE = "resolve_and_lease"
_METHOD_RELEASE = "release_lease"
_METHOD_METADATA = "get_metadata"
# vLLM-side extension (not in the TRT-LLM POC). Debug/observability;
# safe to ignore from peers that don't know about it.
_METHOD_STATS = "stats"


def _result_to_dict(result: RemoteG2ResolveResult) -> dict[str, Any]:
    return {
        "lease_id": result.lease_id,
        "descriptors": [dataclasses.asdict(d) for d in result.descriptors],
        "num_tokens": result.num_tokens,
        "reason": result.reason,
        "source_generation": result.source_generation,
        "per_block_status": [dataclasses.asdict(s) for s in result.per_block_status],
    }


def _descriptor_to_dict(descriptor: RemoteG2Descriptor) -> dict[str, Any]:
    return dataclasses.asdict(descriptor)


class NixlSourceBundle(Protocol):
    """What the M3 NIXL adapter must hand to the RPC loop.

    Field names mirror ``remote_g2_source_setup.NixlSourceBundle`` so the
    target side decodes the bundle the same way regardless of backend.
    The new ``layer_pool_base_ptrs`` / ``layer_pool_size_bytes`` /
    ``page_size_bytes`` fields are vLLM-side additions for the
    multi-layer per-tensor KV layout; TRT-LLM peers that don't know
    about them are unaffected (they only need ``pool_base_ptr`` /
    ``pool_size_bytes`` from the legacy single-pool shape).
    """

    source_generation: int
    remote_name: str
    agent_desc: bytes
    pool_base_ptr: int
    pool_size_bytes: int
    layer_pool_base_ptrs: list[int]
    layer_pool_size_bytes: list[int]
    page_size_bytes: int

    @property
    def agent(self) -> Any: ...


def default_socket_path(dynamo_pid: int | None = None) -> str:
    pid = dynamo_pid if dynamo_pid is not None else os.getppid()
    return f"/tmp/dynamo_remote_g2_ipc_{pid}.sock"


def zmq_endpoint(socket_path: str) -> str:
    """Resolve a socket path to a ZMQ endpoint.

    A bare path is a host-local Unix-domain (ipc) socket; a path that already
    carries a scheme (e.g. ``tcp://0.0.0.0:5555``) is used verbatim. The latter
    is what cross-node 2-pod runs need, where the target connects to the
    source's TCP endpoint over the network.
    """
    return socket_path if "://" in socket_path else f"ipc://{socket_path}"


class SourceG2RpcServer:
    """ZMQ REP server thread serving the TRT-LLM-compatible RPC protocol.

    The NIXL bundle is supplied lazily via ``set_nixl_bundle_provider``
    so the server starts before the worker has finished registering KV
    caches with NIXL. ``get_metadata`` returns ``nixl_source_bundle_not_ready``
    until the provider yields a non-None bundle.
    """

    def __init__(
        self,
        registry: SourceG2DescriptorRegistry,
        *,
        socket_path: str | None = None,
        recv_timeout_ms: int = 200,
        manager: Any = None,
    ) -> None:
        self._registry = registry
        # Optional reference to the local manager so the ``stats`` RPC
        # can report target-side plan counters. None on pure source-only
        # configurations.
        self._manager = manager
        self._socket_path = socket_path or default_socket_path()
        self._recv_timeout_ms = recv_timeout_ms
        self._ctx: zmq.Context | None = None
        self._sock: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._bundle_provider: Any = None
        # Per-rank provider used by the TP>1 scheduler-coordinator path.
        # ``get_metadata(tp_rank=N)`` consults this first; if it returns
        # a bundle, use it. Otherwise fall back to ``_bundle_provider``
        # (TP=1 path).
        self._per_rank_bundle_provider: Any = None

    @property
    def socket_path(self) -> str:
        return self._socket_path

    def set_nixl_bundle_provider(self, provider: Any) -> None:
        """Register a zero-arg callable returning a NixlSourceBundle.

        Called once the NIXL agent + registered pool are ready (M3).
        The provider may return ``None`` if the bundle is still pending.
        """
        self._bundle_provider = provider

    def set_per_rank_bundle_provider(self, provider: Any) -> None:
        """Register a callable ``provider(tp_rank: int) -> bundle | None``.

        Used by the TP>1 scheduler-coordinator path: the scheduler
        caches per-rank NIXL agent payloads via the handshake mechanism
        and exposes them through this provider. ``get_metadata`` reads
        the ``tp_rank`` field from the request payload and calls the
        provider with it. Falls back to ``_bundle_provider`` when
        ``tp_rank`` is missing or unknown.
        """
        self._per_rank_bundle_provider = provider

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("SourceG2RpcServer is already running")
        if "://" not in self._socket_path:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self._socket_path)
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        self._sock.bind(zmq_endpoint(self._socket_path))
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"remote-g2-rpc[{self._socket_path}]",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "RemoteG2 source RPC bound at %s (source_worker_id=%d, source_dp_rank=%d)",
            zmq_endpoint(self._socket_path),
            self._registry.source_worker_id,
            self._registry.source_dp_rank,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:
                logger.warning("Failed closing RemoteG2 RPC socket", exc_info=True)
            self._sock = None
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self._socket_path)

    def _run(self) -> None:
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                raw = self._sock.recv(copy=True)
            except zmq.error.Again:
                continue
            except zmq.error.ZMQError as exc:
                if self._stop_event.is_set():
                    return
                logger.warning("RemoteG2 RPC recv error: %s", exc)
                continue

            try:
                req = pickle.loads(raw)
                response = self._dispatch(req)
            except Exception as exc:
                logger.exception("RemoteG2 RPC dispatch failure")
                response = {"ok": False, "error": repr(exc)}

            try:
                self._sock.send(pickle.dumps(response))
            except zmq.error.ZMQError as exc:
                logger.warning("RemoteG2 RPC send error: %s", exc)

    def _dispatch(self, req: Mapping[str, Any]) -> dict[str, Any]:
        method = req.get("method")
        payload = req.get("payload") or {}
        if method == _METHOD_RESOLVE:
            return self._handle_resolve(payload)
        if method == _METHOD_RELEASE:
            return self._handle_release(payload)
        if method == _METHOD_METADATA:
            return self._handle_metadata(payload)
        if method == _METHOD_STATS:
            return self._handle_stats(payload)
        return {"ok": False, "error": f"unknown method: {method!r}"}

    def _handle_stats(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        sample_limit = int(payload.get("sample_limit", 5))
        with self._registry._lock:
            records = list(self._registry._records.items())
        target_stats: dict[str, int] = {}
        for attr in (
            "plan_seen_count",
            "plan_resolved_count",
            "plan_load_specs_emitted",
            "plan_blocks_loaded",
            "plan_loads_completed",
            "plan_bytes_completed",
        ):
            target_stats[attr] = int(getattr(self._registry, attr, 0))
        return {
            "ok": True,
            "result": {
                "descriptor_count": len(records),
                "lease_count": len(self._registry._leases),
                "source_worker_id": self._registry.source_worker_id,
                "source_dp_rank": self._registry.source_dp_rank,
                "source_generation": self._registry.source_generation,
                "sample_block_hashes": [h for h, _ in records[:sample_limit]],
                "target_stats": target_stats,
                "pin_count_total": int(self._registry.pin_count_total),
                "unpin_count_total": int(self._registry.unpin_count_total),
                "pin_failures": int(self._registry.pin_failures),
                "transport_backend": str(
                    getattr(self._registry, "transport_backend", "unset")
                ),
                "transport_mock": bool(getattr(self._registry, "transport_mock", True)),
                "num_layers": int(getattr(self._registry, "num_layers", 0)),
                "hash_to_key_count": len(self._registry._hash_to_key),
                "policy_registered": self._registry._policy is not None,
            },
        }

    def _handle_resolve(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        plan = payload.get("plan")
        if not isinstance(plan, Mapping):
            return {"ok": False, "error": "missing_plan"}
        result = self._registry.resolve_and_lease(plan)
        return {"ok": True, "result": _result_to_dict(result)}

    def _handle_release(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        lease_id = payload.get("lease_id")
        if not isinstance(lease_id, str):
            return {"ok": False, "error": "missing_lease_id"}
        released = self._registry.release_lease(
            lease_id, reason=str(payload.get("reason", "ack"))
        )
        return {"ok": True, "result": released}

    def _handle_metadata(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        # TP>1 scheduler-coordinator path: caller passes ``tp_rank`` and
        # we return that rank's cached NIXL agent payload. TP=1 path:
        # ``tp_rank`` is absent and we fall back to the legacy single
        # bundle provider.
        tp_rank = payload.get("tp_rank")
        bundle = None
        if tp_rank is not None and self._per_rank_bundle_provider is not None:
            try:
                bundle = self._per_rank_bundle_provider(int(tp_rank))
            except Exception:
                logger.exception(
                    "RemoteG2 per-rank bundle provider raised (tp_rank=%s)",
                    tp_rank,
                )
                bundle = None
        if bundle is None and self._bundle_provider is not None:
            bundle = self._bundle_provider()
        if bundle is None:
            return {"ok": False, "error": "nixl_source_bundle_not_ready"}

        peer_metadata_b64 = payload.get("peer_metadata_b64")
        if peer_metadata_b64:
            try:
                peer_bytes = base64.b64decode(peer_metadata_b64)
                # The TP>1 wrapped bundle has no live ``agent`` (the
                # agent lives in the worker process); skip the
                # add_remote_agent step for that case.
                agent = getattr(bundle, "agent", None)
                if agent is not None:
                    loaded_name = agent.add_remote_agent(peer_bytes)
                    logger.info(
                        "RemoteG2 source add_remote_agent loaded peer "
                        "name=%s (bytes=%d)",
                        loaded_name,
                        len(peer_bytes),
                    )
            except Exception:
                logger.exception("RemoteG2 source add_remote_agent failed")

        return {
            "ok": True,
            "result": {
                "source_worker_id": self._registry.source_worker_id,
                "source_dp_rank": self._registry.source_dp_rank,
                "source_generation": bundle.source_generation,
                "remote_name": bundle.remote_name,
                "agent_metadata_b64": base64.b64encode(bundle.agent_desc).decode(
                    "ascii"
                ),
                # Legacy single-pool fields (= layer 0 in multi-layer
                # mode). Kept for TRT-LLM-compatible peers.
                "pool_base_ptr": bundle.pool_base_ptr,
                "pool_size_bytes": bundle.pool_size_bytes,
                # vLLM multi-layer: per-layer base pointers + sizes +
                # a uniform per-block page size. Peers that know about
                # multi-layer use these; legacy peers ignore them.
                "layer_pool_base_ptrs": list(bundle.layer_pool_base_ptrs),
                "layer_pool_size_bytes": list(bundle.layer_pool_size_bytes),
                "page_size_bytes": int(bundle.page_size_bytes),
            },
        }

    # Convenience for direct metadata lookups (vLLM-internal use, not
    # part of the TRT-LLM-compatible wire surface).
    def lookup_descriptor(self, block_hash: int) -> dict[str, Any] | None:
        record = self._registry.get_descriptor(int(block_hash))
        if record is None or not record.live:
            return None
        return _descriptor_to_dict(
            RemoteG2Descriptor(
                block_hash=int(record.block_hash),
                descriptor_generation=record.descriptor_generation,
                pool_id=record.pool_id,
                byte_offset=record.byte_offset,
                byte_length=record.byte_length,
                metadata=dict(record.metadata),
            )
        )
