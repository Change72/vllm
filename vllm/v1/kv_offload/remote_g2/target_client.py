# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target-side RPC client for Remote G2.

Speaks the same pickle wire format as the engine-side REP server
(``source_rpc.SourceG2RpcServer``). Used by the target's manager /
transfer handler to resolve a plan against the source's registry and
release leases when transfers finish.

In production the target reaches a remote source through the dynamo
runtime ``client.direct(...)`` path (see
``dynamo.trtllm.kv_p2p.target_rpc_client``). For the same-host POC
smoke test, we bypass the dynamo network plane and connect directly
to the source's Unix domain socket.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import threading
from collections.abc import Mapping
from typing import Any

import pybase64 as base64
import zmq

from vllm.logger import init_logger
from vllm.v1.kv_offload.remote_g2.data_model import (
    RemoteG2Descriptor,
    RemoteG2ResolveResult,
    RemoteKvReusePlan,
)

logger = init_logger(__name__)

# Method-name translation for the dynamo parent target bridge. The bridge
# (dynamo.*.kv_p2p.target_rpc_local) routes by a short verb and then calls the
# matching source endpoint via ``client.direct(instance_id=source_worker_id)``.
# Direct (same-host POC) mode keeps the long source-RPC verbs.
_BRIDGE_METHOD = {
    "resolve_and_lease": "resolve",
    "release_lease": "release",
    "get_metadata": "metadata",
}


def target_bridge_socket_path(parent_pid: int | None = None) -> str:
    """Path of the dynamo parent's target-bridge REP socket.

    Mirrors ``source_rpc.default_socket_path``: the dynamo parent binds the
    bridge keyed by its own pid, so the engine subprocess reaches it via its
    parent pid (``os.getppid()``)."""
    pid = parent_pid if parent_pid is not None else os.getppid()
    return f"/tmp/dynamo_remote_g2_target_{pid}.sock"


def _descriptor_from_wire(wire: Mapping[str, Any]) -> RemoteG2Descriptor:
    return RemoteG2Descriptor(
        block_hash=int(wire["block_hash"]),
        descriptor_generation=int(wire["descriptor_generation"]),
        pool_id=str(wire["pool_id"]),
        byte_offset=int(wire["byte_offset"]),
        byte_length=int(wire["byte_length"]),
        metadata=dict(wire.get("metadata", {})),
    )


def _result_from_wire(wire: Mapping[str, Any]) -> RemoteG2ResolveResult:
    return RemoteG2ResolveResult(
        lease_id=wire.get("lease_id"),
        descriptors=tuple(
            _descriptor_from_wire(d) for d in wire.get("descriptors", ())
        ),
        num_tokens=int(wire.get("num_tokens", 0)),
        reason=str(wire.get("reason", "")),
        source_generation=int(wire.get("source_generation", 0)),
        per_block_status=tuple(),  # not used by the target after resolve
    )


class TargetG2RpcClient:
    """Thread-safe synchronous client for the source REP server.

    A single ZMQ REQ socket per source endpoint. REQ/REP is strictly
    lockstep so concurrent callers are serialized by ``self._lock``.
    POC throughput is well below the saturation point of a single
    in-memory IPC round-trip.
    """

    def __init__(
        self,
        source_socket_path: str,
        *,
        timeout_ms: int = 5000,
        via_bridge: bool = False,
        source_worker_id: int | None = None,
    ) -> None:
        """Connect to the source's RPC.

        Two transports:
        * Direct (default, same-host POC): ``source_socket_path`` is the
          source engine's own REP socket and the long source-RPC verbs are
          sent verbatim.
        * Via bridge (``via_bridge=True``): ``source_socket_path`` is the
          dynamo parent's target-bridge REP socket. ``source_worker_id`` is
          injected into every payload and the verbs are mapped to the
          bridge's short verbs so the parent can ``client.direct(...)`` to the
          right source worker — this is the cross-node / >2-worker path.
        """
        self._socket_path = source_socket_path
        self._timeout_ms = int(timeout_ms)
        self._via_bridge = bool(via_bridge)
        self._source_worker_id = (
            int(source_worker_id) if source_worker_id is not None else None
        )
        if self._via_bridge and self._source_worker_id is None:
            raise ValueError("via_bridge=True requires source_worker_id")
        self._ctx = zmq.Context.instance()
        self._sock: zmq.Socket | None = None
        self._lock = threading.Lock()

    def _ensure_socket(self) -> zmq.Socket:
        if self._sock is None:
            sock = self._ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
            sock.setsockopt(zmq.LINGER, 0)
            # A bare path is a host-local ipc socket; a path with a scheme
            # (e.g. tcp://host:port) is used verbatim for cross-node 2-pod runs.
            endpoint = (
                self._socket_path
                if "://" in self._socket_path
                else f"ipc://{self._socket_path}"
            )
            sock.connect(endpoint)
            self._sock = sock
        return self._sock

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                with contextlib.suppress(Exception):
                    self._sock.close(linger=0)
                self._sock = None

    def _request(self, method: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        wire_payload = dict(payload)
        if self._via_bridge:
            # The bridge routes by a short verb and needs to know which source
            # worker to client.direct() to. ``stats`` has no bridge verb (it is
            # a same-host debug call only) so it is sent through unmapped and
            # will fail loudly if ever attempted via the bridge.
            method = _BRIDGE_METHOD.get(method, method)
            wire_payload["source_worker_id"] = self._source_worker_id
        with self._lock:
            sock = self._ensure_socket()
            try:
                sock.send(pickle.dumps({"method": method, "payload": wire_payload}))
                raw = sock.recv()
            except zmq.error.ZMQError as exc:
                # Any ZMQ error (notably a recv timeout, zmq.error.Again, but
                # also a failed send) leaves the REQ socket stuck in the wrong
                # send/recv state (EFSM). Close it so the next call rebuilds.
                with contextlib.suppress(Exception):
                    sock.close(linger=0)
                self._sock = None
                if isinstance(exc, zmq.error.Again):
                    raise TimeoutError(
                        f"RemoteG2 {method} timed out after {self._timeout_ms} ms"
                    ) from exc
                raise
            return pickle.loads(raw)

    def resolve_and_lease(
        self, plan: RemoteKvReusePlan | Mapping[str, Any]
    ) -> RemoteG2ResolveResult:
        plan_dict: Mapping[str, Any]
        if isinstance(plan, RemoteKvReusePlan):
            plan_dict = {
                "plan_id": plan.plan_id,
                "request_id": plan.request_id,
                "target_worker_id": plan.target_worker_id,
                "target_dp_rank": plan.target_dp_rank,
                "source_worker_id": plan.source_worker_id,
                "source_dp_rank": plan.source_dp_rank,
                "source_tier": plan.source_tier,
                "block_hashes": list(plan.block_hashes),
                "kv_block_hashes": list(plan.kv_block_hashes),
                "start_block_index": plan.start_block_index,
                "planned_prefix_blocks": plan.planned_prefix_blocks,
                "block_size_tokens": plan.block_size_tokens,
                "created_at_ms": plan.created_at_ms,
                "expires_at_ms": plan.expires_at_ms,
                "plan_version": plan.plan_version,
            }
        else:
            plan_dict = plan
        reply = self._request("resolve_and_lease", {"plan": plan_dict})
        if not reply.get("ok"):
            return RemoteG2ResolveResult(
                None, (), 0, str(reply.get("error", "rpc_error")), 0
            )
        return _result_from_wire(reply["result"])

    def release_lease(self, lease_id: str, reason: str = "ack") -> bool:
        reply = self._request("release_lease", {"lease_id": lease_id, "reason": reason})
        if not reply.get("ok"):
            logger.warning("RemoteG2 release_lease failed: %s", reply.get("error"))
            return False
        return bool(reply.get("result"))

    def stats(self, sample_limit: int = 5) -> dict[str, Any] | None:
        """Source-side descriptor / lease counts. Useful for tests
        verifying that the publish path is alive. ``sample_limit``
        caps the size of ``sample_block_hashes`` (default 5)."""
        reply = self._request("stats", {"sample_limit": int(sample_limit)})
        if not reply.get("ok"):
            return None
        return dict(reply["result"])

    def get_metadata(
        self,
        peer_agent_metadata: bytes | None = None,
        *,
        tp_rank: int | None = None,
    ) -> dict[str, Any] | None:
        """Fetch the source's NIXL bundle and, optionally, hand it our
        local agent metadata so it can call ``add_remote_agent`` first.

        For TP>1 callers, ``tp_rank`` selects which source-side rank's
        NIXL agent metadata to fetch. TP=1 callers may omit it; the
        server falls back to its single bundle.

        Returns ``None`` when the source isn't ready yet (the M2 stub
        reports ``nixl_source_bundle_not_ready`` until M3 plugs the
        NIXL adapter in via ``set_nixl_bundle_provider``).
        """
        payload: dict[str, Any] = {}
        if peer_agent_metadata is not None:
            payload["peer_metadata_b64"] = base64.b64encode(peer_agent_metadata).decode(
                "ascii"
            )
        if tp_rank is not None:
            payload["tp_rank"] = int(tp_rank)
        reply = self._request("get_metadata", payload)
        if not reply.get("ok"):
            logger.debug("RemoteG2 get_metadata not ready: %s", reply.get("error"))
            return None
        result = dict(reply["result"])
        if isinstance(result.get("agent_metadata_b64"), str):
            result["agent_metadata"] = base64.b64decode(result["agent_metadata_b64"])
        return result
