# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire-protocol tests for the dynamic (dynamo parent-bridge) target transport.

These pin the translation the target client performs when ``via_bridge`` is on:
the source-RPC verbs map to the bridge's short verbs, ``source_worker_id`` is
injected into every payload, and the vLLM NIXL-handshake fields
(``peer_metadata_b64`` + ``tp_rank``) are forwarded. No real ZMQ round-trip: a
fake socket captures the pickled request and returns a canned reply.
"""

import pickle

import pybase64 as base64
import pytest

zmq = pytest.importorskip("zmq")

from vllm.v1.kv_offload.remote_g2.spec import _resolve_bool, _truthy  # noqa: E402
from vllm.v1.kv_offload.remote_g2.target_client import (  # noqa: E402
    _BRIDGE_METHOD,
    TargetG2RpcClient,
    target_bridge_socket_path,
)


@pytest.mark.parametrize(
    "val,expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("1", True),
        ("on", True),
        ("false", False),  # plain bool("false") would be True
        ("0", False),
        ("no", False),
        ("", False),
    ],
)
def test_truthy(val, expected):
    assert _truthy(val) is expected


def test_resolve_bool_string_false_is_false():
    # extra-config precedence: a string "false" must NOT enable the flag.
    assert _resolve_bool({"k": "false"}, "k", "ENV_UNUSED", True) is False
    assert _resolve_bool({"k": True}, "k", "ENV_UNUSED", False) is True


def test_resolve_bool_env_fallback(monkeypatch):
    monkeypatch.delenv("RG2_TEST_BOOL", raising=False)
    assert _resolve_bool({}, "k", "RG2_TEST_BOOL", True) is True  # default
    monkeypatch.setenv("RG2_TEST_BOOL", "off")
    assert _resolve_bool({}, "k", "RG2_TEST_BOOL", True) is False


class _FakeSock:
    """Records the last pickled request and replays a preset reply."""

    def __init__(self, reply: dict) -> None:
        self.reply = reply
        self.sent: dict | None = None

    def send(self, raw: bytes) -> None:
        self.sent = pickle.loads(raw)

    def recv(self) -> bytes:
        return pickle.dumps(self.reply)


def _client_with(reply: dict, **kwargs) -> tuple[TargetG2RpcClient, _FakeSock]:
    client = TargetG2RpcClient("/tmp/unused.sock", **kwargs)
    fake = _FakeSock(reply)
    client._sock = fake  # _ensure_socket() returns it since it is not None
    return client, fake


def _sent(fake: _FakeSock) -> dict:
    """Return the captured request, narrowed to a dict for the type checker."""
    sent = fake.sent
    assert sent is not None, "no request was sent"
    return sent


_RESOLVE_REPLY = {
    "ok": True,
    "result": {
        "lease_id": "L1",
        "descriptors": [],
        "num_tokens": 0,
        "reason": "ok",
        "source_generation": 1,
    },
}


def test_via_bridge_requires_source_worker_id():
    with pytest.raises(ValueError):
        TargetG2RpcClient("/tmp/x.sock", via_bridge=True)


def test_bridge_verb_map_is_the_short_verbs():
    assert _BRIDGE_METHOD["resolve_and_lease"] == "resolve"
    assert _BRIDGE_METHOD["release_lease"] == "release"
    assert _BRIDGE_METHOD["get_metadata"] == "metadata"


def test_bridge_resolve_maps_verb_and_injects_source_worker_id():
    client, fake = _client_with(_RESOLVE_REPLY, via_bridge=True, source_worker_id=42)
    client.resolve_and_lease({"plan_id": "p", "source_worker_id": 42})
    sent = _sent(fake)
    assert sent["method"] == "resolve"  # not resolve_and_lease
    assert sent["payload"]["source_worker_id"] == 42
    assert "plan" in sent["payload"]


def test_bridge_metadata_forwards_peer_metadata_and_tp_rank():
    reply = {"ok": True, "result": {"remote_name": "n", "source_generation": 1}}
    client, fake = _client_with(reply, via_bridge=True, source_worker_id=7)
    client.get_metadata(peer_agent_metadata=b"agentbytes", tp_rank=3)
    sent = _sent(fake)
    assert sent["method"] == "metadata"
    payload = sent["payload"]
    assert payload["source_worker_id"] == 7
    assert payload["tp_rank"] == 3
    assert base64.b64decode(payload["peer_metadata_b64"]) == b"agentbytes"


def test_bridge_release_maps_verb():
    client, fake = _client_with(
        {"ok": True, "result": True}, via_bridge=True, source_worker_id=9
    )
    client.release_lease("L1", reason="ack")
    sent = _sent(fake)
    assert sent["method"] == "release"
    assert sent["payload"]["source_worker_id"] == 9
    assert sent["payload"]["lease_id"] == "L1"


def test_direct_mode_keeps_long_verbs_and_no_source_worker_id():
    client, fake = _client_with(_RESOLVE_REPLY)  # via_bridge defaults False
    client.resolve_and_lease({"plan_id": "p", "source_worker_id": 1})
    sent = _sent(fake)
    assert sent["method"] == "resolve_and_lease"  # unmapped
    assert "source_worker_id" not in sent["payload"]


def test_target_bridge_socket_path_uses_parent_pid():
    assert target_bridge_socket_path(1234) == "/tmp/dynamo_remote_g2_target_1234.sock"
    # No arg -> getppid()-derived, still the canonical shape.
    assert target_bridge_socket_path().startswith("/tmp/dynamo_remote_g2_target_")
