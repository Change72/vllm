# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the opt-in Remote-G2 host-bounce contract.

The first integrated E5 implementation is intentionally narrow.  These tests
pin its configuration guards independently from GPU/NIXL setup and verify the
registry/statistics fields used by the fail-closed performance gate.
"""

from __future__ import annotations

import threading
from typing import Any, cast

import pytest

from vllm.v1.kv_offload.remote_g2.data_model import (
    SourceG2DescriptorRegistry,
)
from vllm.v1.kv_offload.remote_g2.source_rpc import SourceG2RpcServer
from vllm.v1.kv_offload.remote_g2.spec import (
    RemoteG2OffloadingSpec,
    _validate_host_bounce_config,
)

pytestmark = pytest.mark.cpu_test


def _validate(
    *,
    enabled: bool = True,
    tp_size: int = 1,
    use_v2_model_runner: bool = False,
    use_mock_nixl: bool = False,
    block_size_factor: int = 1,
    max_bytes: int = 1 << 30,
    slot_count: int = 2,
) -> None:
    _validate_host_bounce_config(
        enabled=enabled,
        tp_size=tp_size,
        use_v2_model_runner=use_v2_model_runner,
        use_mock_nixl=use_mock_nixl,
        block_size_factor=block_size_factor,
        max_bytes=max_bytes,
        slot_count=slot_count,
    )


@pytest.mark.parametrize(
    (
        "tp_size",
        "use_v2_model_runner",
        "use_mock_nixl",
        "block_size_factor",
        "max_bytes",
    ),
    [
        (2, False, False, 1, 1 << 30),
        (1, True, False, 1, 1 << 30),
        (1, False, True, 1, 1 << 30),
        (1, False, False, 2, 1 << 30),
        (1, False, False, 1, 0),
    ],
)
def test_host_bounce_disabled_is_a_noop(
    tp_size: int,
    use_v2_model_runner: bool,
    use_mock_nixl: bool,
    block_size_factor: int,
    max_bytes: int,
) -> None:
    """Direct-VRAM mode must not inherit host-bounce-only restrictions."""
    _validate(
        enabled=False,
        tp_size=tp_size,
        use_v2_model_runner=use_v2_model_runner,
        use_mock_nixl=use_mock_nixl,
        block_size_factor=block_size_factor,
        max_bytes=max_bytes,
    )


def test_host_bounce_accepts_the_supported_contract() -> None:
    for slot_count in (1, 2):
        _validate(
            tp_size=1,
            use_v2_model_runner=False,
            use_mock_nixl=False,
            block_size_factor=1,
            max_bytes=512 << 20,
            slot_count=slot_count,
        )


@pytest.mark.parametrize(
    (
        "tp_size",
        "use_v2_model_runner",
        "use_mock_nixl",
        "block_size_factor",
        "max_bytes",
        "match",
    ),
    [
        (2, False, False, 1, 1 << 30, "requires TP=1"),
        (1, True, False, 1, 1 << 30, "requires the V1 model runner"),
        (1, False, True, 1, 1 << 30, "requires real NIXL/UCX"),
        (1, False, False, 2, 1 << 30, "requires block_size_factor=1"),
        (1, False, False, 1, 0, "host_bounce_max_bytes must be positive"),
        (1, False, False, 1, -1, "host_bounce_max_bytes must be positive"),
    ],
)
def test_host_bounce_rejects_unsupported_contract(
    tp_size: int,
    use_v2_model_runner: bool,
    use_mock_nixl: bool,
    block_size_factor: int,
    max_bytes: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _validate(
            tp_size=tp_size,
            use_v2_model_runner=use_v2_model_runner,
            use_mock_nixl=use_mock_nixl,
            block_size_factor=block_size_factor,
            max_bytes=max_bytes,
        )


@pytest.mark.parametrize("slot_count", [0, 3])
def test_host_bounce_rejects_unsupported_slot_count(slot_count: int) -> None:
    with pytest.raises(ValueError, match="host_bounce_slot_count must be 1 or 2"):
        _validate(slot_count=slot_count)


def _fresh_registry() -> SourceG2DescriptorRegistry:
    SourceG2DescriptorRegistry._clear_singletons_for_tests()
    return SourceG2DescriptorRegistry.get_or_create(
        source_worker_id=11,
        source_dp_rank=0,
        lease_ttl_ms=10_000,
    )


def _stats(registry: SourceG2DescriptorRegistry) -> dict:
    # No start() means no ZMQ socket is created; _handle_stats is a pure read
    # of the registry state for this test.
    response = SourceG2RpcServer(registry)._handle_stats({"sample_limit": 0})
    assert response["ok"] is True
    return response["result"]


def test_host_bounce_stats_defaults_are_fail_closed() -> None:
    registry = _fresh_registry()

    result = _stats(registry)

    assert registry.transport_mode == "unset"
    assert result["transport_mode"] == "unset"
    assert result["target_stats"]["host_bounce_jobs_completed"] == 0
    assert result["target_stats"]["host_bounce_bytes_completed"] == 0
    assert result["target_stats"]["host_bounce_failures"] == 0


@pytest.mark.parametrize("mode", ["direct_vram", "host_bounce"])
def test_transport_mode_is_surfaced_by_stats(mode: str) -> None:
    registry = _fresh_registry()
    registry.set_transport_info(backend="UCX", mock=False, mode=mode)

    result = _stats(registry)

    assert result["transport_backend"] == "UCX"
    assert result["transport_mock"] is False
    assert result["transport_mode"] == mode


def test_host_bounce_terminal_counters_are_surfaced() -> None:
    registry = _fresh_registry()
    registry.set_transport_info(backend="UCX", mock=False, mode="host_bounce")

    registry.record_host_bounce_result(success=True, num_bytes=4096)
    registry.record_host_bounce_result(success=True, num_bytes=8192)
    registry.record_host_bounce_result(success=False, num_bytes=0)

    result = _stats(registry)
    target = result["target_stats"]
    assert target["host_bounce_jobs_completed"] == 2
    assert target["host_bounce_bytes_completed"] == 12_288
    assert target["host_bounce_failures"] == 1
    assert result["transport_mode"] == "host_bounce"


def test_host_bounce_failure_does_not_claim_completion_or_bytes() -> None:
    registry = _fresh_registry()

    registry.record_host_bounce_result(success=False, num_bytes=123_456)

    target = _stats(registry)["target_stats"]
    assert target["host_bounce_jobs_completed"] == 0
    assert target["host_bounce_bytes_completed"] == 0
    assert target["host_bounce_failures"] == 1


def test_host_bounce_peer_page_mismatch_rejected_before_add_peer() -> None:
    class _Adapter:
        agent_metadata = b"target"

        def __init__(self) -> None:
            self.add_calls = 0

        def add_peer(self, *args, **kwargs) -> None:
            self.add_calls += 1

    class _Client:
        def get_metadata(self, **kwargs):
            return {
                "page_size_bytes": 2048,
                "agent_metadata": b"source",
                "layer_pool_base_ptrs": [1000],
                "layer_pool_size_bytes": [8192],
            }

    spec = object.__new__(RemoteG2OffloadingSpec)
    adapter = _Adapter()
    spec._target_adapter = adapter  # type: ignore[assignment]
    spec._clients_lock = threading.Lock()
    spec._handshaked_peers = set()
    spec._target_clients_by_rank = cast(Any, {(7, 0): _Client()})
    spec.tp_rank = 0
    spec.use_host_bounce = True
    spec._target_page_size_bytes = 4096

    assert spec._ensure_peer("host_pinned:7") is False
    assert adapter.add_calls == 0
