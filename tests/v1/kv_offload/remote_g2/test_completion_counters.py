# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the post-completion transfer counters + transport provenance.

``RemoteG2OffloadingManager.complete_load`` calls
``registry.record_completed_load(entry.bytes_emitted)`` and zeroes its
per-request tally on a True return. All the load-bearing counting logic
lives in ``record_completed_load`` so it is testable at the registry
level (torch-free, mirroring ``test_plan_miss`` / ``test_lease_pin``);
the manager is a thin caller. These counters + the transport-backend
provenance are what a fail-closed perf gate reads over the stats RPC to
prove a real NIXL transfer *completed* (not a control-plane recompute
fallback, and not a mock-memcpy fallback).

What is pinned here:
  * defaults are fail-closed (mock=True, backend "unset", counters 0);
  * ``record_completed_load`` counts a real completion, ignores the
    no-spec-emitted case (bytes_emitted<=0), and -- combined with the
    caller zeroing its tally -- does NOT double-count a re-fired
    ``complete_load`` (the second call passes 0);
  * ``plan_bytes_completed`` is a distinct signal from the plan-time
    ``plan_blocks_loaded``;
  * ``set_transport_info`` + the stats RPC surface backend/mock/num_layers.

The full manager ``prepare_load -> complete_load`` path (emit tracking +
the leased-but-fully-local guard) needs a real engine/connector context
and is covered by the two-engine evaluation and the perf gate's
per-puller equality checks, not this torch-free unit test.
"""

from __future__ import annotations

from vllm.v1.kv_offload.remote_g2.data_model import (
    SourceG2DescriptorRegistry,
)
from vllm.v1.kv_offload.remote_g2.source_rpc import SourceG2RpcServer


def _fresh_registry() -> SourceG2DescriptorRegistry:
    SourceG2DescriptorRegistry._clear_singletons_for_tests()
    return SourceG2DescriptorRegistry.get_or_create(
        source_worker_id=1, source_dp_rank=0, lease_ttl_ms=10_000
    )


def _stats(reg: SourceG2DescriptorRegistry) -> dict:
    # No start() -> no ZMQ socket bound; _handle_stats reads the registry.
    return SourceG2RpcServer(reg)._handle_stats({"sample_limit": 0})["result"]


def test_defaults_are_fail_closed() -> None:
    reg = _fresh_registry()
    assert reg.plan_loads_completed == 0
    assert reg.plan_bytes_completed == 0
    # A gate must reject until a real adapter reports in.
    assert reg.transport_mock is True
    assert reg.transport_backend == "unset"


def test_record_completed_load_counts_and_ignores_nonpositive() -> None:
    reg = _fresh_registry()
    # No spec emitted for this request -> not a transfer -> no count.
    assert reg.record_completed_load(0, 42) is False
    assert reg.record_completed_load(-5, 42) is False
    assert reg.plan_loads_completed == 0
    assert reg.plan_bytes_completed == 0
    # A real completed load from source 42.
    assert reg.record_completed_load(3 * 4096, 42) is True
    assert reg.plan_loads_completed == 1
    assert reg.plan_bytes_completed == 12288
    assert reg.completed_loads_by_source == {42: 1}
    assert reg.completed_bytes_by_source == {42: 12288}


def test_no_double_count_on_refired_complete_load() -> None:
    """complete_load zeroes its tally after a counted completion, so a
    re-fired complete_load passes 0 here and must not double-count."""
    reg = _fresh_registry()
    emitted = 5 * 4096
    assert reg.record_completed_load(emitted, 42) is True  # first completion
    assert reg.record_completed_load(0, 42) is False  # re-fire (tally zeroed)
    assert reg.plan_loads_completed == 1
    assert reg.plan_bytes_completed == emitted
    assert reg.completed_loads_by_source == {42: 1}
    # A genuinely new completion (next request) advances again.
    assert reg.record_completed_load(2 * 4096, 42) is True
    assert reg.plan_loads_completed == 2
    assert reg.plan_bytes_completed == emitted + 2 * 4096
    assert reg.completed_loads_by_source == {42: 2}


def test_completions_bucketed_by_source() -> None:
    """A gate must be able to prove every completion came from the expected
    source; the per-source breakdown makes that checkable."""
    reg = _fresh_registry()
    reg.record_completed_load(4096, 100)  # from A
    reg.record_completed_load(4096, 100)  # from A
    reg.record_completed_load(4096, 200)  # from a different source
    assert reg.completed_loads_by_source == {100: 2, 200: 1}
    assert reg.completed_bytes_by_source == {100: 8192, 200: 4096}
    res = _stats(reg)
    # Stats surface string-keyed maps (JSON-friendly for the perf gate).
    assert res["completed_loads_by_source"] == {"100": 2, "200": 1}
    assert res["completed_bytes_by_source"] == {"100": 8192, "200": 4096}


def test_completion_is_distinct_from_plan_time_blocks() -> None:
    reg = _fresh_registry()
    reg.plan_blocks_loaded += 3  # plan-time (proves nothing transferred)
    reg.record_completed_load(3 * 4096, 100)  # post-completion (proves it did)
    ts = _stats(reg)["target_stats"]
    assert ts["plan_blocks_loaded"] == 3
    assert ts["plan_loads_completed"] == 1
    assert ts["plan_bytes_completed"] == 12288
    assert ts["plan_bytes_completed"] != ts["plan_blocks_loaded"]


def test_boot_id_is_stable_and_surfaced() -> None:
    """boot_id is stable for a registry instance (a gate compares before/
    after to catch a mid-run restart) and distinct across instances."""
    reg = _fresh_registry()
    bid = reg.boot_id
    assert bid and _stats(reg)["boot_id"] == bid
    assert reg.boot_id == bid  # stable across reads
    other = _fresh_registry()  # simulates a restart (fresh singleton)
    assert other.boot_id != bid


def test_stats_surface_all_gate_fields() -> None:
    """The perf gate reads these exact keys over the source-RPC socket."""
    reg = _fresh_registry()
    reg.set_transport_info(backend="UCX", mock=False)
    res = _stats(reg)
    ts = res["target_stats"]
    for key in (
        "plan_seen_count",
        "plan_resolved_count",
        "plan_load_specs_emitted",
        "plan_blocks_loaded",
        "plan_loads_completed",
        "plan_bytes_completed",
    ):
        assert key in ts and isinstance(ts[key], int)
    assert res["transport_backend"] == "UCX"
    assert res["transport_mock"] is False
    assert isinstance(res["num_layers"], int)
    assert isinstance(res["boot_id"], str) and res["boot_id"]
    assert res["completed_loads_by_source"] == {}
    assert res["completed_bytes_by_source"] == {}


def test_set_transport_info_mock_is_visible() -> None:
    reg = _fresh_registry()
    reg.set_transport_info(backend="MOCK", mock=True)
    res = _stats(reg)
    assert res["transport_mock"] is True
    assert res["transport_backend"] == "MOCK"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--no-header"])
