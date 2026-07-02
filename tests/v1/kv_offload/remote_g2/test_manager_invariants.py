# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manager-level regressions found by the Remote-G2 AgentX canary.

These tests deliberately use the real CPU manager policies.  Registry-only
tests with a stub policy cannot detect drift between ``ref_cnt`` and LRU's
dedicated evictable index, and they do not exercise the scheduler manager's
single-source load contract.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.remote_g2.data_model import (
    REMOTE_KV_REUSE_PLAN_VERSION,
    RemoteG2Descriptor,
    RemoteG2ResolveResult,
    RemoteKvReusePlan,
    SourceG2DescriptorRegistry,
)
from vllm.v1.kv_offload.remote_g2.load_spec import RemoteG2LoadSpec
from vllm.v1.kv_offload.remote_g2.manager import (
    REMOTE_G2_PLAN_KEY,
    RemoteG2OffloadingManager,
    block_hash_to_router_int,
)


def _key(value: int) -> OffloadKey:
    return make_offload_key(value.to_bytes(16, "big"), 0)


def _hash(key: OffloadKey) -> int:
    return block_hash_to_router_int(key[:-4])


def _plan(req_id: str, hashes: list[int]) -> RemoteKvReusePlan:
    return RemoteKvReusePlan(
        plan_id=f"plan-{req_id}",
        request_id=req_id,
        target_worker_id=2,
        target_dp_rank=0,
        source_worker_id=99,
        source_dp_rank=0,
        source_tier="host_pinned",
        block_hashes=tuple(hashes),
        kv_block_hashes=tuple(hashes),
        start_block_index=0,
        planned_prefix_blocks=len(hashes),
        block_size_tokens=16,
        created_at_ms=0,
        expires_at_ms=10**15,
        plan_version=REMOTE_KV_REUSE_PLAN_VERSION,
    )


def _ctx(req_id: str, plan: RemoteKvReusePlan | None = None) -> ReqContext:
    params = {REMOTE_G2_PLAN_KEY: plan} if plan is not None else None
    return ReqContext(req_id=req_id, kv_transfer_params=params)


def _manager(
    *, num_blocks: int = 4, cache_policy: Literal["lru", "arc"] = "lru"
) -> RemoteG2OffloadingManager:
    SourceG2DescriptorRegistry._clear_singletons_for_tests()
    manager = RemoteG2OffloadingManager(
        num_blocks=num_blocks,
        source_worker_id=2,
        source_dp_rank=0,
        cache_policy=cache_policy,
    )
    manager.set_pool_layout(
        layer_pool_base_ptrs=[0x100000],
        layer_pool_size_bytes=[num_blocks * 4096],
        page_size_bytes=4096,
    )
    return manager


def _block(manager: RemoteG2OffloadingManager, key: OffloadKey) -> BlockStatus:
    block = manager._policy.get(key)
    assert block is not None
    return block


def _store(
    manager: RemoteG2OffloadingManager,
    keys: list[OffloadKey],
    req_id: str = "store",
) -> None:
    context = _ctx(req_id)
    output = manager.prepare_store(keys, context)
    assert output is not None
    assert output.keys_to_store == keys
    manager.complete_store(output.keys_to_store, context)


@dataclass
class _FakeTargetClient:
    descriptors: tuple[RemoteG2Descriptor, ...]
    resolve_calls: int = 0
    releases: int = 0

    def resolve_and_lease(self, plan: RemoteKvReusePlan) -> RemoteG2ResolveResult:
        self.resolve_calls += 1
        return RemoteG2ResolveResult(
            lease_id=f"lease-{plan.request_id}-{self.resolve_calls}",
            descriptors=self.descriptors,
            num_tokens=len(self.descriptors) * 16,
        )

    def release_lease(self, lease_id: str, reason: str = "ack") -> bool:
        self.releases += 1
        return True


def _descriptor(key: OffloadKey, block_id: int = 0) -> RemoteG2Descriptor:
    return RemoteG2Descriptor(
        block_hash=_hash(key),
        descriptor_generation=1,
        pool_id="remote",
        byte_offset=block_id * 4096,
        byte_length=4096,
    )


def test_remote_first_stops_before_local_only_suffix() -> None:
    """One load job must never mix a remote-only and local-only key."""
    manager = _manager()
    remote_key, local_key = _key(10), _key(20)
    _store(manager, [local_key])
    plan = _plan("remote-first", [_hash(remote_key)])
    context = _ctx("remote-first", plan)
    client = _FakeTargetClient((_descriptor(remote_key),))
    manager.set_target_client_factory(lambda _: client)

    assert manager.lookup(remote_key, context) is LookupResult.HIT
    # Although this key exists locally, the request already selected the
    # remote source.  The prefix must stop instead of creating a mixed job.
    assert manager.lookup(local_key, context) is LookupResult.MISS

    spec = manager.prepare_load([remote_key], context)
    assert isinstance(spec, RemoteG2LoadSpec)
    assert spec.lease_id is not None
    client.release_lease(spec.lease_id, reason="load_done")
    manager.complete_load([remote_key], context)
    assert client.releases == 1  # no scheduler hot-path release RPC
    assert _block(manager, local_key).ref_cnt == 0
    manager.on_request_finished(context)
    assert client.releases == 2  # idempotent request-finish cleanup retry


def test_local_first_never_switches_to_remote_suffix() -> None:
    manager = _manager()
    local_key, remote_key = _key(10), _key(20)
    _store(manager, [local_key])
    plan = _plan("local-first", [_hash(remote_key)])
    context = _ctx("local-first", plan)
    client = _FakeTargetClient((_descriptor(remote_key),))
    manager.set_target_client_factory(lambda _: client)

    assert manager.lookup(local_key, context) is LookupResult.HIT
    assert manager.lookup(remote_key, context) is LookupResult.MISS
    assert client.resolve_calls == 0

    spec = manager.prepare_load([local_key], context)
    assert isinstance(spec, CPULoadStoreSpec)
    assert _block(manager, local_key).ref_cnt == 1
    manager.complete_load([local_key], context)
    assert _block(manager, local_key).ref_cnt == 0
    manager.on_request_finished(context)


def test_cached_remote_lease_does_not_override_later_local_selection() -> None:
    """A resolved plan entry is not proof that the submitted job is remote."""
    manager = _manager()
    local_key, planned_key, first_miss = _key(10), _key(20), _key(30)
    _store(manager, [local_key])
    plan = _plan("resolved-then-local", [_hash(planned_key)])
    context = _ctx("resolved-then-local", plan)
    client = _FakeTargetClient((_descriptor(planned_key),))
    manager.set_target_client_factory(lambda _: client)

    # Sliding-window lookup can inspect a non-covered key first.  This creates
    # a leased resolve entry but does not select remote as the load source.
    assert manager.lookup(first_miss, context) is LookupResult.MISS
    assert client.resolve_calls == 1
    assert manager.lookup(local_key, context) is LookupResult.HIT
    assert client.releases == 1
    assert context.req_id not in manager._resolve_cache

    spec = manager.prepare_load([local_key], context)
    assert isinstance(spec, CPULoadStoreSpec)
    manager.complete_load([local_key], context)
    assert _block(manager, local_key).ref_cnt == 0
    manager.on_request_finished(context)
    assert client.releases == 1


def test_unselected_remote_resolve_is_released_at_schedule_end() -> None:
    manager = _manager()
    planned_key, first_miss = _key(20), _key(30)
    plan = _plan("unused-resolve", [_hash(planned_key)])
    context = _ctx("unused-resolve", plan)
    client = _FakeTargetClient((_descriptor(planned_key),))
    manager.set_target_client_factory(lambda _: client)

    assert manager.lookup(first_miss, context) is LookupResult.MISS
    assert context.req_id in manager._resolve_cache
    manager.on_schedule_end()
    assert context.req_id not in manager._resolve_cache
    assert client.releases == 1


def test_remote_prepare_fails_closed_if_batch_crosses_source_boundary() -> None:
    manager = _manager()
    remote_key, uncovered_key = _key(10), _key(20)
    plan = _plan("bad-batch", [_hash(remote_key)])
    context = _ctx("bad-batch", plan)
    client = _FakeTargetClient((_descriptor(remote_key),))
    manager.set_target_client_factory(lambda _: client)

    assert manager.lookup(remote_key, context) is LookupResult.HIT
    with pytest.raises(RuntimeError, match="not covered by the selected remote plan"):
        manager.prepare_load([remote_key, uncovered_key], context)


def test_second_remote_load_for_same_request_resolves_a_fresh_lease() -> None:
    """Preemption/resume must not reuse descriptors from a released lease."""
    manager = _manager()
    key = _key(10)
    plan = _plan("remote-reload", [_hash(key)])
    context = _ctx("remote-reload", plan)
    client = _FakeTargetClient((_descriptor(key),))
    manager.set_target_client_factory(lambda _: client)

    assert manager.lookup(key, context) is LookupResult.HIT
    first = manager.prepare_load([key], context)
    assert isinstance(first, RemoteG2LoadSpec)
    assert first.lease_id == "lease-remote-reload-1"
    client.release_lease(first.lease_id, reason="load_done")
    manager.complete_load([key], context)
    assert client.releases == 1
    assert context.req_id not in manager._resolve_cache

    # The source selection remains remote, but its consumed resolve entry does
    # not.  A new lookup must call resolve_and_lease again.
    assert manager.lookup(key, context) is LookupResult.HIT
    second = manager.prepare_load([key], context)
    assert isinstance(second, RemoteG2LoadSpec)
    assert second.lease_id == "lease-remote-reload-2"
    assert client.resolve_calls == 2
    client.release_lease(second.lease_id, reason="load_done")
    manager.complete_load([key], context)
    assert client.releases == 2
    manager.on_request_finished(context)
    assert client.releases == 4


@pytest.mark.parametrize("source", ["local", "remote"])
def test_finish_before_complete_preserves_prepared_load_state(source: str) -> None:
    """Cancellation may call on_request_finished while a load is pending."""
    manager = _manager()
    key = _key(10)
    if source == "local":
        _store(manager, [key])
        context = _ctx(f"cancel-{source}")
        assert manager.lookup(key, context) is LookupResult.HIT
        client = None
    else:
        plan = _plan(f"cancel-{source}", [_hash(key)])
        context = _ctx(f"cancel-{source}", plan)
        client = _FakeTargetClient((_descriptor(key),))
        manager.set_target_client_factory(lambda _: client)
        assert manager.lookup(key, context) is LookupResult.HIT

    manager.prepare_load([key], context)
    manager.on_request_finished(context)
    if source == "remote":
        assert client is not None
        # Releasing here would let the source evict while the READ is pending.
        assert client.releases == 0
        client.release_lease(f"lease-{context.req_id}-1", reason="load_done")

    manager.complete_load([key], context)
    assert context.req_id not in manager._prepared_load_source_by_req
    assert context.req_id not in manager._load_source_by_req
    assert context.req_id not in manager._resolve_cache
    assert context.req_id not in manager._finished_with_pending_load
    if source == "local":
        assert _block(manager, key).ref_cnt == 0
    else:
        assert client is not None and client.releases == 2


@pytest.mark.parametrize("cache_policy", ["lru", "arc"])
def test_lease_pin_uses_real_policy_bookkeeping(
    cache_policy: Literal["lru", "arc"],
) -> None:
    manager = _manager(num_blocks=3, cache_policy=cache_policy)
    pinned, other, third, replacement = (_key(i) for i in range(4))
    _store(manager, [pinned, other, third])
    registry = manager.registry
    plan = _plan("pin-evict", [_hash(pinned)])
    # This manager is the source for the direct registry call.
    plan = replace(plan, source_worker_id=2, target_worker_id=3)

    result = registry.resolve_and_lease(plan)
    assert result.lease_id is not None
    assert _block(manager, pinned).ref_cnt == 1
    assert manager._num_evictable_cache_blocks == 2
    if cache_policy == "lru":
        assert isinstance(manager._policy, LRUCachePolicy)
        assert pinned not in manager._policy.evictable_blocks

    output = manager.prepare_store([replacement], _ctx("replacement"))
    assert output is not None
    assert pinned not in output.evicted_keys
    assert len(output.evicted_keys) == 1

    assert registry.release_lease(result.lease_id)
    assert _block(manager, pinned).ref_cnt == 0
    if cache_policy == "lru":
        assert isinstance(manager._policy, LRUCachePolicy)
        assert pinned in manager._policy.evictable_blocks


@pytest.mark.parametrize("cache_policy", ["lru", "arc"])
def test_all_blocks_leased_reports_no_eviction_capacity(
    cache_policy: Literal["lru", "arc"],
) -> None:
    manager = _manager(num_blocks=1, cache_policy=cache_policy)
    pinned, replacement = _key(10), _key(20)
    _store(manager, [pinned])
    plan = replace(
        _plan("all-pinned", [_hash(pinned)]),
        source_worker_id=2,
        target_worker_id=3,
    )
    result = manager.registry.resolve_and_lease(plan)
    assert result.lease_id is not None
    assert manager._num_evictable_cache_blocks == 0

    # In particular, LRU must not enter evict() and assert on a pinned key.
    assert manager.prepare_store([replacement], _ctx("while-pinned")) is None

    manager.registry.release_lease(result.lease_id)
    output = manager.prepare_store([replacement], _ctx("after-release"))
    assert output is not None
    assert output.evicted_keys == [pinned]


def test_multiple_leases_and_local_load_share_one_ref_transition() -> None:
    manager = _manager(num_blocks=2)
    key = _key(10)
    _store(manager, [key])
    registry = manager.registry
    plan = _plan("multi-lease", [_hash(key)])
    plan = replace(plan, source_worker_id=2, target_worker_id=3)
    base_evictable = manager._num_evictable_cache_blocks

    first = registry.resolve_and_lease(plan)
    second = registry.resolve_and_lease(plan)
    assert first.lease_id is not None and second.lease_id is not None
    assert _block(manager, key).ref_cnt == 2
    assert manager._num_evictable_cache_blocks == base_evictable - 1

    context = _ctx("local-overlap")
    manager.prepare_load([key], context)
    assert _block(manager, key).ref_cnt == 3
    manager.complete_load([key], context)
    assert _block(manager, key).ref_cnt == 2

    registry.release_lease(first.lease_id)
    assert _block(manager, key).ref_cnt == 1
    assert manager._num_evictable_cache_blocks == base_evictable - 1
    registry.release_lease(second.lease_id)
    assert _block(manager, key).ref_cnt == 0
    assert manager._num_evictable_cache_blocks == base_evictable


def test_reset_fails_closed_with_active_inbound_lease() -> None:
    manager = _manager(num_blocks=2)
    key = _key(10)
    _store(manager, [key])
    plan = _plan("reset", [_hash(key)])
    plan = replace(plan, source_worker_id=2, target_worker_id=3)
    result = manager.registry.resolve_and_lease(plan)
    assert result.lease_id is not None
    local_context = _ctx("local-before-reset")
    assert manager.lookup(key, local_context) is LookupResult.HIT
    assert local_context.req_id in manager._load_source_by_req

    with pytest.raises(RuntimeError, match="active inbound leases"):
        manager.reset_cache()
    # Failure is atomic: the source block remains pinned and request state is
    # untouched until the peer releases its lease.
    assert _block(manager, key).ref_cnt == 1
    assert result.lease_id in manager.registry._leases
    assert local_context.req_id in manager._load_source_by_req
    assert manager.registry.get_descriptor(_hash(key)) is not None
    assert _hash(key) in manager.registry._hash_to_key

    assert manager.registry.release_lease(result.lease_id)
    manager.reset_cache()
    assert manager._policy.get(key) is None
    assert manager._num_evictable_cache_blocks == 0
    assert not manager.registry._leases
    assert not manager._load_source_by_req
    assert not manager._prepared_load_source_by_req
    assert not manager._finished_with_pending_load
    assert not manager._consumed_resolve_entries


def test_reset_fails_closed_with_pending_outbound_load() -> None:
    manager = _manager()
    key = _key(10)
    plan = _plan("reset-outbound", [_hash(key)])
    context = _ctx("reset-outbound", plan)
    client = _FakeTargetClient((_descriptor(key),))
    manager.set_target_client_factory(lambda _: client)

    assert manager.lookup(key, context) is LookupResult.HIT
    spec = manager.prepare_load([key], context)
    assert isinstance(spec, RemoteG2LoadSpec)
    with pytest.raises(RuntimeError, match="pending load jobs"):
        manager.reset_cache()
    assert context.req_id in manager._prepared_load_source_by_req
    assert context.req_id in manager._resolve_cache
    assert client.releases == 0

    assert spec.lease_id is not None
    client.release_lease(spec.lease_id, reason="load_done")
    manager.complete_load([key], context)
    manager.on_request_finished(context)
    manager.reset_cache()
    assert not manager._prepared_load_source_by_req
    assert not manager._resolve_cache
    assert not manager._consumed_resolve_entries


def test_ttl_expiry_restores_real_lru_bookkeeping() -> None:
    manager = _manager(num_blocks=2)
    key = _key(10)
    _store(manager, [key])
    now = [0]
    manager.registry._clock_ms = lambda: now[0]
    manager.registry.lease_ttl_ms = 100
    plan = replace(
        _plan("ttl", [_hash(key)]),
        source_worker_id=2,
        target_worker_id=3,
    )
    result = manager.registry.resolve_and_lease(plan)
    assert result.lease_id is not None
    assert _block(manager, key).ref_cnt == 1
    assert manager._num_evictable_cache_blocks == 0

    now[0] = 101
    assert manager.registry.expire_stale_leases() == 1
    assert _block(manager, key).ref_cnt == 0
    assert manager._num_evictable_cache_blocks == 1
    assert isinstance(manager._policy, LRUCachePolicy)
    assert key in manager._policy.evictable_blocks
