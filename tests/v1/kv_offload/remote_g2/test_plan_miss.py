# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan-miss handling unit tests for SourceG2DescriptorRegistry +
RemoteG2OffloadingManager.

These exercise the failure modes that the two-engine evaluation
cannot easily target with a real engine:

* A plan whose hashes don't exist at all in source's registry — the
  resolve must return ``reason=missing`` with no descriptors, the
  manager's plan-driven path must fall through to the local-compute
  path on the scheduler.
* A plan whose hashes match a contiguous prefix only — resolve
  returns descriptors for the prefix, anything past the first miss
  is dropped, the lease pins only the prefix.
* A plan that crosses worker/dp boundaries — resolve rejects with
  the right ``reason`` string.
* A plan whose version is unsupported — also rejected.

The goal is to confirm that *every* path a malformed or partially
satisfiable plan can take leaves the registry in a clean state (no
leaked pins, no leases), so the manager can keep handling subsequent
requests safely.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from typing import Any

import pytest

from vllm.v1.kv_offload.remote_g2.data_model import (
    REMOTE_KV_REUSE_PLAN_VERSION,
    RemoteKvReusePlan,
    SourceG2DescriptorRecord,
    SourceG2DescriptorRegistry,
)


class _StubBlock:
    def __init__(self, block_id: int) -> None:
        self.block_id = int(block_id)
        self.ref_cnt = 0

    @property
    def is_ready(self) -> bool:
        return self.ref_cnt >= 0


class _StubPolicy:
    def __init__(self) -> None:
        self._blocks: dict[bytes, _StubBlock] = {}

    def get(self, key: bytes) -> _StubBlock | None:
        return self._blocks.get(key)

    def insert(self, key: bytes, block: _StubBlock) -> None:
        self._blocks[key] = block


def _fresh_registry(
    *,
    populated: Iterable[int] = (10, 11, 12, 13),
) -> tuple[SourceG2DescriptorRegistry, _StubPolicy]:
    SourceG2DescriptorRegistry._clear_singletons_for_tests()
    reg = SourceG2DescriptorRegistry.get_or_create(
        source_worker_id=1, source_dp_rank=0, lease_ttl_ms=10_000
    )
    policy = _StubPolicy()
    reg.set_policy(policy)
    reg.set_pool_layout(
        layer_pool_base_ptrs=[0x2000],
        layer_pool_size_bytes=[1024 * 4096],
        page_size_bytes=4096,
    )
    for h in populated:
        key = f"k{h}".encode()
        policy.insert(key, _StubBlock(h))
        reg.record_key(h, key)
        reg.upsert_descriptor(
            SourceG2DescriptorRecord(
                block_hash=h,
                source_worker_id=1,
                source_dp_rank=0,
                tier="host_pinned",
                descriptor_generation=1,
                pool_id="pool",
                byte_offset=h * 4096,
                byte_length=4096,
                block_id=h,
            )
        )
    return reg, policy


def _plan(
    hashes: list[int],
    *,
    plan_id: str = "p",
    source_worker_id: int = 1,
    source_dp_rank: int = 0,
    plan_version: int = REMOTE_KV_REUSE_PLAN_VERSION,
    expires_at_ms: int = 10**15,
) -> RemoteKvReusePlan:
    return RemoteKvReusePlan(
        plan_id=plan_id,
        request_id=f"r-{plan_id}",
        target_worker_id=2,
        target_dp_rank=0,
        source_worker_id=source_worker_id,
        source_dp_rank=source_dp_rank,
        source_tier="host_pinned",
        block_hashes=tuple(hashes),
        start_block_index=0,
        planned_prefix_blocks=len(hashes),
        block_size_tokens=16,
        created_at_ms=0,
        expires_at_ms=expires_at_ms,
        plan_version=plan_version,
    )


def _assert_clean(reg: SourceG2DescriptorRegistry, policy: _StubPolicy) -> None:
    """No leases outstanding, no block leaked a pin."""
    assert len(reg._leases) == 0, f"leaked leases: {list(reg._leases.keys())}"
    leaked = [
        h for h in reg._records
        if policy.get(b"k" + str(h).encode())
        and policy.get(b"k" + str(h).encode()).ref_cnt != 0
    ]
    assert not leaked, f"leaked pins on blocks {leaked}"


def test_full_miss_returns_no_lease_no_pins() -> None:
    """Plan referring to hashes that simply aren't in the registry."""
    reg, policy = _fresh_registry()
    result = reg.resolve_and_lease(_plan([900, 901, 902]))
    assert result.lease_id is None
    assert len(result.descriptors) == 0
    # The first per_block_status entry tells the caller WHY.
    assert result.per_block_status[0].status == "missing"
    _assert_clean(reg, policy)


def test_partial_miss_returns_prefix_descriptors() -> None:
    """Plan matches blocks 10, 11 but not 99."""
    reg, policy = _fresh_registry()
    result = reg.resolve_and_lease(_plan([10, 11, 99, 12]))
    assert result.lease_id is not None
    assert len(result.descriptors) == 2
    assert tuple(d.block_hash for d in result.descriptors) == (10, 11)
    # Block 99's status is reported.
    assert any(s.status == "missing" for s in result.per_block_status)
    # Only the prefix is pinned.
    assert policy.get(b"k10").ref_cnt == 1
    assert policy.get(b"k11").ref_cnt == 1
    assert policy.get(b"k12").ref_cnt == 0  # never reached

    reg.release_lease(result.lease_id)
    _assert_clean(reg, policy)


def test_wrong_source_worker_id_rejected() -> None:
    reg, policy = _fresh_registry()
    bad = _plan([10], source_worker_id=42)
    result = reg.resolve_and_lease(bad)
    assert result.lease_id is None
    assert result.reason == "wrong_source_worker"
    _assert_clean(reg, policy)


def test_wrong_source_dp_rank_rejected() -> None:
    reg, policy = _fresh_registry()
    bad = _plan([10], source_dp_rank=7)
    result = reg.resolve_and_lease(bad)
    assert result.lease_id is None
    assert result.reason == "wrong_source_rank"
    _assert_clean(reg, policy)


def test_unsupported_plan_version_rejected() -> None:
    reg, policy = _fresh_registry()
    bad = _plan([10], plan_version=999_999)
    result = reg.resolve_and_lease(bad)
    assert result.lease_id is None
    assert result.reason == "unsupported_plan_version"
    _assert_clean(reg, policy)


def test_expired_plan_rejected() -> None:
    reg, policy = _fresh_registry()
    bad = _plan([10], expires_at_ms=-1)
    result = reg.resolve_and_lease(bad)
    assert result.lease_id is None
    assert result.reason == "plan_expired"
    _assert_clean(reg, policy)


def test_malformed_plan_dict_rejected() -> None:
    reg, policy = _fresh_registry()
    # Missing required fields.
    result = reg.resolve_and_lease({"plan_id": "bad"})
    assert result.lease_id is None
    assert result.reason == "invalid_plan"
    _assert_clean(reg, policy)


def test_empty_plan_rejected() -> None:
    """Plan with planned_prefix_blocks=0 — resolves cleanly to an empty
    result without pinning anything."""
    reg, policy = _fresh_registry()
    plan = RemoteKvReusePlan(
        plan_id="empty",
        request_id="empty",
        target_worker_id=2,
        target_dp_rank=0,
        source_worker_id=1,
        source_dp_rank=0,
        source_tier="host_pinned",
        block_hashes=(),
        start_block_index=0,
        planned_prefix_blocks=0,
        block_size_tokens=16,
        created_at_ms=0,
        expires_at_ms=10**15,
        plan_version=REMOTE_KV_REUSE_PLAN_VERSION,
    )
    result = reg.resolve_and_lease(plan)
    assert result.lease_id is None
    assert len(result.descriptors) == 0
    _assert_clean(reg, policy)


def test_full_match_then_release_returns_to_clean_state() -> None:
    """Sanity: the happy path also returns to clean state after release."""
    reg, policy = _fresh_registry()
    res = reg.resolve_and_lease(_plan([10, 11, 12]))
    assert res.lease_id is not None
    assert reg.release_lease(res.lease_id) is True
    _assert_clean(reg, policy)


def test_many_consecutive_resolves_dont_leak() -> None:
    """A long sequence of mixed full/partial/miss resolves leaves the
    registry in a clean state (no slow leak)."""
    reg, policy = _fresh_registry(populated=range(20))

    cases = [
        [0, 1, 2],                     # full
        [5, 6, 7, 100],                # partial
        [200, 201],                    # full miss
        [10, 11, 12, 13, 14, 15],      # full
        [99, 100, 101],                # full miss
    ]
    for _ in range(50):
        for plan_hashes in cases:
            res = reg.resolve_and_lease(_plan(plan_hashes))
            if res.lease_id is not None:
                reg.release_lease(res.lease_id)
    _assert_clean(reg, policy)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header"])
