# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the lease pin protection on SourceG2DescriptorRegistry.

The registry is supposed to bump the policy block's ``ref_cnt`` on
``resolve_and_lease`` and decrement on ``release_lease`` (or on
TTL expiry). This protects in-flight cross-engine transfers from a
concurrent eviction.

Scenarios covered here exercise the registry directly without going
through a real vLLM engine — they pin a stub policy that mimics the
CPUOffloadingManager surface the registry depends on, so we can
assert ref_cnt invariants under every combination of lifecycle event.
"""

from __future__ import annotations

import threading
import time

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
        # Match BlockStatus semantics: ref_cnt == -1 is "not ready"; we
        # mark blocks ready by default in these tests since the registry
        # is only handed blocks that complete_store has finalised.
        self.ref_cnt = 0

    @property
    def is_ready(self) -> bool:
        return self.ref_cnt >= 0


class _StubPolicy:
    """Minimal CachePolicy stand-in.

    Stores ``key -> _StubBlock`` and exposes the subset of the policy
    surface the registry actually touches (``get``). Eviction is done
    by calling :meth:`drop`, which mimics what
    ``RemoteG2OffloadingManager.prepare_store`` would do after the LRU
    policy returned evicted blocks (i.e. only blocks with ``ref_cnt
    == 0``).
    """

    def __init__(self) -> None:
        self._blocks: dict[bytes, _StubBlock] = {}

    def get(self, key: bytes) -> _StubBlock | None:
        return self._blocks.get(key)

    def insert(self, key: bytes, block: _StubBlock) -> None:
        self._blocks[key] = block

    def drop(self, key: bytes) -> _StubBlock | None:
        """Emulate eviction. Like LRUCachePolicy.evict, refuses to
        drop a block whose ref_cnt > 0."""
        block = self._blocks.get(key)
        if block is None or block.ref_cnt > 0:
            return None
        return self._blocks.pop(key)


def _make_registry(
    *,
    lease_ttl_ms: int = 1000,
    clock: list[int] | None = None,
) -> tuple[SourceG2DescriptorRegistry, _StubPolicy]:
    SourceG2DescriptorRegistry._clear_singletons_for_tests()
    clk = clock if clock is not None else [0]
    reg = SourceG2DescriptorRegistry.get_or_create(
        source_worker_id=1,
        source_dp_rank=0,
        lease_ttl_ms=lease_ttl_ms,
        clock_ms=lambda: clk[0],
    )
    policy = _StubPolicy()
    reg.set_policy(policy)
    reg.set_pool_layout(
        layer_pool_base_ptrs=[0x1000],
        layer_pool_size_bytes=[64 * 4096],
        page_size_bytes=4096,
    )
    return reg, policy


def _register_block(
    reg: SourceG2DescriptorRegistry,
    policy: _StubPolicy,
    block_hash: int,
    block_id: int,
) -> None:
    key = f"key{block_hash}".encode("utf-8")
    policy.insert(key, _StubBlock(block_id))
    reg.record_key(block_hash, key)
    reg.upsert_descriptor(
        SourceG2DescriptorRecord(
            block_hash=block_hash,
            source_worker_id=reg.source_worker_id,
            source_dp_rank=reg.source_dp_rank,
            tier="host_pinned",
            descriptor_generation=1,
            pool_id="pool",
            byte_offset=block_id * 4096,
            byte_length=4096,
            block_id=block_id,
        )
    )


def _plan(hashes: list[int], plan_id: str = "p") -> RemoteKvReusePlan:
    return RemoteKvReusePlan(
        plan_id=plan_id,
        request_id=f"r-{plan_id}",
        target_worker_id=2,
        target_dp_rank=0,
        source_worker_id=1,
        source_dp_rank=0,
        source_tier="host_pinned",
        block_hashes=tuple(hashes),
        start_block_index=0,
        planned_prefix_blocks=len(hashes),
        block_size_tokens=16,
        created_at_ms=0,
        expires_at_ms=10**15,
        plan_version=REMOTE_KV_REUSE_PLAN_VERSION,
    )


# --- lifecycle: pin / unpin balance ---


def test_resolve_pins_and_release_unpins() -> None:
    reg, policy = _make_registry()
    for h in (10, 11, 12):
        _register_block(reg, policy, h, h)

    # Before resolve — every block ref_cnt == 0.
    for h in (10, 11, 12):
        assert policy.get(b"key" + str(h).encode()).ref_cnt == 0

    result = reg.resolve_and_lease(_plan([10, 11, 12]))
    assert result.reason == "ok"
    assert result.lease_id is not None
    assert len(result.descriptors) == 3

    # After resolve — every leased block's ref_cnt == 1.
    for h in (10, 11, 12):
        assert policy.get(b"key" + str(h).encode()).ref_cnt == 1
    assert reg.pin_count_total == 3
    assert reg.unpin_count_total == 0

    assert reg.release_lease(result.lease_id) is True
    for h in (10, 11, 12):
        assert policy.get(b"key" + str(h).encode()).ref_cnt == 0
    assert reg.unpin_count_total == 3
    # Re-release is a no-op.
    assert reg.release_lease(result.lease_id) is False


def test_partial_resolve_releases_partial_pins() -> None:
    """If a plan has hashes A, B, C where C is missing, the prefix
    resolve stops at C and the pin on A+B must be released so they
    don't stay leaked."""
    reg, policy = _make_registry()
    _register_block(reg, policy, 10, 0)
    _register_block(reg, policy, 11, 1)
    # 12 is intentionally NOT registered.

    result = reg.resolve_and_lease(_plan([10, 11, 12]))
    # Prefix is [10, 11], stopping at 12.
    assert result.lease_id is not None
    assert len(result.descriptors) == 2
    # Pins are held on the lease — only released on release_lease.
    assert policy.get(b"key10").ref_cnt == 1
    assert policy.get(b"key11").ref_cnt == 1

    reg.release_lease(result.lease_id)
    assert policy.get(b"key10").ref_cnt == 0
    assert policy.get(b"key11").ref_cnt == 0


def test_first_block_missing_releases_no_pin() -> None:
    """Plan whose first hash is missing returns no lease and leaves
    nothing pinned (no partial state to leak)."""
    reg, policy = _make_registry()
    _register_block(reg, policy, 11, 1)

    result = reg.resolve_and_lease(_plan([99, 11]))
    assert result.lease_id is None
    assert len(result.descriptors) == 0
    assert policy.get(b"key11").ref_cnt == 0
    # No pins taken, no unpins logged.
    assert reg.pin_count_total == 0


def test_pin_failure_after_eviction_in_middle() -> None:
    """If a block disappears from the policy *between* the registry
    record lookup and the pin attempt, the resolve stops at that block
    and the prior pins are still released cleanly on release_lease."""
    reg, policy = _make_registry()
    _register_block(reg, policy, 10, 0)
    _register_block(reg, policy, 11, 1)
    _register_block(reg, policy, 12, 2)

    # Manually evict block 11 from the policy WITHOUT going through
    # remove_descriptor, so the registry's record still claims 11 is
    # there. (Simulates the narrow race between resolve start and
    # eviction commit.)
    policy._blocks.pop(b"key11")

    result = reg.resolve_and_lease(_plan([10, 11, 12]))
    # Block 10 resolves fine, 11 fails to pin → break.
    assert result.lease_id is not None
    assert len(result.descriptors) == 1
    assert result.descriptors[0].block_hash == 10
    assert policy.get(b"key10").ref_cnt == 1
    # 11 was missing from policy → no ref_cnt change.
    # 12 never got resolved.
    assert policy.get(b"key12").ref_cnt == 0

    reg.release_lease(result.lease_id)
    assert policy.get(b"key10").ref_cnt == 0


# --- eviction protection ---


def test_pinned_block_cannot_be_evicted() -> None:
    """The LRU policy's evict respects ref_cnt > 0. We simulate that
    on the stub policy and confirm the leased blocks are immune."""
    reg, policy = _make_registry()
    for h in (10, 11, 12):
        _register_block(reg, policy, h, h)

    result = reg.resolve_and_lease(_plan([10, 11]))
    assert result.lease_id is not None
    # Try to evict — only block 12 (unpinned) should be drop-able.
    assert policy.drop(b"key10") is None  # pinned
    assert policy.drop(b"key11") is None  # pinned
    assert policy.drop(b"key12") is not None  # ok

    reg.release_lease(result.lease_id)
    # After release, 10 and 11 become evictable.
    assert policy.drop(b"key10") is not None
    assert policy.drop(b"key11") is not None


# --- TTL ---


def test_ttl_expiry_releases_pins() -> None:
    """A lease that lives past ``lease_ttl_ms`` is reaped by
    ``expire_stale_leases``, and its pins must be released so the
    blocks become evictable again."""
    clk = [0]
    reg, policy = _make_registry(lease_ttl_ms=100, clock=clk)
    for h in (10, 11):
        _register_block(reg, policy, h, h)

    result = reg.resolve_and_lease(_plan([10, 11]))
    assert result.lease_id is not None
    assert policy.get(b"key10").ref_cnt == 1

    # Time passes within TTL — no expiry.
    clk[0] = 50
    assert reg.expire_stale_leases() == 0
    assert policy.get(b"key10").ref_cnt == 1

    # Time passes past TTL.
    clk[0] = 200
    assert reg.expire_stale_leases() == 1
    assert policy.get(b"key10").ref_cnt == 0
    assert policy.get(b"key11").ref_cnt == 0
    # Lease entry is gone — re-releasing returns False.
    assert reg.release_lease(result.lease_id) is False


# --- concurrency ---


def test_concurrent_resolve_and_release_pin_balance() -> None:
    """N threads concurrently resolve+release the same plan thousands
    of times. After every thread joins, every block's ref_cnt must be
    exactly 0 (no double-release, no leaked pin)."""
    reg, policy = _make_registry()
    for h in range(8):
        _register_block(reg, policy, h, h)
    plan = _plan(list(range(8)))

    n_threads = 8
    iters = 500
    errors: list[str] = []

    def worker() -> None:
        for _ in range(iters):
            res = reg.resolve_and_lease(plan)
            if res.lease_id is None or len(res.descriptors) != 8:
                errors.append(f"unexpected resolve: {res!r}")
                return
            if not reg.release_lease(res.lease_id):
                errors.append(f"release returned False for {res.lease_id}")
                return

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:5]
    # Final state: every block back to ref_cnt 0, every lease cleared.
    for h in range(8):
        assert policy.get(b"key" + str(h).encode()).ref_cnt == 0
    assert len(reg._leases) == 0
    assert reg.pin_count_total == reg.unpin_count_total
    assert reg.pin_count_total == n_threads * iters * 8


def test_concurrent_resolve_under_eviction_attempts() -> None:
    """One thread resolves+releases repeatedly while another tries to
    evict every block. The eviction thread should never drop a block
    whose ref_cnt > 0; the resolve thread should always see a
    consistent prefix that is either fully resolvable or empty."""
    reg, policy = _make_registry()
    block_hashes = list(range(8))
    for h in block_hashes:
        _register_block(reg, policy, h, h)
    plan = _plan(block_hashes)

    stop = threading.Event()
    iters_resolve = [0]
    iters_evict = [0]
    errors: list[str] = []

    def resolve_loop() -> None:
        while not stop.is_set():
            res = reg.resolve_and_lease(plan)
            if res.lease_id is None:
                # Allowed — eviction wiped a block.
                continue
            for d in res.descriptors:
                block = policy.get(b"key" + str(d.block_hash).encode())
                if block is None or block.ref_cnt < 1:
                    errors.append(
                        f"resolved block {d.block_hash} but policy says "
                        f"ref_cnt={block.ref_cnt if block else 'gone'}"
                    )
                    stop.set()
                    return
            reg.release_lease(res.lease_id)
            iters_resolve[0] += 1

    def evict_loop() -> None:
        idx = 0
        while not stop.is_set():
            h = block_hashes[idx % len(block_hashes)]
            key = b"key" + str(h).encode()
            with reg._lock:
                evicted = policy.drop(key)
                if evicted is not None:
                    reg.remove_descriptor(h)
                    reg.forget_key(h)
                    # Reinsert immediately so the resolve thread keeps
                    # finding blocks.
                    policy.insert(key, _StubBlock(h))
                    reg.record_key(h, key)
                    reg.upsert_descriptor(
                        SourceG2DescriptorRecord(
                            block_hash=h,
                            source_worker_id=reg.source_worker_id,
                            source_dp_rank=reg.source_dp_rank,
                            tier="host_pinned",
                            descriptor_generation=2,
                            pool_id="pool",
                            byte_offset=h * 4096,
                            byte_length=4096,
                            block_id=h,
                        )
                    )
                    iters_evict[0] += 1
            idx += 1

    rt = threading.Thread(target=resolve_loop)
    et = threading.Thread(target=evict_loop)
    rt.start()
    et.start()
    time.sleep(1.0)
    stop.set()
    rt.join(timeout=5.0)
    et.join(timeout=5.0)

    assert not errors, errors[:5]
    assert iters_resolve[0] > 100, f"resolve loop barely ran: {iters_resolve[0]}"
    assert iters_evict[0] > 5, f"evict loop barely ran: {iters_evict[0]}"
    # Final ref_cnt invariant.
    for h in block_hashes:
        block = policy.get(b"key" + str(h).encode())
        if block is not None:
            assert block.ref_cnt == 0, (
                f"block {h} leaked pin: ref_cnt={block.ref_cnt}"
            )


# --- stress: many concurrent leases ---


def test_thousand_concurrent_leases() -> None:
    """Stress the lease table itself: spin up 1000 leases on disjoint
    block sets and confirm every lease can be released cleanly."""
    reg, policy = _make_registry()
    n_blocks = 1000
    for h in range(n_blocks):
        _register_block(reg, policy, h, h)

    leases = []
    for i in range(n_blocks):
        res = reg.resolve_and_lease(_plan([i], plan_id=f"p{i}"))
        assert res.lease_id is not None
        leases.append(res.lease_id)

    # Every block pinned exactly once.
    for h in range(n_blocks):
        assert policy.get(b"key" + str(h).encode()).ref_cnt == 1

    for lid in leases:
        assert reg.release_lease(lid) is True
    for h in range(n_blocks):
        assert policy.get(b"key" + str(h).encode()).ref_cnt == 0
    assert reg.pin_count_total == reg.unpin_count_total == n_blocks


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header"])
