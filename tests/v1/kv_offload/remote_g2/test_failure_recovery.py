# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Failure-recovery tests for the RemoteG2 transfer handler.

The adapter exposes a ``fault_inject_every`` knob: every Nth logical
block raises ``RuntimeError`` instead of moving data, independent of
how blocks are grouped into NIXL transactions.
The transfer handler must raise synchronously from ``submit_load``.  The
offloading connector invokes that method before model forward; deferring the
failure to ``get_finished`` would let stale or partially-written KV participate
in one forward pass before the connector notices the failed result.

These tests pin down the contract end-to-end without standing up a
real vLLM engine:

* A failed READ raises before ``submit_load`` returns.
* Failed jobs never appear in the successful completion queue.
* Subsequent transfers after a failure recover (no stuck state).
* When *every* read fails (fault_inject_every=1), no partial-success
  masquerades as success.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence

import pytest

from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.remote_g2.load_spec import (
    RemoteG2LoadSpec,
    _RemoteBlockHandle,
)
from vllm.v1.kv_offload.remote_g2.nixl_adapter import RawNixlRemoteG2Adapter
from vllm.v1.kv_offload.remote_g2.transfer_handler import (
    RemoteG2TransferHandler,
)

PAGE_SIZE = 1024
NUM_BLOCKS = 4
NUM_LAYERS = 2


def _ptr_of(buf: bytearray) -> int:
    return ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf))


def _make_handler(
    *,
    fault_inject_every: int = 0,
) -> tuple[
    RemoteG2TransferHandler, RawNixlRemoteG2Adapter, list[bytearray], list[bytearray]
]:
    src_pools = [bytearray(NUM_BLOCKS * PAGE_SIZE) for _ in range(NUM_LAYERS)]
    tgt_pools = [bytearray(NUM_BLOCKS * PAGE_SIZE) for _ in range(NUM_LAYERS)]
    for layer, pool in enumerate(src_pools):
        for b in range(NUM_BLOCKS):
            fill = bytes([(layer * 31 + b * 7 + i) & 0xFF for i in range(PAGE_SIZE)])
            pool[b * PAGE_SIZE : (b + 1) * PAGE_SIZE] = fill

    adapter = RawNixlRemoteG2Adapter(
        "tgt",
        [_ptr_of(p) for p in tgt_pools],
        [len(p) for p in tgt_pools],
        use_mock=True,
        fault_inject_every=fault_inject_every,
    )
    adapter.add_peer(
        "src",
        peer_agent_metadata=b"mock:src",
        peer_layer_pool_base_ptrs=[_ptr_of(p) for p in src_pools],
        peer_layer_pool_size_bytes=[len(p) for p in src_pools],
    )

    peer_added = {"called": 0}

    def ensure_peer(peer_name: str) -> bool:
        peer_added["called"] += 1
        return True

    handler = RemoteG2TransferHandler(
        adapter=adapter,
        gpu_page_size_bytes=PAGE_SIZE,
        ensure_peer=ensure_peer,
    )
    return handler, adapter, src_pools, tgt_pools


def _spec_for(
    block_ids: list[int], lease_id: str = "L"
) -> tuple[RemoteG2LoadSpec, GPULoadStoreSpec]:
    blocks = [
        _RemoteBlockHandle(
            block_hash=1000 + bid,
            descriptor_generation=1,
            byte_offset=bid * PAGE_SIZE,
            byte_length=PAGE_SIZE,
        )
        for bid in block_ids
    ]
    src = RemoteG2LoadSpec(
        peer_name="src",
        lease_id=lease_id,
        blocks=blocks,
        source_worker_id=1,
        source_dp_rank=0,
    )
    dst = GPULoadStoreSpec(
        block_ids=block_ids,
        group_sizes=[len(block_ids)],
        block_indices=[0],
    )
    return src, dst


def test_happy_path_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    handler, adapter, src_pools, tgt_pools = _make_handler()
    calls: list[tuple[str, list[int], list[int], list[int]]] = []
    original_read_blocks = adapter.read_blocks

    def read_blocks_spy(
        peer_name: str,
        peer_byte_offsets: Sequence[int],
        local_byte_offsets: Sequence[int],
        byte_lengths: Sequence[int],
    ) -> None:
        calls.append(
            (
                peer_name,
                list(peer_byte_offsets),
                list(local_byte_offsets),
                list(byte_lengths),
            )
        )
        original_read_blocks(
            peer_name, peer_byte_offsets, local_byte_offsets, byte_lengths
        )

    monkeypatch.setattr(adapter, "read_blocks", read_blocks_spy)
    src, dst = _spec_for([0, 1, 2, 3])
    assert handler.submit_load(101, src, dst) is True
    finished = handler.get_finished()
    assert len(finished) == 1
    r = finished[0]
    assert r.job_id == 101
    assert r.success is True
    assert r.transfer_size == NUM_BLOCKS * PAGE_SIZE
    assert calls == [
        (
            "src",
            [block * PAGE_SIZE for block in range(NUM_BLOCKS)],
            [block * PAGE_SIZE for block in range(NUM_BLOCKS)],
            [PAGE_SIZE] * NUM_BLOCKS,
        )
    ]
    # And the bytes actually landed (each layer).
    for layer in range(NUM_LAYERS):
        assert bytes(tgt_pools[layer]) == bytes(src_pools[layer])


def test_every_read_fails_before_forward() -> None:
    """fault_inject_every=1 -> every call raises -> the FIRST read
    in the job fails, the rest are skipped, success=False."""
    handler, _adapter, _src, _tgt = _make_handler(fault_inject_every=1)
    src, dst = _spec_for([0, 1])
    with pytest.raises(RuntimeError, match="NIXL READ failed"):
        handler.submit_load(202, src, dst)
    assert handler.get_finished() == []


def test_one_failure_then_recovery() -> None:
    """fault_inject_every=3 -> 3rd / 6th / ... read fails. With 2
    blocks per job, jobs 1 and 2 (which together do 4 reads) include
    one failure; jobs 3+ should recover."""
    handler, _adapter, src_pools, tgt_pools = _make_handler(fault_inject_every=3)

    src1, dst1 = _spec_for([0, 1])
    src2, dst2 = _spec_for([2, 3])

    assert handler.submit_load(301, src1, dst1) is True
    with pytest.raises(RuntimeError, match="NIXL READ failed"):
        handler.submit_load(302, src2, dst2)
    results = handler.get_finished()
    assert [result.job_id for result in results] == [301]
    assert results[0].success is True

    # Subsequent transfers complete cleanly (handler isn't stuck).
    src3, dst3 = _spec_for([0])
    handler.submit_load(303, src3, dst3)
    # Need to issue enough additional successful reads so the counter
    # walks past the next multiple of 3 — adapter's counter is shared
    # across all logical block reads.
    src4, dst4 = _spec_for([1])
    handler.submit_load(304, src4, dst4)
    later = sorted(handler.get_finished(), key=lambda r: r.job_id)
    assert [r.job_id for r in later] == [303, 304]
    assert all(r.success for r in later)


def test_get_finished_is_idempotent_drain() -> None:
    """get_finished should drain results and return [] afterwards."""
    handler, _adapter, _src, _tgt = _make_handler()
    src, dst = _spec_for([0])
    handler.submit_load(401, src, dst)
    first = handler.get_finished()
    second = handler.get_finished()
    assert len(first) == 1
    assert second == []


def test_missing_peer_fails_before_forward() -> None:
    handler, _adapter, _src, _tgt = _make_handler()
    # Replace ensure_peer with one that always says no.
    handler._ensure_peer = lambda peer_name: False
    src, dst = _spec_for([0, 1])
    with pytest.raises(RuntimeError, match="metadata not available"):
        handler.submit_load(501, src, dst)
    assert handler.get_finished() == []


def test_block_count_mismatch_rejected_synchronously() -> None:
    """Mismatched src/dst block counts is a programmer error; we
    surface it by raising before forward (rather than enqueueing a deferred
    failure result)."""
    handler, _adapter, _src, _tgt = _make_handler()
    src, _dst = _spec_for([0, 1])
    # Build a DST spec with a different cardinality.
    bad_dst = GPULoadStoreSpec(
        block_ids=[0, 1, 2],
        group_sizes=[3],
        block_indices=[0],
    )
    with pytest.raises(RuntimeError, match="block count mismatch"):
        handler.submit_load(601, src, bad_dst)
    assert handler.get_finished() == []


def test_wrong_spec_types_rejected() -> None:
    handler, _adapter, _src, _tgt = _make_handler()
    # Wrong src type.
    bad_src = object()
    dst = GPULoadStoreSpec(block_ids=[0], group_sizes=[1], block_indices=[0])
    assert handler.submit_load(701, bad_src, dst) is False  # type: ignore[arg-type]


def test_cuda_completion_failure_raises_before_forward_and_releases_source_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    class _DirectAdapter:
        use_mock = False

        def read_blocks(self, *args, **kwargs) -> None:
            return

        def close(self) -> None:
            return

    released: list[str] = []
    handler = RemoteG2TransferHandler(
        adapter=_DirectAdapter(),  # type: ignore[arg-type]
        gpu_page_size_bytes=PAGE_SIZE,
        ensure_peer=lambda _peer: True,
        on_load_done=released.append,
    )

    def fail_sync() -> None:
        raise RuntimeError("sync failed")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_sync)
    src, dst = _spec_for([0])

    with pytest.raises(RuntimeError, match="CUDA completion failed"):
        handler.submit_load(801, src, dst)

    assert released == ["L"]
    assert handler.get_finished() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-header"])
