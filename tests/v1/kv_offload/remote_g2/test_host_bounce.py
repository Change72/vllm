# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only contract tests for the Remote-G2 host-bounce pipeline."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest

import vllm.v1.kv_offload.remote_g2.host_bounce as host_bounce
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.remote_g2.host_bounce import (
    CudaHostBounceCopyEngine,
    HostBounceTransferError,
    HostBounceTransferStats,
    RemoteG2HostBounceTransport,
)
from vllm.v1.kv_offload.remote_g2.load_spec import (
    RemoteG2LoadSpec,
    _RemoteBlockHandle,
)
from vllm.v1.kv_offload.remote_g2.transfer_handler import (
    RemoteG2TransferHandler,
)

pytestmark = pytest.mark.cpu_test


PAGE_SIZE = 4096
BLOCKS_PER_SLOT = 64


class _FakeAdapter:
    use_mock = False

    def __init__(
        self,
        events: list[str],
        *,
        fail_read_at: int | None = None,
        fail_close_times: int = 0,
    ) -> None:
        self.events = events
        self.fail_read_at = fail_read_at
        self.read_calls: list[dict[str, Any]] = []
        self.close_calls = 0
        self.fail_close_times = fail_close_times
        self.closed = False

    def read_blocks(
        self,
        peer_name: str,
        peer_byte_offsets: Sequence[int],
        local_byte_offsets: Sequence[int],
        byte_lengths: Sequence[int],
    ) -> None:
        call = len(self.read_calls)
        self.events.append(f"read:{call}")
        self.read_calls.append(
            {
                "peer_name": peer_name,
                "peer_byte_offsets": list(peer_byte_offsets),
                "local_byte_offsets": list(local_byte_offsets),
                "byte_lengths": list(byte_lengths),
            }
        )
        if self.fail_read_at == call:
            raise RuntimeError(f"injected NIXL failure at read {call}")

    def close(self) -> None:
        if self.closed:
            return
        self.events.append("adapter.close")
        self.close_calls += 1
        if self.fail_close_times > 0:
            self.fail_close_times -= 1
            raise RuntimeError("injected deregistration failure")
        self.closed = True


class _FakeCopyEngine:
    layer_pool_base_ptrs = [1000]
    layer_pool_size_bytes = [2 * BLOCKS_PER_SLOT * PAGE_SIZE]

    def __init__(
        self,
        events: list[str],
        *,
        fail_enqueue_at: int | None = None,
        fail_normal_drain: bool = False,
        fail_normal_drain_times: int = 0,
        fail_error_drain: bool = False,
        fail_release_times: int = 0,
    ) -> None:
        self.events = events
        self.fail_enqueue_at = fail_enqueue_at
        self.fail_normal_drain = fail_normal_drain
        self.fail_normal_drain_times = fail_normal_drain_times
        self.fail_error_drain = fail_error_drain
        self.active_slots: set[int] = set()
        self.wait_calls: list[int] = []
        self.enqueue_calls: list[dict[str, Any]] = []
        self.drain_calls: list[bool] = []
        self.release_calls = 0
        self.release_attempts = 0
        self.fail_release_times = fail_release_times

    def wait_slot(self, slot: int) -> float:
        self.events.append(f"wait:{slot}")
        self.wait_calls.append(slot)
        if slot not in self.active_slots:
            return 0.0
        self.active_slots.remove(slot)
        return 0.01

    def enqueue(
        self,
        slot: int,
        gpu_block_ids: Sequence[int],
        byte_lengths: Sequence[int],
    ) -> float:
        call = len(self.enqueue_calls)
        self.events.append(f"enqueue:{slot}:{call}")
        if slot in self.active_slots:
            raise AssertionError(f"slot {slot} reused without wait_slot")
        self.enqueue_calls.append(
            {
                "slot": slot,
                "gpu_block_ids": list(gpu_block_ids),
                "byte_lengths": list(byte_lengths),
            }
        )
        # Model a driver that accepted work before reporting an enqueue error.
        self.active_slots.add(slot)
        if self.fail_enqueue_at == call:
            raise RuntimeError(f"injected copy failure at enqueue {call}")
        return 0.02

    def drain(self, *, after_error: bool = False) -> float:
        self.events.append(f"drain:{after_error}")
        self.drain_calls.append(after_error)
        if not after_error and (
            self.fail_normal_drain or self.fail_normal_drain_times > 0
        ):
            if self.fail_normal_drain_times > 0:
                self.fail_normal_drain_times -= 1
            raise RuntimeError("injected final drain failure")
        if self.fail_error_drain and after_error:
            raise RuntimeError("injected error-path drain failure")
        self.active_slots.clear()
        return 0.03

    def release(self) -> None:
        self.events.append("copy.release")
        self.release_attempts += 1
        if self.active_slots:
            raise AssertionError("released copy engine with active slots")
        if self.fail_release_times > 0:
            self.fail_release_times -= 1
            raise RuntimeError("injected buffer release failure")
        self.release_calls += 1


class _FakeDescriptorBuffer:
    """Minimal CPU descriptor tensor used by the real copy-engine methods."""

    def __init__(self, size: int) -> None:
        self.values = [0] * size

    def __getitem__(self, key: slice) -> _FakeDescriptorBuffer:
        view = _FakeDescriptorBuffer(0)
        view.values = self.values[key]
        return view

    def numpy(self) -> list[int]:
        return self.values


class _FakePayloadTensor:
    def __init__(self, ptr: int, rows: int) -> None:
        self._ptr = ptr
        self.shape = (rows, PAGE_SIZE)

    def data_ptr(self) -> int:
        return self._ptr


class _FakeEvent:
    def __init__(
        self,
        events: list[str],
        *,
        fail_record: bool = False,
        fail_sync: bool = False,
    ) -> None:
        self.events = events
        self.fail_record = fail_record
        self.fail_sync = fail_sync

    def record(self, _stream: _FakeStream) -> None:
        self.events.append("event.record")
        if self.fail_record:
            raise RuntimeError("injected event record failure")

    def synchronize(self) -> None:
        self.events.append("event.synchronize")
        if self.fail_sync:
            raise RuntimeError("injected event synchronize failure")


class _FakeStream:
    cuda_stream = 123

    def __init__(
        self,
        events: list[str],
        *,
        fail_sync: bool = False,
    ) -> None:
        self.events = events
        self.fail_sync = fail_sync
        self.sync_calls = 0
        self.observe_active: Any = None
        self.active_seen: list[bool] = []

    def synchronize(self) -> None:
        self.events.append("stream.synchronize")
        self.sync_calls += 1
        if self.observe_active is not None:
            self.active_seen.append(bool(self.observe_active.active))
        if self.fail_sync:
            raise RuntimeError("injected stream synchronize failure")


def _bare_cuda_copy_engine(
    events: list[str],
    *,
    event: _FakeEvent | None = None,
    stream: _FakeStream | None = None,
) -> tuple[CudaHostBounceCopyEngine, Any, _FakeStream]:
    """Build the real engine state machine without allocating CUDA objects."""

    engine = object.__new__(CudaHostBounceCopyEngine)
    engine.page_size_bytes = PAGE_SIZE
    engine.blocks_per_slot = BLOCKS_PER_SLOT
    engine.slot_count = 2
    engine._closed = False
    engine._gpu_tensors = cast(Any, [_FakePayloadTensor(20_000, 512)])
    engine._bounce_tensors = cast(
        Any, [_FakePayloadTensor(10_000, 2 * BLOCKS_PER_SLOT)]
    )
    event = event or _FakeEvent(events)
    slot = SimpleNamespace(
        src_ptrs=_FakeDescriptorBuffer(BLOCKS_PER_SLOT),
        dst_ptrs=_FakeDescriptorBuffer(BLOCKS_PER_SLOT),
        sizes=_FakeDescriptorBuffer(BLOCKS_PER_SLOT),
        event=event,
        active=False,
    )
    idle_slot = SimpleNamespace(
        src_ptrs=_FakeDescriptorBuffer(BLOCKS_PER_SLOT),
        dst_ptrs=_FakeDescriptorBuffer(BLOCKS_PER_SLOT),
        sizes=_FakeDescriptorBuffer(BLOCKS_PER_SLOT),
        event=_FakeEvent(events),
        active=False,
    )
    engine._slots = cast(Any, [slot, idle_slot])
    stream = stream or _FakeStream(events)
    engine._stream = stream  # type: ignore[assignment]
    stream.observe_active = slot
    return engine, slot, stream


@pytest.mark.parametrize("failure_stage", ["submit", "event"])
def test_real_copy_engine_error_drain_syncs_stream_without_active_event(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    events: list[str] = []
    event = _FakeEvent(events, fail_record=failure_stage == "event")
    engine, slot, stream = _bare_cuda_copy_engine(events, event=event)

    def submit(*_args: Any, **_kwargs: Any) -> None:
        events.append("swap_blocks_batch")
        if failure_stage == "submit":
            raise RuntimeError("injected batch submit failure")

    monkeypatch.setattr(host_bounce.ops, "swap_blocks_batch", submit)
    monkeypatch.setattr(
        host_bounce,
        "current_platform",
        SimpleNamespace(stream=lambda _stream: contextlib.nullcontext()),
    )

    with pytest.raises(RuntimeError, match="injected"):
        engine.enqueue(0, [7], [PAGE_SIZE])

    # No completion event was established, but the driver may have accepted a
    # prefix. Error drain must therefore synchronize the stream unconditionally.
    assert slot.active is False
    engine.drain(after_error=True)
    assert stream.sync_calls == 1
    assert "stream.synchronize" in events


def test_real_copy_engine_event_failure_does_not_clear_before_stream_sync() -> None:
    events: list[str] = []
    event = _FakeEvent(events, fail_sync=True)
    stream = _FakeStream(events, fail_sync=True)
    engine, slot, stream = _bare_cuda_copy_engine(
        events,
        event=event,
        stream=stream,
    )
    slot.active = True

    with pytest.raises(RuntimeError, match="event synchronize failure"):
        engine.drain(after_error=True)

    # The fallback stream sync observed the slot as active. Because that sync
    # also failed, completion remains unknown and active must stay set.
    assert stream.active_seen == [True]
    assert slot.active is True


def _transport(
    *,
    adapter: _FakeAdapter | None = None,
    copy_engine: _FakeCopyEngine | None = None,
    slot_count: int = 2,
) -> tuple[
    RemoteG2HostBounceTransport,
    _FakeAdapter,
    _FakeCopyEngine,
    list[str],
]:
    if adapter is None and copy_engine is None:
        events: list[str] = []
    elif adapter is not None:
        events = adapter.events
    else:
        assert copy_engine is not None
        events = copy_engine.events
    adapter = adapter or _FakeAdapter(events)
    copy_engine = copy_engine or _FakeCopyEngine(events)
    assert adapter.events is copy_engine.events
    return (
        RemoteG2HostBounceTransport(
            adapter=adapter,  # type: ignore[arg-type]
            copy_engine=copy_engine,
            page_size_bytes=PAGE_SIZE,
            blocks_per_slot=BLOCKS_PER_SLOT,
            slot_count=slot_count,
        ),
        adapter,
        copy_engine,
        events,
    )


def _run_transfer(
    transport: RemoteG2HostBounceTransport,
    block_count: int,
) -> HostBounceTransferStats:
    return transport.transfer(
        "source",
        peer_byte_offsets=[(1000 + block) * PAGE_SIZE for block in range(block_count)],
        gpu_block_ids=list(reversed(range(block_count))),
        byte_lengths=[PAGE_SIZE] * block_count,
    )


@pytest.mark.parametrize(
    ("block_count", "chunk_sizes", "slots"),
    [
        (65, [64, 1], [0, 1]),
        (129, [64, 64, 1], [0, 1, 0]),
    ],
)
def test_chunks_slots_and_partial_tail(
    block_count: int,
    chunk_sizes: list[int],
    slots: list[int],
) -> None:
    transport, adapter, copy, events = _transport()

    stats = _run_transfer(transport, block_count)

    assert stats.logical_bytes == block_count * PAGE_SIZE
    assert stats.chunk_count == len(chunk_sizes)
    assert copy.wait_calls == slots
    assert [call["slot"] for call in copy.enqueue_calls] == slots
    assert [len(call["gpu_block_ids"]) for call in copy.enqueue_calls] == chunk_sizes
    assert [len(call["byte_lengths"]) for call in adapter.read_calls] == chunk_sizes
    for call, slot, size in zip(adapter.read_calls, slots, chunk_sizes):
        expected_start = slot * BLOCKS_PER_SLOT * PAGE_SIZE
        assert call["local_byte_offsets"] == [
            expected_start + block * PAGE_SIZE for block in range(size)
        ]
        assert call["byte_lengths"] == [PAGE_SIZE] * size

    # The third chunk reuses slot 0 only after wait_slot(0), and before its READ.
    if block_count == 129:
        second_enqueue = events.index("enqueue:1:1")
        reuse_wait = events.index("wait:0", second_enqueue)
        tail_read = events.index("read:2")
        tail_enqueue = events.index("enqueue:0:2")
        assert second_enqueue < reuse_wait < tail_read < tail_enqueue


def test_single_slot_serializes_each_read_after_previous_h2d() -> None:
    transport, adapter, copy, events = _transport(slot_count=1)

    stats = _run_transfer(transport, 129)

    assert stats.chunk_count == 3
    assert copy.wait_calls == [0, 0, 0]
    assert [call["slot"] for call in copy.enqueue_calls] == [0, 0, 0]
    assert [call["local_byte_offsets"][0] for call in adapter.read_calls] == [0, 0, 0]
    for chunk in (1, 2):
        previous_enqueue = events.index(f"enqueue:0:{chunk - 1}")
        reuse_wait = events.index("wait:0", previous_enqueue)
        next_read = events.index(f"read:{chunk}")
        assert previous_enqueue < reuse_wait < next_read


def test_transfer_stats_include_stage_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, _adapter, _copy, _events = _transport()
    timestamps = iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.3])
    monkeypatch.setattr(
        host_bounce,
        "time",
        SimpleNamespace(perf_counter=timestamps.__next__),
    )

    stats = _run_transfer(transport, 129)

    assert stats.logical_bytes == 129 * PAGE_SIZE
    assert stats.chunk_count == 3
    assert stats.read_seconds == pytest.approx(0.6)
    assert stats.copy_enqueue_seconds == pytest.approx(0.06)
    # Only slot 0 is reused (0.01), then the final drain contributes 0.03.
    assert stats.copy_wait_seconds == pytest.approx(0.04)


@pytest.mark.parametrize("block_count", [65, 129])
def test_late_partial_page_is_rejected_before_first_read_and_does_not_poison(
    block_count: int,
) -> None:
    transport, adapter, copy, _events = _transport()
    lengths = [PAGE_SIZE] * block_count
    lengths[-1] -= 1

    with pytest.raises(ValueError, match="full-page descriptors"):
        transport.transfer(
            "source",
            peer_byte_offsets=[block * PAGE_SIZE for block in range(block_count)],
            gpu_block_ids=list(range(block_count)),
            byte_lengths=lengths,
        )

    assert adapter.read_calls == []
    assert copy.wait_calls == []
    assert copy.enqueue_calls == []

    # Validation happens before entering the pipeline and must not poison it.
    stats = _run_transfer(transport, 1)
    assert stats.logical_bytes == PAGE_SIZE
    assert len(adapter.read_calls) == 1


def test_nixl_failure_drains_and_poisons_transport() -> None:
    events: list[str] = []
    adapter = _FakeAdapter(events, fail_read_at=1)
    copy = _FakeCopyEngine(events)
    transport, _, _, _ = _transport(adapter=adapter, copy_engine=copy)

    with pytest.raises(HostBounceTransferError) as exc_info:
        _run_transfer(transport, 129)

    assert exc_info.value.safe_to_release_source_lease is False
    assert copy.drain_calls == [True]
    assert copy.active_slots == set()
    reads_after_failure = len(adapter.read_calls)
    with pytest.raises(RuntimeError, match="poisoned"):
        _run_transfer(transport, 1)
    assert len(adapter.read_calls) == reads_after_failure


def test_copy_enqueue_failure_drains_and_poisons_transport() -> None:
    events: list[str] = []
    adapter = _FakeAdapter(events)
    copy = _FakeCopyEngine(events, fail_enqueue_at=0)
    transport, _, _, _ = _transport(adapter=adapter, copy_engine=copy)

    with pytest.raises(HostBounceTransferError) as exc_info:
        _run_transfer(transport, 65)

    assert exc_info.value.safe_to_release_source_lease is True
    assert copy.drain_calls == [True]
    assert copy.active_slots == set()
    with pytest.raises(RuntimeError, match="poisoned"):
        _run_transfer(transport, 1)


def test_final_drain_failure_retries_error_drain_and_poisons() -> None:
    events: list[str] = []
    adapter = _FakeAdapter(events)
    copy = _FakeCopyEngine(events, fail_normal_drain=True)
    transport, _, _, _ = _transport(adapter=adapter, copy_engine=copy)

    with pytest.raises(HostBounceTransferError) as exc_info:
        _run_transfer(transport, 1)

    assert exc_info.value.safe_to_release_source_lease is True
    assert copy.drain_calls == [False, True]
    assert copy.active_slots == set()
    with pytest.raises(RuntimeError, match="poisoned"):
        _run_transfer(transport, 1)


def test_error_path_drain_failure_after_read_keeps_source_lease_safe() -> None:
    events: list[str] = []
    adapter = _FakeAdapter(events)
    copy = _FakeCopyEngine(
        events,
        fail_enqueue_at=0,
        fail_error_drain=True,
    )
    transport, _, _, _ = _transport(adapter=adapter, copy_engine=copy)

    with pytest.raises(HostBounceTransferError) as exc_info:
        _run_transfer(transport, 1)

    assert exc_info.value.safe_to_release_source_lease is True
    assert copy.drain_calls == [True]
    assert copy.active_slots == {0}
    with pytest.raises(RuntimeError, match="poisoned"):
        _run_transfer(transport, 1)

    # Unknown DMA completion also forbids deregistration/free during close.
    with pytest.raises(RuntimeError, match="error-path drain failure"):
        transport.close()
    assert adapter.close_calls == 0
    assert copy.release_calls == 0


def test_close_orders_drain_deregister_release_and_is_idempotent() -> None:
    transport, adapter, copy, events = _transport()

    transport.close()
    transport.close()

    assert events == ["drain:False", "adapter.close", "copy.release"]
    assert adapter.close_calls == 1
    assert copy.release_calls == 1
    assert copy.drain_calls == [False]
    with pytest.raises(RuntimeError, match="closed"):
        _run_transfer(transport, 1)


def test_close_drain_failure_retains_registration_and_can_retry() -> None:
    events: list[str] = []
    adapter = _FakeAdapter(events)
    copy = _FakeCopyEngine(events, fail_normal_drain_times=1)
    transport, _, _, _ = _transport(adapter=adapter, copy_engine=copy)

    with pytest.raises(RuntimeError, match="final drain failure"):
        transport.close()
    assert adapter.close_calls == 0
    assert copy.release_attempts == 0
    with pytest.raises(RuntimeError, match="closing"):
        _run_transfer(transport, 1)

    transport.close()
    assert adapter.close_calls == 1
    assert copy.release_calls == 1
    assert events == [
        "drain:False",
        "drain:False",
        "adapter.close",
        "copy.release",
    ]


def test_close_deregister_failure_retains_buffers_and_can_retry() -> None:
    events: list[str] = []
    adapter = _FakeAdapter(events, fail_close_times=1)
    copy = _FakeCopyEngine(events)
    transport, _, _, _ = _transport(adapter=adapter, copy_engine=copy)

    with pytest.raises(RuntimeError, match="deregistration failure"):
        transport.close()
    assert copy.release_attempts == 0
    assert adapter.closed is False

    transport.close()
    assert adapter.closed is True
    assert copy.release_calls == 1
    assert events == [
        "drain:False",
        "adapter.close",
        "drain:False",
        "adapter.close",
        "copy.release",
    ]


def test_close_release_failure_retries_without_rederegistering() -> None:
    events: list[str] = []
    adapter = _FakeAdapter(events)
    copy = _FakeCopyEngine(events, fail_release_times=1)
    transport, _, _, _ = _transport(adapter=adapter, copy_engine=copy)

    with pytest.raises(RuntimeError, match="buffer release failure"):
        transport.close()
    assert adapter.closed is True
    assert copy.release_calls == 0

    transport.close()
    assert adapter.close_calls == 1
    assert copy.release_calls == 1
    assert copy.release_attempts == 2


class _HandlerTransport:
    def __init__(
        self,
        events: list[str],
        *,
        failure: Exception | None = None,
        fail_close_times: int = 0,
    ) -> None:
        self.events = events
        self.failure = failure
        self.close_calls = 0
        self.fail_close_times = fail_close_times
        self.calls: list[dict[str, Any]] = []

    def transfer(
        self,
        peer_name: str,
        *,
        peer_byte_offsets: Sequence[int],
        gpu_block_ids: Sequence[int],
        byte_lengths: Sequence[int],
    ) -> HostBounceTransferStats:
        self.events.append("transport.transfer")
        self.calls.append(
            {
                "peer_name": peer_name,
                "peer_byte_offsets": list(peer_byte_offsets),
                "gpu_block_ids": list(gpu_block_ids),
                "byte_lengths": list(byte_lengths),
            }
        )
        # The real transport drains all slot events before either return path.
        self.events.append("transport.drain_done")
        if self.failure is not None:
            raise self.failure
        logical_bytes = sum(int(value) for value in byte_lengths)
        return HostBounceTransferStats(
            logical_bytes=logical_bytes,
            chunk_count=1,
            read_seconds=0.1,
            copy_enqueue_seconds=0.02,
            copy_wait_seconds=0.03,
        )

    def close(self) -> None:
        self.events.append("transport.close")
        self.close_calls += 1
        if self.fail_close_times > 0:
            self.fail_close_times -= 1
            raise RuntimeError("injected transport close failure")


def _load_specs(
    block_count: int,
    *,
    lease_id: str = "lease-1",
) -> tuple[RemoteG2LoadSpec, GPULoadStoreSpec]:
    blocks = [
        _RemoteBlockHandle(
            block_hash=10_000 + block,
            descriptor_generation=1,
            byte_offset=(2000 + block) * PAGE_SIZE,
            byte_length=PAGE_SIZE,
        )
        for block in range(block_count)
    ]
    src = RemoteG2LoadSpec(
        peer_name="source",
        lease_id=lease_id,
        blocks=blocks,
        source_worker_id=1,
        source_dp_rank=0,
    )
    dst = GPULoadStoreSpec(
        block_ids=list(range(3000, 3000 + block_count)),
        group_sizes=[block_count],
        block_indices=[0],
    )
    return src, dst


def _handler(
    events: list[str],
    transport: _HandlerTransport,
) -> tuple[RemoteG2TransferHandler, list[tuple[bool, int]]]:
    metrics: list[tuple[bool, int]] = []

    def release(lease_id: str) -> None:
        events.append(f"lease.release:{lease_id}")

    def record(success: bool, num_bytes: int) -> None:
        events.append(f"metric:{success}:{num_bytes}")
        metrics.append((success, num_bytes))

    adapter = _FakeAdapter(events)
    return (
        RemoteG2TransferHandler(
            adapter=adapter,  # type: ignore[arg-type]
            gpu_page_size_bytes=PAGE_SIZE,
            ensure_peer=lambda _peer: True,
            on_load_done=release,
            host_bounce=transport,  # type: ignore[arg-type]
            on_host_bounce_result=record,
        ),
        metrics,
    )


def test_handler_success_releases_lease_after_drain_and_records_metrics() -> None:
    events: list[str] = []
    transport = _HandlerTransport(events)
    handler, metrics = _handler(events, transport)
    src, dst = _load_specs(3)

    assert handler.submit_load(71, src, dst) is True

    assert events.index("transport.drain_done") < events.index("lease.release:lease-1")
    assert metrics == [(True, 3 * PAGE_SIZE)]
    result = handler.get_finished()
    assert len(result) == 1
    assert result[0].job_id == 71
    assert result[0].success is True
    assert result[0].transfer_size == 3 * PAGE_SIZE
    assert result[0].transfer_time is not None
    assert handler.get_finished() == []


def test_handler_failure_raises_synchronously_then_releases_and_records_failure() -> (
    None
):
    events: list[str] = []
    transport = _HandlerTransport(
        events,
        failure=HostBounceTransferError(
            "copy failed after a successful drain",
            safe_to_release_source_lease=True,
        ),
    )
    handler, metrics = _handler(events, transport)
    src, dst = _load_specs(2)

    with pytest.raises(RuntimeError, match="host-bounce transfer failed"):
        handler.submit_load(72, src, dst)

    assert events.index("transport.drain_done") < events.index("lease.release:lease-1")
    assert metrics == [(False, 0)]
    assert handler.get_finished() == []


def test_uncertain_source_read_keeps_lease_for_explicit_cleanup_or_teardown() -> None:
    events: list[str] = []
    transport = _HandlerTransport(
        events,
        failure=HostBounceTransferError(
            "source READ completion is unknown",
            safe_to_release_source_lease=False,
        ),
    )
    handler, metrics = _handler(events, transport)
    src, dst = _load_specs(2)

    with pytest.raises(RuntimeError, match="host-bounce transfer failed"):
        handler.submit_load(73, src, dst)

    assert "lease.release:lease-1" not in events
    assert metrics == [(False, 0)]
    assert handler.get_finished() == []


def test_handler_shutdown_closes_host_bounce_once() -> None:
    events: list[str] = []
    transport = _HandlerTransport(events)
    handler, _metrics = _handler(events, transport)

    handler.shutdown()
    handler.shutdown()

    assert transport.close_calls == 1
    assert events == ["transport.close"]


def test_handler_shutdown_close_failure_can_retry() -> None:
    events: list[str] = []
    transport = _HandlerTransport(events, fail_close_times=1)
    handler, _metrics = _handler(events, transport)

    with pytest.raises(RuntimeError, match="transport close failure"):
        handler.shutdown()
    handler.shutdown()
    handler.shutdown()

    assert transport.close_calls == 2
    assert events == ["transport.close", "transport.close"]
