# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SimpleCPUOffloadConnector metrics (no GPU required)."""

from types import SimpleNamespace

import pytest
from prometheus_client import Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    _DEPRECATED_TOTAL_BYTES,
    OffloadingConnectorStats,
    _MetricType,
    _StatsKey,
    _TransferMetricName,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
    SimpleCPUOffloadPromMetrics,
)
from vllm.v1.simple_kv_offload import copy_backend
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyEvent, _EventPairPool
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.metrics import (
    PoolMetricName,
    get_pool_gauge_definitions,
)
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

pytestmark = pytest.mark.skip_global_cleanup

STORE_BYTES = _TransferMetricName.STORE_BYTES
STORE_TIME = _TransferMetricName.STORE_TIME
STORE_SIZE = _TransferMetricName.STORE_SIZE
LOAD_BYTES = _TransferMetricName.LOAD_BYTES


class _FakeMetric:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.observed: list[float] = []
        self.increments: list[float] = []
        self.set_values: list[float] = []
        self.labelvalues: tuple[object, ...] = ()

    def labels(self, *labelvalues):
        child = _FakeMetric(**self.kwargs)
        child.labelvalues = labelvalues
        return child

    def observe(self, value):
        self.observed.append(value)

    def inc(self, value):
        self.increments.append(value)

    def set(self, value):
        self.set_values.append(value)


class _FakeVllmConfig:
    def __init__(self):
        self.kv_transfer_config = SimpleNamespace(kv_connector_extra_config={})


def _make_prom_metrics() -> SimpleCPUOffloadPromMetrics:
    return SimpleCPUOffloadConnector.build_prom_metrics(
        _FakeVllmConfig(),  # type: ignore[arg-type]
        {Gauge: _FakeMetric, Counter: _FakeMetric, Histogram: _FakeMetric},
        ["model_name", "engine"],
        {0: ["model", "0"]},
    )


class _FakeTimingEvent:
    """Stand-in for a CUDA timing event; elapsed_time returns fixed ms."""

    def __init__(self, elapsed_ms: float = 0.0, enable_timing: bool = False):
        self._elapsed_ms = elapsed_ms

    def elapsed_time(self, _end) -> float:
        return self._elapsed_ms


# --- Connector stats/prom factories ---


def test_build_kv_connector_stats_none_and_dict():
    empty = SimpleCPUOffloadConnector.build_kv_connector_stats(data=None)
    assert isinstance(empty, OffloadingConnectorStats)
    assert empty.is_empty()

    data = {
        _StatsKey.TYPES: {STORE_BYTES: _MetricType.COUNTER},
        _StatsKey.DATA: {STORE_BYTES: {(): 42}},
    }
    stats = SimpleCPUOffloadConnector.build_kv_connector_stats(data=data)
    assert isinstance(stats, OffloadingConnectorStats)
    assert not stats.is_empty()
    assert stats.data[_StatsKey.DATA][STORE_BYTES][()] == 42


def test_build_prom_metrics_type():
    prom = _make_prom_metrics()
    assert isinstance(prom, SimpleCPUOffloadPromMetrics)


# --- Prometheus observe() end-to-end (the path #41790 never tested) ---


def test_prom_observe_transfer_and_pool_gauges():
    prom = _make_prom_metrics()
    prom.observe(
        {
            _StatsKey.TYPES: {
                STORE_BYTES: _MetricType.COUNTER,
                STORE_TIME: _MetricType.COUNTER,
                STORE_SIZE: _MetricType.HISTOGRAM,
                PoolMetricName.TOTAL_BLOCKS: _MetricType.GAUGE,
                PoolMetricName.USAGE_PERC: _MetricType.GAUGE,
            },
            _StatsKey.DATA: {
                STORE_BYTES: {(): 500},
                STORE_TIME: {(): 0.25},
                STORE_SIZE: {(): [100, 400]},
                PoolMetricName.TOTAL_BLOCKS: {(): 7},
                PoolMetricName.USAGE_PERC: {(): 0.5},
            },
        }
    )
    assert prom.offloading_metrics[(0, STORE_BYTES, ())].increments == [500]
    assert prom.offloading_metrics[(0, STORE_TIME, ())].increments == [0.25]
    assert prom.offloading_metrics[(0, STORE_SIZE, ())].observed == [100, 400]
    assert prom.offloading_metrics[(0, PoolMetricName.TOTAL_BLOCKS, ())].set_values == [
        7
    ]
    assert prom.offloading_metrics[(0, PoolMetricName.USAGE_PERC, ())].set_values == [
        0.5
    ]


def test_prom_registers_all_pool_gauges():
    prom = _make_prom_metrics()
    for name in get_pool_gauge_definitions():
        assert name in prom._offloading_metric_defs


def test_prom_no_deprecated_series():
    """The deprecated transfer_type metrics belong to the native CPU spec only."""
    prom = _make_prom_metrics()
    assert _DEPRECATED_TOTAL_BYTES not in prom._offloading_metric_defs
    assert prom._observe_deprecated_metrics is False
    # Observing a store must NOT populate the deprecated counters.
    prom.observe(
        {
            _StatsKey.TYPES: {STORE_BYTES: _MetricType.COUNTER},
            _StatsKey.DATA: {STORE_BYTES: {(): 8}},
        }
    )
    assert prom.counter_kv_bytes == {}


def test_prom_rejects_undeclared_metric():
    prom = _make_prom_metrics()
    with pytest.raises(AssertionError):
        prom.observe(
            {
                _StatsKey.TYPES: {"vllm:not_a_metric": _MetricType.COUNTER},
                _StatsKey.DATA: {"vllm:not_a_metric": {(): 1}},
            }
        )


# --- Scheduler-side pool gauges ---


def test_scheduler_pool_gauges():
    stub = SimpleNamespace(
        cpu_block_pool=SimpleNamespace(num_gpu_blocks=8, get_num_free_blocks=lambda: 6),
        _reqs_to_load={"r1": object(), "r2": object()},
        _store_event_to_blocks={5: object()},
        _abandoned_reqs_to_load={},
        _abandoned_store_event_to_blocks={},
    )
    stats = SimpleCPUOffloadScheduler.get_kv_connector_stats(stub)
    assert isinstance(stats, OffloadingConnectorStats)
    values = stats.data[_StatsKey.DATA]
    assert values[PoolMetricName.TOTAL_BLOCKS][()] == 7  # num_gpu_blocks - 1 (null)
    assert values[PoolMetricName.FREE_BLOCKS][()] == 6
    assert values[PoolMetricName.USED_BLOCKS][()] == 1
    assert values[PoolMetricName.USAGE_PERC][()] == pytest.approx(1 / 7)
    assert values[PoolMetricName.PENDING_LOADS][()] == 2
    assert values[PoolMetricName.PENDING_STORES][()] == 1


# --- Worker-side transfer recording + reset-on-read ---


def _make_worker() -> SimpleCPUOffloadWorker:
    return SimpleCPUOffloadWorker(
        vllm_config=None,  # type: ignore[arg-type]
        kv_cache_config=None,
        cpu_capacity_bytes=1024,
    )


def test_worker_record_transfer_populates_stats():
    worker = _make_worker()
    store_ev = DmaCopyEvent(
        event_idx=0,
        start_event=_FakeTimingEvent(elapsed_ms=2.0),
        end_event=_FakeTimingEvent(),
        num_bytes=500,
        is_store=True,
    )
    load_ev = DmaCopyEvent(
        event_idx=1,
        start_event=_FakeTimingEvent(elapsed_ms=1.0),
        end_event=_FakeTimingEvent(),
        num_bytes=200,
        is_store=False,
    )
    worker._record_transfer(store_ev)
    worker._record_transfer(load_ev)

    reduced = worker._stats.reduce()
    assert reduced[STORE_BYTES] == 500
    assert reduced[STORE_TIME] == pytest.approx(0.002)  # 2.0 ms -> s
    assert reduced[f"{STORE_SIZE}_count"] == 1
    assert reduced[LOAD_BYTES] == 200


def test_worker_record_transfer_skips_zero_bytes():
    worker = _make_worker()
    worker._record_transfer(
        DmaCopyEvent(
            event_idx=0,
            start_event=_FakeTimingEvent(elapsed_ms=5.0),
            end_event=_FakeTimingEvent(),
            num_bytes=0,
            is_store=True,
        )
    )
    assert worker._stats.is_empty()


def test_worker_get_kv_connector_stats_resets():
    worker = _make_worker()
    assert worker.get_kv_connector_stats() is None  # empty -> None
    worker._record_transfer(
        DmaCopyEvent(
            event_idx=0,
            start_event=_FakeTimingEvent(elapsed_ms=1.0),
            end_event=_FakeTimingEvent(),
            num_bytes=64,
            is_store=True,
        )
    )
    stats = worker.get_kv_connector_stats()
    assert isinstance(stats, OffloadingConnectorStats)
    assert stats.data[_StatsKey.DATA][STORE_BYTES][()] == 64
    # Second call returns None (reset on read).
    assert worker.get_kv_connector_stats() is None


def test_release_event_is_idempotent():
    calls = []
    ev = DmaCopyEvent(
        event_idx=0,
        start_event=_FakeTimingEvent(),
        end_event=_FakeTimingEvent(),
        num_bytes=1,
        is_store=True,
        release=lambda: calls.append(1),
    )
    SimpleCPUOffloadWorker._release_event(ev)
    SimpleCPUOffloadWorker._release_event(ev)  # already released -> no-op
    assert calls == [1]
    assert ev.release is None


# --- Event pool reuse ---


def test_event_pool_reuse(monkeypatch):
    monkeypatch.setattr(copy_backend.torch, "Event", _FakeTimingEvent)
    pool = _EventPairPool(2)
    a = pool.acquire()
    b = pool.acquire()
    c = pool.acquire()  # pool empty -> freshly allocated
    assert a != b and b != c
    pool.release(*a)
    pool.release(*b)
    # Subsequent acquires return recycled pairs (identity), not new ones.
    reacquired = {id(pool.acquire()[0]), id(pool.acquire()[0])}
    assert reacquired == {id(a[0]), id(b[0])}
