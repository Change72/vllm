# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metric names and definitions for SimpleCPUOffloadConnector.

Transfer metrics reuse the shared ``vllm:kv_offload_*`` family defined in
``offloading/metrics.py``; the CPU-pool gauges below are specific to the
SimpleCPUOffloadConnector's own CPU ``BlockPool``.
"""

from vllm.v1.kv_offload.base import (
    OffloadingGaugeMetadata,
    OffloadingMetricMetadata,
)


class PoolMetricName:
    """Gauge names for the SimpleCPUOffloadConnector CPU block pool."""

    TOTAL_BLOCKS = "vllm:simple_cpu_offload_total_blocks"
    FREE_BLOCKS = "vllm:simple_cpu_offload_free_blocks"
    USED_BLOCKS = "vllm:simple_cpu_offload_used_blocks"
    USAGE_PERC = "vllm:simple_cpu_offload_usage_perc"
    PENDING_LOADS = "vllm:simple_cpu_offload_pending_loads"
    PENDING_STORES = "vllm:simple_cpu_offload_pending_stores"


def get_pool_gauge_definitions() -> dict[str, OffloadingMetricMetadata]:
    return {
        PoolMetricName.TOTAL_BLOCKS: OffloadingGaugeMetadata(
            documentation="Total usable CPU KV cache blocks managed by "
            "SimpleCPUOffloadConnector."
        ),
        PoolMetricName.FREE_BLOCKS: OffloadingGaugeMetadata(
            documentation="Free usable CPU KV cache blocks managed by "
            "SimpleCPUOffloadConnector."
        ),
        PoolMetricName.USED_BLOCKS: OffloadingGaugeMetadata(
            documentation="Used usable CPU KV cache blocks managed by "
            "SimpleCPUOffloadConnector."
        ),
        PoolMetricName.USAGE_PERC: OffloadingGaugeMetadata(
            documentation="CPU KV cache usage for SimpleCPUOffloadConnector; "
            "1 means 100 percent usage."
        ),
        PoolMetricName.PENDING_LOADS: OffloadingGaugeMetadata(
            documentation="Requests with pending CPU-to-GPU loads in "
            "SimpleCPUOffloadConnector."
        ),
        PoolMetricName.PENDING_STORES: OffloadingGaugeMetadata(
            documentation="Store events pending worker completion in "
            "SimpleCPUOffloadConnector."
        ),
    }
