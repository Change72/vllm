# Production Metrics

vLLM exposes a number of metrics that can be used to monitor the health of the
system. These metrics are exposed via the `/metrics` endpoint on the vLLM
OpenAI compatible API server.

You can start the server using Python, or using [Docker](../deployment/docker.md):

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
```

Then query the endpoint to get the latest metrics from the server:

??? console "Output"

    ```console
    $ curl http://0.0.0.0:8000/metrics

    # HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
    # TYPE vllm:iteration_tokens_total histogram
    vllm:iteration_tokens_total_sum{model_name="unsloth/Llama-3.2-1B-Instruct"} 0.0
    vllm:iteration_tokens_total_bucket{le="1.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="8.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="16.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="32.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="64.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="128.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="256.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    vllm:iteration_tokens_total_bucket{le="512.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
    ...
    ```

The following metrics are exposed:

## General Metrics

--8<-- "docs/generated/metrics/general.inc.md"

## Speculative Decoding Metrics

--8<-- "docs/generated/metrics/spec_decode.inc.md"

## NIXL KV Connector Metrics

--8<-- "docs/generated/metrics/nixl_connector.inc.md"

## SimpleCPU KV Offload Metrics

When `SimpleCPUOffloadConnector` is enabled, vLLM reports KV transfer volume and
timing through the shared `vllm:kv_offload_*` metric family:

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:kv_offload_store_bytes` | Counter | Bytes stored GPU→CPU. |
| `vllm:kv_offload_store_time` | Counter | Store time GPU→CPU, in seconds. |
| `vllm:kv_offload_store_size` | Histogram | Store operation size, in bytes. |
| `vllm:kv_offload_load_bytes` | Counter | Bytes loaded CPU→GPU. |
| `vllm:kv_offload_load_time` | Counter | Load time CPU→GPU, in seconds. |
| `vllm:kv_offload_load_size` | Histogram | Load operation size, in bytes. |

It also reports the state of its CPU block pool:

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:simple_cpu_offload_total_blocks` | Gauge | Total usable CPU KV cache blocks. |
| `vllm:simple_cpu_offload_free_blocks` | Gauge | Free usable CPU KV cache blocks. |
| `vllm:simple_cpu_offload_used_blocks` | Gauge | Used usable CPU KV cache blocks. |
| `vllm:simple_cpu_offload_usage_perc` | Gauge | CPU KV cache usage; `1` means 100 percent. |
| `vllm:simple_cpu_offload_pending_loads` | Gauge | Requests with pending CPU→GPU loads. |
| `vllm:simple_cpu_offload_pending_stores` | Gauge | Store events pending worker completion. |

Load (`CPU_to_GPU`) samples appear only when a workload replays after GPU cache
eviction; a workload that fits in the GPU KV cache may show stores only.

## Model Flops Utilization (MFU) Performance Metrics

These metrics are available via `--enable-mfu-metrics`:

--8<-- "docs/generated/metrics/perf.inc.md"

## Deprecation Policy

Note: when metrics are deprecated in version `X.Y`, they are hidden in version `X.Y+1`
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch,
and are then removed in version `X.Y+2`.
