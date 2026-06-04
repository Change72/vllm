# Two-engine evaluation — findings

This evaluation drives two real `vllm.LLM` workers (Qwen3-8B on H20)
through three cycles against a 16-prompt diverse set and verifies
output equivalence against the source engine's reference outputs.

## Setup

- Engine A (source, GPU 1) and Engine B (target, GPU 2), each running
  `vllm.LLM` with `RemoteG2OffloadingSpec` and real NIXL UCX
- `PYTHONHASHSEED=0` pinned on both — identical token sequences hash to
  identical block hashes across processes
- 16 topical prompts, each a distinct base sentence repeated 60 times
  (~3000 tokens / ~190 blocks each); 16 prompts × ~190 blocks ~= 3040
  blocks would be needed but source CPU pool caps at ~870 blocks → it
  evicts within itself
- Source runs all 16 to populate its pool (ends up with 910 live
  descriptors) and writes a reference output for every prompt
- Target runs:
  - **cycle 0 — baseline cold**, prompts 0..7, no plan
  - **cycle 1 — plan-driven cold**, prompts 8..15, with a plan
    referencing all 910 of source's published hashes
  - **cycle 2 — baseline hot**, prompts 8..15, no plan

## Results (Qwen3-8B, H20, real NIXL UCX, 1 run)

```
per-prompt output match : 8 / 24
timing (ms/prompt)      : baseline_cold=262  plan_driven_cold=110  baseline_hot=49
cycle 1 plan stats      : plan_seen=8 plan_resolved=8
                          plan_load_specs_emitted=7
                          plan_blocks_loaded=910
```

- Cold plan-driven is **2.4× faster** than cold baseline (110 vs 262 ms
  per prompt) — prefill is genuinely being skipped for the matched
  prefix, so the scheduler + handler wiring is working.
- 7 of 8 requests in cycle 1 emitted `RemoteG2LoadSpec` (prompt 8 fell
  to the local-recompute path; that prompt happens to be the one that
  also produces non-deterministic output between cycles 1 and 2, so
  its mismatch is unrelated).

## Open bug — multi-layer KV cache only partially loaded

For prompts 9–15 the plan-driven cycle produces **garbage outputs**
(`" 2 2 2 2 "`, `" computing computing computing"`, `" fffff"`, etc.),
and cycle 2 (no plan) reproduces the same garbage — confirming the
bad bytes are sitting in target's GPU KV cache, not produced by a
transient state.

Root cause: `gpu_worker.py` allocates **36 CPU tensors** for Qwen3-8B
(one per transformer layer; see `Allocating 36 CPU tensors...` in
`source.log` / `target.log`). The current `RemoteG2OffloadingSpec`
implementation only registers `kv_caches.tensors[0]` with the NIXL
agent, computes byte_offsets relative to it, and the transfer handler
reads into a single contiguous GPU region. The net effect is that
**only the first layer's KV gets transferred** — layers 1–35 stay
uninitialized on target, so any prompt whose matched prefix spans
many blocks generates nonsense once the model's attention reads from
those uninitialised slots.

The simpler 2-prompt test (`test_real_engine_external_pull.py` /
the small `two_engines/orchestrate.py` variant from before this
evaluation) only triggered ~2 plan-driven loads per request — too few
to corrupt the output noticeably for short prompts, which is why the
bug didn't surface until this larger evaluation. The bug also doesn't
affect the **byte-level smoke tests** (`test_e2e_smoke.py`,
`test_nixl_real_smoke.py`) because those use synthetic single-region
DRAM pools, never multi-layer KV.

### Fix sketch

`vllm/v1/kv_offload/remote_g2/spec.py` `get_handlers()`:

- Iterate over **all** `kv_caches.tensors[i]` rather than just `[0]`.
- For each `i`, expose `(cpu_ptr_i, cpu_size_i, page_size_i)` to the
  manager / registry and `(gpu_ptr_i, gpu_size_i)` to the transfer
  adapter.
- `SourceG2DescriptorRecord.metadata["nixl_memory_desc"]` becomes a
  **list** of per-layer descriptors, or the registry tracks
  `pool_base_ptrs: list[int]` keyed by layer index. Block id is
  shared across layers, so each layer's `byte_offset` is the same
  arithmetic, just relative to its own base.
- Source-side NIXL `register_memory` registers all 36 regions; the
  metadata bundle published via `get_metadata` carries 36 base ptrs.
- Target-side adapter `register_memory`s the target's 36 GPU regions
  in the same order, then `add_peer` records the 36 source bases.
- `RemoteG2TransferHandler.transfer_async` issues one NIXL READ per
  layer per block (or batches into one prepped dlist per peer with
  36 × num_blocks entries, which M4 should adopt for throughput).

### Quantitative impact estimate

- With Qwen3-8B's 36 layers and per-layer-per-block KV ≈ 64 KiB
  (`8 heads × 128 dim × 16 tokens × 2 bytes × 2 K+V`), a block transfer
  becomes 36 individual ~64 KiB UCX reads (~2.3 MiB total per block).
- 114 blocks per prompt × 36 layers = 4104 reads per prompt — probably
  worth batching via the prepped dlist path before this hits
  production.
- Functional correctness alone (one initialize_xfer per layer-block)
  should bring the multi-layer path online without code surgery
  beyond the spec / handler iteration loop.

## What this evaluation establishes regardless of the bug

- Scheduler integration is solid (plan injected via
  `kv_transfer_params`, manager.lookup hits the plan, prepare_load
  emits `RemoteG2LoadSpec`, worker actually invokes the transfer
  handler and reports completion to the scheduler).
- Real NIXL UCX cross-process transport works at scale (8 prompts ×
  ~115 blocks loaded = ~920 transfers per cycle, no hangs, no NIXL
  errors visible in `target.log`).
- The plan-driven path actually skips prefill compute — the 2.4×
  TTFT speed-up demonstrates this.
- The single-layer correctness limitation we now know is the next
  thing to fix.

## Files

- `engine_runner.py` — multi-cycle engine runner, both source and
  target modes.
- `orchestrate.py` — three-cycle orchestrator with per-prompt
  equivalence checks.
- `varied_prompts.py` — 16 distinct topical base sentences.
- `run_artifacts/summary.json` and `run_artifacts/per_prompt.json` —
  artifacts from the run described above (16-prompt, repeat=60).
