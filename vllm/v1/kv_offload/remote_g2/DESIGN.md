# Remote G2 KV-P2P for vLLM — Design

**Status:** POC, all milestones M1–M5 implemented and end-to-end verified
on Qwen3-8B with real Dynamo runtime + real NIXL UCX.
**Authors:** vLLM port team
**Reviewers (target):** Olga Andreeva, Adit Ranadive (NVIDIA HQ)
**Related:** TRT-LLM KV-P2P POC; Dynamo `oandreeva-kv-p2p-v1-followups`
branch (`lib/kv-router/src/remote_g2_plan.rs`).

---

## 1. Goal

Port NVIDIA's TRT-LLM Remote-G2 (host-pinned tier) KV-P2P prototype to
vLLM, exposing each vLLM worker's CPU offload pool as a NIXL-readable
source and letting peer workers fetch a request's KV prefix instead of
recomputing it.

Two constraints we hold ourselves to:

1. **Byte-compatible source RPC** with the TRT-LLM POC — the same
   pickle-over-ZMQ protocol, so existing Dynamo Router test rigs and
   the in-progress `remote_g2_plan.rs` Rust client both work against
   our vLLM source without modification.
2. **No fork of vLLM's offloading framework** — `RemoteG2OffloadingSpec`
   is a `CPUOffloadingSpec` subclass that plugs into the v1
   `OffloadingConnector`. The plan-driven path is an opt-in extension
   on top of the unmodified inherited cache management.

## 2. Non-goals

- **Cross-host UCX tuning** — we run on `--network=host` co-located
  workers; cross-host UCX configuration is deployment-time.
- **Disk tier** — Remote-G3 in TRT-LLM. Out of scope; the registry's
  tier field allows a future plug-in without protocol changes.
- **Router policy** — we don't decide which blocks to send in a plan;
  that's the Router's job. We faithfully execute whatever plan we
  receive and report per-block status back.

## 3. Architecture

### 3.1 Component map

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Client (OpenAI HTTP)                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Dynamo Frontend  (HTTP :8000)  +  KV Router                        │
│     ─ NATS (control plane), etcd (discovery)                        │
│     ─ --router-mode kv     (prefix-aware worker selection)          │
│     ─ on `oandreeva-kv-p2p-v1-followups`: also emits                │
│       `extra_args["remote_kv_reuse_plan"]` on selected requests     │
└─────────────────────────────────────────────────────────────────────┘
                ┌───────────────┴───────────────┐
                ▼                               ▼
   ┌────────────────────────────┐    ┌────────────────────────────┐
   │   vLLM Worker 1            │    │   vLLM Worker 2            │
   │   (dynamo.vllm process)    │    │   (dynamo.vllm process)    │
   │ ┌────────────────────────┐ │    │ ┌────────────────────────┐ │
   │ │ OffloadingConnector    │ │    │ │ OffloadingConnector    │ │
   │ │  + RemoteG2Offloading- │ │    │ │  + RemoteG2Offloading- │ │
   │ │    Spec                │ │    │ │    Spec                │ │
   │ │  + RemoteG2Offloading- │ │    │ │  + RemoteG2Offloading- │ │
   │ │    Manager (target)    │ │    │ │    Manager (target)    │ │
   │ └─────┬──────────────────┘ │    │ └─────┬──────────────────┘ │
   │       │  prepare_load      │    │       │                    │
   │       │  (if plan in       │    │       │                    │
   │       │   extra_args)      │    │       │                    │
   │       ▼                    │    │       ▼                    │
   │  ┌──────────────────────┐  │    │  ┌──────────────────────┐  │
   │  │ TargetG2RpcClient    │──┼────┼─►│ SourceG2RpcServer    │  │
   │  │  (ZMQ REQ, pickle)   │  │    │  │  (ZMQ REP, pickle)   │  │
   │  └──────────┬───────────┘  │    │  └──────────┬───────────┘  │
   │             │              │    │             │              │
   │   RemoteG2LoadSpec         │    │  ┌──────────▼───────────┐  │
   │   (descriptors + lease)    │    │  │ SourceG2Descriptor-  │  │
   │             │              │    │  │   Registry           │  │
   │             ▼              │    │  │  ─ _records          │  │
   │  ┌──────────────────────┐  │    │  │  ─ _hash_to_key      │  │
   │  │ NIXL UCX agent (tgt) │◄─┼────┼──┤  ─ _policy ref       │  │
   │  │  initialize_xfer     │  │    │  │  ─ leases (TTL)      │  │
   │  │  READ                │  │    │  └──────────────────────┘  │
   │  └──────────┬───────────┘  │    │             │              │
   │             │              │    │             │              │
   │     GPU KV Cache           │    │  ┌──────────▼───────────┐  │
   │     (target slots)         │    │  │ CPU Pool (host-      │  │
   │                            │    │  │   pinned, 36 layers) │  │
   │                            │    │  │  ◄─ NIXL READ origin │  │
   └────────────────────────────┘    │  └──────────────────────┘  │
                                     └────────────────────────────┘
```

### 3.2 Data flow for a plan-driven load

1. Request arrives at Dynamo Frontend; Router picks **target worker**
   (KV-prefix-aware). When `remote_g2_plan.rs` is enabled, Router also
   attaches `RemoteKvReusePlan` to `extra_args["remote_kv_reuse_plan"]`.
2. dynamo.vllm `handlers.py` maps that into
   `sampling_params.extra_args["kv_transfer_params"]["remote_g2_plan"]`
   (vLLM's standard channel). Target's
   `RemoteG2OffloadingManager._peek_plan()` discovers it.
3. Target's manager calls `TargetG2RpcClient.resolve_and_lease(plan)`:
   1 ZMQ REQ → source's `SourceG2RpcServer`.
4. Source's `SourceG2DescriptorRegistry.resolve_and_lease(plan)`:
   - Validates `plan_version`, `source_worker_id`, `source_dp_rank`,
     `source_tier`, expiry.
   - Walks `block_hashes`, looks up each in `_records`.
   - For each match, pins `_policy[_hash_to_key[hash]].ref_cnt += 1`
     to suppress eviction while the lease is alive.
   - Stops at the first miss (returns the longest valid prefix).
   - Records lease with TTL (default 30 s).
   - Returns descriptors + lease_id over ZMQ REP.
5. Target's manager intersects the returned descriptors with the
   request's own per-block hashes. Only the intersection becomes a
   `RemoteG2LoadSpec`; misses fall through to local compute.
6. vLLM scheduler emits a load transfer for `(RemoteG2LoadSpec, GPU)`.
   Worker-side transfer handler calls `RawNixlRemoteG2Adapter.read_blocks`.
   It preserves source/destination block pairing, flattens every block's
   layers into aligned descriptor lists, and issues bounded multi-block
   NIXL `initialize_xfer(READ)` transactions (64 blocks by default).
7. After the transfer is known terminal (or on request cleanup), target sends
   `release_lease(id)`; source decrements `_policy[key].ref_cnt`, eviction is
   unblocked. The lease records an expiry timestamp for diagnostics/tests, but
   production does not auto-expire it while an unbounded READ may still live.

### 3.3 Singletons & process layout

```
EngineCore Python process (UniProcExecutor, TP=1)
├── KVConnectorRole.WORKER  ──┐
│   constructed FIRST during   │  both reach the SAME
│   _initialize_kv_caches      │  SourceG2DescriptorRegistry
│   (gpu_worker.py)            │  singleton via
│   ─ Worker-side manager,     │  get_or_create((src_wid, dp_rank))
│     policy stays empty       │
├── KVConnectorRole.SCHEDULER ─┘
│   constructed SECOND during
│   Scheduler.__init__
│   ─ Scheduler-side manager,
│     policy receives blocks
│     via complete_store
│
└── SourceG2DescriptorRegistry  (process-wide singleton, keyed by
                                 (source_worker_id, source_dp_rank))
    ├── _records       (block_hash → SourceG2DescriptorRecord)
    ├── _hash_to_key   (block_hash → OffloadKey)
    ├── _policy        (LRUCachePolicy; LAST-WINS, see §4.1)
    ├── _leases        (lease_id → (hashes, pin_refs, deadline))
    └── per-layer pool layout (ptrs, sizes, page size, row stride)
```

## 4. Key design decisions

### 4.1 `set_policy` is last-wins

The original implementation was first-wins on the assumption that the
SCHEDULER-role connector is constructed first. That assumption is
**wrong** for vLLM v1: `EngineCore.__init__` calls
`_initialize_kv_caches()` **before** `Scheduler(...)` is constructed.
`_initialize_kv_caches` ultimately invokes
`gpu_worker.ensure_kv_transfer_initialized` which builds the
WORKER-role connector first. The WORKER's manager never receives
`complete_store` calls, so its `_policy` stays empty forever.

Last-wins lets the SCHEDULER's populated policy overwrite the WORKER's
empty placeholder, and the source RPC's `_pin_block_locked` can find
live blocks during plan resolve.

This was the root cause of the `pin_failed (policy_missing_block,
policy_size=0)` failures during M5 bring-up. Captured in `data_model.py`
near `set_policy`, and in `manager.py`'s `__init__` comment.

### 4.2 Eviction protection via `ref_cnt` (pin-based lease)

A plan can be resolved at t=0, executed at t=Δ. In the meantime, the
scheduler-side cache manager is free to evict blocks under memory
pressure. To prevent the in-flight NIXL READ from reading stale or
recycled bytes, the registry holds an explicit pin: for each resolved
descriptor, `_policy[key].ref_cnt += 1`. `LRUCachePolicy.evict` skips
blocks with `ref_cnt > 0`.

Lease state is explicit:
- Successful target release (`release_lease(id, "ack")`) on request
  finish is the fast path.
- Automatic wall-clock reaping is intentionally disabled until the protocol
  has lease renewal plus a bounded NIXL abort guarantee; otherwise a reaper
  could unpin source memory during an active READ.
- Test surface: `tests/.../test_lease_pin.py` exercises 8-thread /
  500-iter concurrency, 1000-lease stress, and balance invariants.

### 4.3 Per-layer multi-pool registration

vLLM v1 allocates **one CPU tensor per transformer layer** (e.g. 36
for Qwen3-8B). Earlier prototypes registered only `tensors[0]` with
NIXL and copied that layer; the other 35 layers stayed uninitialised
on the target and the model produced garbage. The current path:

- `RemoteG2OffloadingSpec.init_kv_caches_layout` iterates
  `kv_caches.tensors`, collects per-layer base ptrs and sizes.
- `set_pool_layout` plumbs the full list down to the registry.
- `upsert_for_block` stores only the block_id + the layer-0 byte
  offset in the descriptor; the transfer handler reconstructs each
  layer's pointer at READ time from the bundle metadata.
- `RawNixlRemoteG2Adapter.read_blocks` uses block-major, layer-minor
  descriptor lists. One `initialize_xfer` covers up to the configured
  number of logical blocks across all layers; `read_block` is retained as
  a one-block compatibility wrapper.

### 4.4 Block-hash projection

The TRT-LLM Router's plan carries 64-bit XXH3-64 block hashes;
vLLM's block hashes are XXH3-128 (or SHA-256, configurable). To
round-trip, `block_hash_to_router_int` projects vLLM's full hash to a
64-bit int by taking `int.from_bytes(block_hash_bytes[:8], "big")`.
The Router and source agree on this projection — confirmed against
the TRT-LLM POC's wire bytes.

When `oandreeva-kv-p2p-v1-followups` lands, the Router will emit BOTH
the projected 64-bit hash (for matching) AND the original vLLM hash
(for cross-check); `RemoteKvReusePlan.kv_block_hashes` is the carrier
for the latter and already understood by `resolve_and_lease`.

### 4.5 Plan-driven path is additive, not replacement

`RemoteG2OffloadingManager` is a `CPUOffloadingManager` subclass.
Without a plan in `extra_args`, the inherited `lookup` /
`prepare_store` / `complete_load` logic runs unchanged — local
prefix-cache hits, eviction, etc. all work. The plan-driven branch
fires only when `_peek_plan()` finds a `RemoteKvReusePlan` and the
target has a configured `_target_client_factory`. In every other
case, the worker behaves like a vanilla `CPUOffloadingSpec`.

### 4.6 Failure isolation

- Source RPC errors → `TargetG2RpcClient` returns `None` / empty
  result → manager falls through to local compute; request still
  succeeds.
- NIXL READ or CUDA completion failure raises synchronously from
  `submit_load`, which is called before model forward. Failed jobs are not
  queued as deferred `TransferResult`s, so stale or partially-written KV
  cannot participate in one forward pass before the connector notices.
  This is fail-closed (the request/engine step fails rather than silently
  recomputing); recovery through `kv_load_failure_policy` remains separate
  upstream work. If a failed NIXL call cannot prove remote source access is
  terminal, its source lease is deliberately retained until explicit cleanup
  or process teardown. A wall-clock reaper is unsafe without lease renewal and
  a bounded NIXL abort: it could unpin a block while a long READ is still live.
- Source policy_missing_block / non_live / wrong_owner →
  `pin_failed`/`missing`/`wrong_owner` reasons returned in the
  per-block status; target drops the affected suffix.

### 4.7 Interaction with the GPU prefix cache (verification-time only)

`RemoteG2OffloadingManager.lookup` (the entry point that calls
`_peek_plan`) is invoked by the v1 scheduler only for token ranges
that are **not already covered by the GPU prefix cache**. Concretely,
the scheduler computes the GPU prefix-cache prefix length first,
then asks the offloading connector to look up the *suffix*. If the
prefix cache covers the entire request, the connector is never
called and `_peek_plan` therefore never sees the plan that was
injected into `extra_args`.

In production this is the correct behaviour — a GPU hit is strictly
faster than a cross-host NIXL pull, so taking it is the right
decision. But it does mean that any verification workload that
sends the *same* prompt twice into a worker with prefix caching on
will see `plan_seen_count = 0` on the second hit even though the
plan reached the request: the connector's `lookup` was simply not
called. The launcher script in `POC_OVERVIEW.md` §2.5 therefore sets
`--no-enable-prefix-caching` and `--gpu-memory-utilization 0.2`
paired with a 16 GiB `cpu_bytes_to_use`, so every request is forced
through the CPU/Remote tier and the connector observes the plan
deterministically. Production deployments should leave prefix
caching enabled; this knob is purely a verification artefact.

## 5. Wire protocol

ZMQ REP socket bound at `ipc://<source_rpc_socket_path>`. Each request
is `pickle.dumps({"method": str, "payload": dict})`; response is
`pickle.dumps({"ok": bool, "result"|"error": ...})`.

Methods (byte-compatible with TRT-LLM POC):

| method                   | purpose                                |
| ------------------------ | -------------------------------------- |
| `"resolve_and_lease"`    | Validate plan, pin prefix, return lease|
| `"release_lease"`        | Drop pins + lease (called on req end)  |
| `"get_metadata"`         | Exchange NIXL agent metadata           |
| `"stats"`                | Counters + sample hashes (debugging)   |

Plan shape (`vllm.v1.kv_offload.remote_g2.data_model.RemoteKvReusePlan`):

```python
{
  "plan_id": str,
  "request_id": str,
  "target_worker_id": int,
  "target_dp_rank": int,
  "source_worker_id": int,
  "source_dp_rank": int,
  "source_tier": "host_pinned",
  "block_hashes": tuple[int, ...],        # projected 64-bit hashes
  "kv_block_hashes": tuple[int, ...],     # original vLLM hashes (or empty)
  "start_block_index": int,
  "planned_prefix_blocks": int,
  "block_size_tokens": int,
  "created_at_ms": int,
  "expires_at_ms": int,
  "plan_version": int,                    # REMOTE_KV_REUSE_PLAN_VERSION
}
```

## 6. Compatibility with TRT-LLM POC

| Aspect                         | TRT-LLM POC               | vLLM port           | Compat? |
| ------------------------------ | ------------------------- | ------------------- | ------- |
| Source RPC wire format         | pickle-over-ZMQ           | pickle-over-ZMQ     | ✅ byte |
| Plan struct fields             | `RemoteKvReusePlan`       | `RemoteKvReusePlan` | ✅ byte |
| NIXL transport                 | UCX READ                  | UCX READ            | ✅      |
| Host-pinned tier name          | `"host_pinned"`           | `"host_pinned"`     | ✅      |
| `descriptor_generation`        | int, monotonic per hash   | same                | ✅      |
| Eviction protection            | TRT-side ref bump         | ref_cnt on policy   | ✅ same intent |
| Layer registration             | single buffer (TRT alloc) | one tensor / layer  | (impl) |
| `plan_version`                 | shared constant           | shared constant     | ✅      |

The two implementations have been **independently tested against the
same Rust Router shim**; no protocol divergence found.

## 7. Migration path: Router-emitted plans

Our Dynamo branch (`feat/kv-p2p-vllm-bridge`) is based on
`oandreeva-kv-p2p-v1-followups`, so the Rust Router code
(`lib/kv-router/src/remote_g2_plan.rs`) that emits
`extra_args["remote_kv_reuse_plan"]` is **already present** for
anyone installing Dynamo from source on our branch. In that
configuration the vLLM-side manager receives plans natively and the
shim below is not needed.

However, prebuilt Dynamo runtimes that are built from
`ai-dynamo/dynamo:main` (PyPI wheels, the prebuilt
`dynamo:vllm-kvbm-v2-fix1` container we used during M5 verification,
etc.) do **not** include `remote_g2_plan.rs` yet. To verify the
end-to-end path against those runtimes — and that was the
configuration M5 actually ran in — we use a Python-side
plan-injection shim (`kvp2p_plan_inject.py`) patched into
`dynamo/vllm/handlers.py`. See `POC_OVERVIEW.md` §2.7 for the shim's
wire shape and removal path.

When `oandreeva-kv-p2p-v1-followups` is merged to `main` (so the
prebuilt artifacts pick it up):
1. Router emits `extra_args["remote_kv_reuse_plan"]` natively in
   every artifact, not just source builds of our branch.
2. dynamo.vllm `handlers.py` already plumbs `extra_args` through to
   vLLM — only the key name needs to match what our manager looks
   for. **Pre-coordinated to be `"remote_kv_reuse_plan"` (top-level
   in `extra_args`) on the Router side, mapped to
   `kv_transfer_params["remote_g2_plan"]` on the vLLM side**, exactly
   what handlers.py already does for prefill/decode disagg.
3. The shim (`kvp2p_plan_inject.py` + the `_kvp2p_inject(...)` call
   in `handlers.py`) is no longer required for the production path
   and may be removed. We recommend keeping it in-tree under
   `components/src/dynamo/vllm/kv_p2p/` as a verification tool: it
   forces a known, deterministic plan into `extra_args` independent
   of the Router's reuse predicate, which is the only way to
   exercise the source-RPC + NIXL data plane in single-host or
   co-located topologies (where the predicate intentionally
   declines to emit). Gate it behind an env var so it does not run
   unless explicitly requested. No vLLM-side code changes required
   either way.

## 8. Open coordination items with HQ

| # | Item                                                  | Owner | Status |
| - | ----------------------------------------------------- | ----- | ------ |
| 1 | Router emits `remote_kv_reuse_plan` key on main       | HQ    | branch ready |
| 2 | Confirm wire compatibility against vLLM source RPC    | HQ    | independent re-test recommended |
| 3 | Dual-hash carriage (router-int + vLLM-bytes) in plan  | HQ    | field present in `RemoteKvReusePlan.kv_block_hashes`; needs HQ to populate |
| 4 | Disk tier (Remote-G3) scope-in?                       | both  | not in POC |

## 9. Test surface

Module: `tests/v1/kv_offload/remote_g2/`

| File                              | Coverage                              | # tests |
| --------------------------------- | ------------------------------------- | ------- |
| `test_e2e_smoke.py`               | Mock-NIXL end-to-end (4 cases)        | 4       |
| `test_lease_pin.py`               | Pin/unpin balance, TTL, concurrency   | 9       |
| `test_plan_miss.py`               | All plan-rejection reason codes       | 10      |
| `test_failure_recovery.py`        | NIXL fault injection → recovery       | 7       |
| `two_engines/orchestrate.py`      | Two real vLLM engines + real NIXL     | (eval)  |

All 30 unit tests pass after the `set_policy` last-wins fix.

## 10. References

- TRT-LLM KV-P2P POC source — internal NVIDIA branch
- Dynamo Router: `lib/kv-router/src/remote_g2_plan.rs` on
  `oandreeva-kv-p2p-v1-followups`
- vLLM v1 OffloadingConnector framework:
  `vllm/distributed/kv_transfer/kv_connector/v1/offloading/`
- NIXL UCX backend docs — NVIDIA NIXL repo
