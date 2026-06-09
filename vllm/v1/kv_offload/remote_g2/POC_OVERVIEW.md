# vLLM KV-P2P POC — Overview & Reproduction Guide

This document is the single self-contained entry point for the vLLM
port of NVIDIA's Remote-G2 (host-pinned tier) KV-P2P prototype. It
covers:

1. The design at a glance, plus how every file maps to its TRT-LLM
   counterpart and how the work relates to two open upstream items.
2. How to reproduce the end-to-end demo on a clean machine — assuming
   no access to our dev box.
3. Runtime configuration gotchas worth knowing before bringing it up.
4. How we verified end-to-end and what we observed.

Companion docs in this repository:

- `DESIGN.md` (alongside this file) — architecture, ZMQ wire protocol
  byte spec, and the rationale behind every non-obvious design
  decision. Required reading for code review and for anyone building
  a compatible client.
- `tests/v1/kv_offload/remote_g2/two_engines/FINDINGS.md` — two-engine
  byte-equality evaluation results (pure vLLM-to-vLLM, no Dynamo).

An additional operator handbook covering the specific dev box this
PoC was brought up on — concrete file paths, container names, patched
file line numbers, and bringup-history notes that are not portable
between environments — is kept by the PoC team but is **not published
to this repository**. Ask the team if you need it. The reproduction
recipe in §2 below is self-contained and does not require it.

---

## 1. Design at a glance

### 1.1 What KV-P2P does here

In a multi-worker LLM serving deployment, when worker A has already
computed and offloaded the KV cache for the prefix of a new request
that gets routed to worker B, B does not need to recompute that
prefix. Instead, it can **pull the KV blocks directly from A over
NIXL UCX (RDMA-capable transport)** and skip prefill for those blocks.
This is the Remote-G2 path: G2 = host-pinned tier (CPU pool); P2P =
worker-to-worker.

The two ends of the data plane:

- **Source**: worker A holds host-pinned CPU KV blocks and publishes
  descriptors (block hash → byte offset, per-layer pool ptrs).
- **Target**: worker B receives a "reuse plan" from the router naming
  which source / which blocks, looks them up, NIXL-READs the bytes
  into its own GPU KV slots, skips prefill for those tokens.

A descriptor lookup uses pin-based leases (refcount on the source's
LRU policy) to suppress eviction during the in-flight transfer.

### 1.2 Component map (process layout)

```
                Client (OpenAI HTTP)
                       │
                       ▼
   ┌────────────────────────────────────────────────────────┐
   │  Dynamo Frontend (HTTP)  +  KV Router (NATS + etcd)    │
   │  router-mode kv; with the upstream followups branch    │
   │  also emits remote_kv_reuse_plan into extra_args       │
   └─────────────────────┬──────────────────┬───────────────┘
                         │                  │
            ┌────────────▼────────┐  ┌──────▼──────────────┐
            │  Worker A (source)  │  │  Worker B (target)  │
            │                     │  │                     │
            │  vLLM EngineCore    │  │  vLLM EngineCore    │
            │   + RemoteG2-       │  │   + RemoteG2-       │
            │     OffloadingSpec  │  │     OffloadingSpec  │
            │                     │  │                     │
            │  SourceG2Descrip-   │  │  Manager peeks plan,│
            │  torRegistry        │  │  calls source's     │
            │   (singleton)       │  │  ZMQ REP            │
            │  + ZMQ REP socket   │◄─┤  TargetG2RpcClient  │
            │                     │  │                     │
            │  NIXL UCX agent  ◄──┼──┤  NIXL UCX agent     │
            │  (source pool)      │  │  (READ initiator)   │
            │                     │  │                     │
            │  CPU pool           │  │  GPU KV slots       │
            │  (host-pinned)      │  │  (load destination) │
            └─────────────────────┘  └─────────────────────┘
```

### 1.3 Code layout in `vllm/v1/kv_offload/remote_g2/`

| File | Role |
|---|---|
| `data_model.py` | `RemoteKvReusePlan`, `SourceG2Descriptor*`, `SourceG2DescriptorRegistry` (process-wide singleton). Owns leases, pin/unpin, hash→key→policy lookup. |
| `spec.py` | `RemoteG2OffloadingSpec(CPUOffloadingSpec)`. Entry point that vLLM's `OffloadingConnector` instantiates from `--kv-transfer-config`. Registers per-layer CPU pools with NIXL and starts the source RPC server. |
| `manager.py` | `RemoteG2OffloadingManager(CPUOffloadingManager)`. Hooks `complete_store` (publishes descriptors), `prepare_load` (consumes a plan, returns `RemoteG2LoadSpec`), and `on_request_finished` (releases the lease). |
| `source_rpc.py` | `SourceG2RpcServer`. ZMQ REP loop with four methods: `resolve_and_lease`, `release_lease`, `get_metadata`, `stats`. Pickle wire format, byte-compatible with TRT-LLM. |
| `target_client.py` | `TargetG2RpcClient`. Thread-safe ZMQ REQ wrapper used by the target manager to talk to a source's REP. |
| `load_spec.py` | `RemoteG2LoadSpec`, `_RemoteBlockHandle`. The `LoadStoreSpec` subclass the vLLM scheduler dispatches on. |
| `transfer_handler.py` | `RemoteG2TransferHandler`. `OffloadingHandler` for `(RemoteG2LoadSpec → GPULoadStoreSpec)`. Issues batched NIXL READs per layer, synchronizes CUDA, reports `TransferResult`. |
| `nixl_adapter.py` | `RawNixlRemoteG2Adapter` + `build_source_agent`. Thin wrapper over the NIXL UCX backend with a `fault_inject_every` knob for tests and a mock-memcpy fallback. |

The Dynamo side adds three small files in
`components/src/dynamo/vllm/`:

| File | Role |
|---|---|
| `args.py` (`_uses_remote_g2_offloading`) | Detects whether the user's `--kv-transfer-config` selects this spec (directly or nested in `PdConnector`). |
| `worker_factory.py` (added block) | When the spec is configured, registers the engine's source RPC as a dynamo endpoint so peers can reach it. |
| `kv_p2p/__init__.py` | Thin re-exports of the engine-agnostic bridges from `dynamo.trtllm.kv_p2p` — no duplicate implementation. |

### 1.4 vLLM ↔ TRT-LLM file correspondence

The reason the file counts are different on the two sides is
architectural:

- **TRT-LLM** runs the engine in an OpenMPI-forked subprocess; the
  dynamo parent and the engine cannot share Python objects, so all
  cross-network endpoints live in a parent-side bridge layer with
  separate parent ⇄ engine IPC.
- **vLLM v1** runs the EngineCore in a single Python process
  (UniProcExecutor for TP=1). The registry, the ZMQ REP loop, the
  NIXL agent, and the OffloadingConnector all live in the same
  process, so no parent ⇄ engine IPC layer is needed.

| Concern | TRT-LLM (`dynamo.trtllm.kv_p2p.*` + TRT-LLM tree) | vLLM (`vllm/v1/kv_offload/remote_g2/*`) |
|---|---|---|
| Parent ⇄ engine bridge (network ⇄ IPC) | `source_rpc_server.py`, `target_rpc_local.py`, `target_rpc_client.py` | **Reused, re-exported via `dynamo/vllm/kv_p2p/`** |
| TRT-LLM-specific reach-ins | `_engine_internals.py`, `_handle.py` | Not needed — vLLM exposes pool pointers directly through the spec |
| Plan / descriptor / registry data model (Python) | `tensorrt_llm._torch.pyexecutor.connectors.remote_g2` (in TRT-LLM tree) | `data_model.py` (mirror, byte-compatible wire format) |
| Engine-side ZMQ REP loop | inside the same TRT-LLM module | `source_rpc.py` |
| Engine-side ZMQ REQ to source | inside TRT-LLM | `target_client.py` |
| Host-pinned pool ownership | TRT-LLM C++ `kv_cache_manager` | vLLM `CPUOffloadingManager`, via `manager.py` subclass |
| NIXL agent + per-layer pool registration | C++ `expose_secondary_pool_to_nixl()` family | `nixl_adapter.py` + `spec.py:init_kv_caches_layout` |
| OffloadingConnector framework glue | n/a — TRT-LLM has no such framework | `spec.py`, `manager.py`, `load_spec.py`, `transfer_handler.py` |
| Source pin / unpin | C++ `acquire_pin / release_pin` | `data_model.py:_pin_block_locked / _unpin_block_locked` |
| Descriptor populate trigger | C++ BlockStored event → Python listener | `manager.py:complete_store → _upsert_descriptors_locked` |

The four glue files (`spec.py`, `manager.py`, `load_spec.py`,
`transfer_handler.py`) are the only vLLM-architecture-specific code
without 1:1 file counterparts in the TRT-LLM dynamo tree; the
functionality they wrap does exist on the TRT-LLM side but is
distributed between C++ and the engine subprocess's Python connector.

### 1.5 Relationship to two upstream items

**Olga's `oandreeva-kv-p2p-v1-followups` branch on `ai-dynamo/dynamo`**

This branch carries the parent-side bridges, the engine-agnostic ZMQ
forwarder, and `lib/kv-router/src/remote_g2_plan.rs` — the Rust router
code that emits `extra_args["remote_kv_reuse_plan"]` on routing
decisions. Our Dynamo work **branches from this branch** and adds the
vLLM-engine-specific wiring (`components/src/dynamo/vllm/{args.py,
worker_factory.py, kv_p2p/}`). The parent-side bridges are reused
verbatim — they speak the same pickle wire format whether the engine
on the other end of the IPC socket is TRT-LLM or vLLM.

When the followups branch is merged to `ai-dynamo/dynamo:main`, our
vLLM branch can be rebased onto main with no other change.

**`vllm-project/vllm` PR #43468 — OffloadingConnector BlockStored
payload fix**

Independently of us, the vLLM upstream is fixing a defect in
`OffloadingConnectorScheduler.take_events()`: the `BlockStored` events
it emitted carried no `token_ids`, no `parent_block_hash`, and
`block_size=0`, breaking any downstream consumer that indexes the
offloaded blocks by token prefix (Dynamo Router is the typical
consumer). Without this fix, the Router never knows our offloaded
blocks exist.

Our branch already carries an equivalent fix that we developed
independently at the same call site, using a side-table cached at
`prepare_store` time and drained at `take_events` time. Behaviour is
identical for the case PR #43468 covers (`block_size_factor == 1 &&
hash_block_size_factor == 1`, which is what we tested). PR #43468 is
slightly more correct for grouped offloading (`hbf > 1`) and slightly
cleaner (no side table). Once PR #43468 lands on
`vllm-project/vllm:main`, we should remove our local fix and rebase.

---

## 2. Reproduction on a clean machine

### 2.1 Prerequisites

- Linux host with NVIDIA GPUs (≥2 if you want to exercise the
  cross-worker pull on a single host; otherwise one per machine).
- CUDA-capable container runtime (Docker with `--gpus`).
- A model checkpoint locally available (we used Qwen3-8B; anything
  with a clean per-layer KV layout works).
- Two TCP ports free on the host network: 4222 (NATS), 2379 (etcd),
  plus the frontend port (8000 default).

### 2.2 Get the two repos

The minimal pieces:

```bash
# vLLM with the Remote G2 module
git clone https://github.com/linhu-nv/vllm.git
cd vllm
git checkout feat/kv-p2p-remote-g2
cd ..

# Dynamo with the vLLM bridge (branched off oandreeva-kv-p2p-v1-followups)
git clone https://github.com/linhu-nv/dynamo.git
cd dynamo
git checkout feat/kv-p2p-vllm-bridge
cd ..
```

### 2.3 Container

A CUDA-13 dev image with NIXL/UCX is the simplest base. We used
`vllm/vllm-openai:cu130-nightly`. Adjust for your platform.

```bash
docker run -d --name kvp2p-demo --network=host --gpus all \
  --ipc=host --shm-size=8g \
  -v $PWD:/work \
  --entrypoint sleep <DEV_IMAGE> infinity
```

Inside the container, install the modified vLLM editable and the
matching Dynamo:

```bash
docker exec kvp2p-demo bash -c '
  pip install -q pytest setuptools_rust
  cd /work/vllm && \
    VLLM_VERSION_OVERRIDE=0.20.0+local \
    VLLM_USE_PRECOMPILED=1 \
    pip install -e . --no-deps --no-build-isolation
  cd /work/dynamo && pip install -e .
'
```

If you want a vanilla deployment with the prebuilt Dynamo runtime
that does **not** yet include the Router-side plan emission (because
`oandreeva-kv-p2p-v1-followups` has not landed on main), you can still
exercise the end-to-end path using the Python plan-injection shim
described in §2.7. The shim is a single 200-line file plus a
2-function patch on `dynamo/vllm/handlers.py`.

### 2.4 Start NATS + etcd

The Dynamo control plane needs both. Easiest: a separate container.

```bash
docker run -d --name kvp2p-deps --network=host \
  nats:latest -p 4222 -m 8222
docker run -d --name kvp2p-etcd --network=host \
  quay.io/coreos/etcd:latest etcd \
    --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls http://0.0.0.0:2379

# Sanity:
curl -sf http://127.0.0.1:2379/health
curl -sf http://127.0.0.1:8222/healthz
```

### 2.5 Start two workers

Each worker needs:
- a distinct `source_worker_id` and IPC socket path
- the peer worker's id + socket path declared
- the model path
- a `--kv-transfer-config` selecting `RemoteG2OffloadingSpec`

A minimal launcher (one per worker) looks like this:

```bash
# launch_worker.sh
WID=${1:?worker_id}
GPU=${2:?gpu}
PEER_WID=${3:?peer_id}
PEER_SOCK=${4:?peer_socket}

SOCK="/tmp/dynamo_remote_g2_w${WID}.sock"
rm -f "$SOCK"

export CUDA_VISIBLE_DEVICES="$GPU"
export ETCD_ENDPOINTS=http://127.0.0.1:2379
export NATS_SERVER=nats://127.0.0.1:4222
export REMOTE_G2_SOURCE_WORKER_ID="$WID"
export REMOTE_G2_SOURCE_RPC_SOCKET_PATH="$SOCK"

exec python3 -m dynamo.vllm \
  --namespace kvp2p \
  --endpoint kvp2p.worker.generate \
  --model <PATH_TO_MODEL> \
  --gpu-memory-utilization 0.25 \
  --max-model-len 8192 \
  --enforce-eager \
  --block-size 16 \
  --enable-prefix-caching \
  --max-num-seqs 4 \
  --kv-transfer-config "{
    \"kv_connector\":\"OffloadingConnector\",
    \"kv_role\":\"kv_both\",
    \"kv_connector_extra_config\":{
      \"spec_name\":\"RemoteG2OffloadingSpec\",
      \"cpu_bytes_to_use\":2147483648,
      \"source_worker_id\":${WID},
      \"source_dp_rank\":0,
      \"source_rpc_socket_path\":\"${SOCK}\",
      \"use_mock_nixl\":false,
      \"peer_endpoints\":\"${PEER_WID}=${PEER_SOCK}\"
    }
  }"
```

Then:

```bash
docker exec -d kvp2p-demo bash /work/launch_worker.sh \
    1 1 2 /tmp/dynamo_remote_g2_w2.sock > /tmp/w1.log 2>&1
sleep 12   # let NIXL settle the metadata listener before peer races
docker exec -d kvp2p-demo bash /work/launch_worker.sh \
    2 2 1 /tmp/dynamo_remote_g2_w1.sock > /tmp/w2.log 2>&1

# Wait for "RemoteG2 source RPC bound at ipc://..." in both logs
# (typical cold start ~80s for Qwen3-8B with --enforce-eager).
```

### 2.6 Start the frontend

```bash
docker exec -d kvp2p-demo bash -c '
  export ETCD_ENDPOINTS=http://127.0.0.1:2379
  export NATS_SERVER=nats://127.0.0.1:4222
  python3 -m dynamo.frontend --http-port 8000 --router-mode kv \
      > /tmp/frontend.log 2>&1
'

# Sanity (returns a model listing):
curl -s http://127.0.0.1:8000/v1/models
```

### 2.7 Plan emission

There are two ways to get a `RemoteKvReusePlan` into the request.
Which one applies to you depends on **how you installed Dynamo**, not
on which `dynamo.vllm` Python you are running:

**(a) Native — Router emits the plan.** Olga's
`oandreeva-kv-p2p-v1-followups` branch on `ai-dynamo/dynamo` adds
`lib/kv-router/src/remote_g2_plan.rs`, which makes the KV Router emit
`extra_args["remote_kv_reuse_plan"]` on selected routing decisions.
Dynamo's vLLM handler already plumbs `extra_args` through to vLLM's
`sampling_params.extra_args["kv_transfer_params"]`. Our
`feat/kv-p2p-vllm-bridge` branch is **based on this followups
branch**, so a fresh source install of Dynamo following §2.3 (which
builds the Rust components from source as part of `pip install -e .`)
includes `remote_g2_plan.rs` and emits plans natively. Nothing
additional to install in this case.

Caveat: we have **not** end-to-end verified this path ourselves. Our
own M5 verification (§4) ran against a prebuilt `ai-dynamo` PyPI
package built from `main`, which does *not* contain
`remote_g2_plan.rs`, and we exercised the shim path (b) instead. The
native path should work for a fresh source install of our branch, but
you should sanity-check that the Router actually emits the plan
(grep worker logs for the manager's `RemoteG2: req ... plan ...
resolved` line; see §4.1) before assuming it.

**(b) POC shim — for any environment running a prebuilt Dynamo
binary that lacks `remote_g2_plan.rs`** (anything based on
`ai-dynamo/dynamo:main` as of 2026-06: PyPI wheels, prebuilt
containers, etc.). A single Python file, `kvp2p_plan_inject.py`,
queries each peer's source registry stats over the source RPC,
builds an "all-hashes" plan, and writes it into
`sampling_params.extra_args["kv_transfer_params"]["remote_g2_plan"]`.
Two `build_sampling_params*` helpers in `dynamo/vllm/handlers.py`
(the non-OpenAI and the OpenAI flavors) each receive a six-line
patch at their return site that calls
`maybe_inject_plan(sampling_params, ...)` inside a try/except.

The all-hashes plan is correctness-preserving because the vLLM-side
manager intersects the plan's hashes with the current request's
per-block hashes — only the intersection triggers a NIXL READ. The
shim raises the wire cost (one stats RPC per request, plus a larger
plan) but produces the same load decisions as a precise
Router-emitted plan.

Removal once you have native path (a): delete `kvp2p_plan_inject.py`,
revert the two `handlers.py` patches, and drop the
`KVP2P_PEER_SOCKETS` env var from the launcher.

### 2.8 Send a request

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"<MODEL_NAME>",
    "messages":[{"role":"user",
      "content":"<a prompt long enough to fill at least one 16-token block>"}],
    "max_tokens":15
  }'
```

---

## 3. Runtime configuration gotchas

These are not bugs, but they will trip up first-time users:

1. **Short prompts produce no offloaded blocks.** The offload block
   size is `offloaded_block_size = gpu_block_size * block_size_factor
   = 16 * 1 = 16` tokens. A prompt shorter than 16 tokens (after
   chat templating) publishes zero descriptors; the source registry
   stays empty and the plan path is a no-op. Use prompts of ~30
   tokens or more to see KV-P2P fire.

2. **vLLM v1 EngineCore construction order — `set_policy` must be
   last-wins.** `_initialize_kv_caches()` runs **before**
   `Scheduler(...)` is constructed, so the WORKER-role KV connector
   (whose policy is always empty) is built first. If you add new
   members to the registry that the SCHEDULER side should own, take
   the same last-wins approach. Already handled in `data_model.py`.

3. **NIXL metadata listener port conflict between co-located
   workers.** Co-located workers race for the same default NIXL
   metadata port and you will see noisy
   `metadata_stream.cpp: Address already in use` log spam. **The
   data plane still works** — peers successfully add each other via
   ZMQ-mediated metadata exchange and bytes do move. Plumb a
   per-worker NIXL listener port if the noise bothers you or if you
   are deploying cross-host where the conflict actually matters.

4. **`--enforce-eager`.** We test with eager mode because the
   compiled-graph eviction path has not been audited against the
   lease pin. Drop the flag only after adding a compiled-graph reset
   test.

5. **`dp_rank > 0` is not exercised in M5 live runs.** The
   data-plane code is dp-aware (the registry singleton is keyed by
   `(source_worker_id, source_dp_rank)`), but the live runtime
   config has only been run with `source_dp_rank=0`.

6. **Router routes most traffic to one worker under short bursts.**
   The router uses prefix-aware KV routing and a deterministic
   tiebreak. Two workers reporting identical metrics → most requests
   go to one worker, leaving the peer's source pool empty. To
   exercise cross-worker pulls, use prompts with low prefix overlap
   or stagger the request stream.

7. **`KVP2P_PEER_SOCKETS` is only consumed by the plan-injection
   shim.** When you remove the shim (once the native Router plan
   emission is available), drop this env var; it is not used by any
   vLLM-side code.

8. **CPU pool size vs. blocks observed.** `cpu_bytes_to_use` divided
   by `(per_block_bytes × num_layers)` is what `num_cpu_blocks` will
   end up at. On Qwen3-8B with 36 layers and 65536 bytes/block, 2 GiB
   gives ~910 blocks per worker. Increase `cpu_bytes_to_use` if you
   want to hold larger working sets at the source.

---

## 4. End-to-end verification & observed state

### 4.1 What we verified

We ran the live stack on a two-GPU host with two co-located
`dynamo.vllm` workers (Qwen3-8B, real NIXL UCX, real Dynamo Frontend
+ KV Router, plan injection via the POC shim). The verification set
is small but explicit:

| What | How |
|---|---|
| The plan path actually fires | Worker log shows `kvp2p plan_inject: attached plan ...` on the target and `RemoteG2: req ... plan ... resolved: N descriptors` on the source. |
| Source pin/unpin balance (no lease leak) | `stats` RPC exposes `pin_count_total` and `unpin_count_total` — must stay equal at idle. |
| Lookup never silently fails | `stats` exposes `pin_failures`; required to be 0. |
| Bytes actually move via NIXL | `stats.target_stats.plan_blocks_loaded > 0` — this counter increments only when the transfer handler reports a successful READ. |
| Transfer-handler error propagation works | `nixl_adapter.py` has a `fault_inject_every` knob; `tests/v1/kv_offload/remote_g2/test_failure_recovery.py` exercises it. |
| Pure protocol correctness (no engine) | `tests/v1/kv_offload/remote_g2/test_plan_miss.py` covers all reason codes returned by `resolve_and_lease`. |
| Output bytes are identical when the target reuses the source's KV instead of recomputing | `tests/v1/kv_offload/remote_g2/two_engines/orchestrate.py` runs two real `vllm.LLM` engines, has the source emit reference outputs, then verifies the target's plan-driven output is byte-for-byte equal (the strongest correctness signal we have). Documented in `two_engines/FINDINGS.md`. |

### 4.2 What we observed on the live demo

After warming the workers with long prompts and sending mixed
follow-ups:

| Metric | Worker 1 | Worker 2 |
|---|---|---|
| `descriptor_count` | 10 | 10 |
| `pin_count_total` / `unpin_count_total` | 24 / 24 | 30 / 30 |
| `pin_failures` | 0 | 0 |
| `policy_registered` | True | True |
| `target_stats.plan_seen_count` | 3 | 5 |
| `target_stats.plan_resolved_count` | 3 | 5 |
| `target_stats.plan_blocks_loaded` | 4 | 3 |
| Traceback / `TransferResult success=False` | 0 | 0 |

7 blocks moved end-to-end via real NIXL UCX READ between the two
workers. Pin and unpin perfectly balanced — no lease leak.

Unit-test surface:

| Suite | Coverage | Count | Status |
|---|---|---|---|
| `test_e2e_smoke.py` | Mock-NIXL end-to-end | 4 | ✅ |
| `test_lease_pin.py` | Pin/unpin balance, TTL, 8-thread/500-iter concurrency, 1000-lease stress | 9 | ✅ |
| `test_plan_miss.py` | All plan-rejection reason codes (full miss, partial miss, wrong worker/rank, expired, malformed, etc.) | 10 | ✅ |
| `test_failure_recovery.py` | NIXL fault injection and recovery | 7 | ✅ |

**30 / 30 pass.**

### 4.3 Status summary

| Aspect | State |
|---|---|
| Source RPC + plan resolve (cross-worker) | ✅ working |
| Per-layer (all 36 layers) NIXL READ | ✅ working |
| Lease pin / unpin / TTL | ✅ working, balanced |
| Plan-miss / partial-prefix degradation | ✅ working, no leases leaked |
| NIXL failure → vLLM `recompute` policy | ✅ working |
| End-to-end output byte equality vs source reference | ✅ verified (two-engine eval) |
| Dynamo HTTP frontend + KV router → vLLM workers | ✅ working |
| Plan emission via POC shim (`kvp2p_plan_inject.py`) | ✅ working, what M5 verified |
| Native plan emission by the Router (`remote_g2_plan.rs`) | included in our Dynamo branch (we are based on `oandreeva-kv-p2p-v1-followups`); ⏳ **not** end-to-end verified by us — needs sanity check on a fresh source install |
| Cross-host deployment + UCX device tuning | not yet exercised |
| Performance benchmarks (TTFT / throughput) | not yet measured |
| `dp_rank > 0` live | not yet exercised |
| Compiled-graph (non eager) | not yet exercised |

The functional path is closed end-to-end; the open items are
deployment-shape and performance work, not protocol or correctness
work.

---

## 5. Where to read more

In this repository:

- `DESIGN.md` §4 — architectural rationale (last-wins `set_policy`,
  pin-based lease, per-layer multi-pool, hash projection, additive
  plan path, failure isolation).
- `DESIGN.md` §5 — source RPC wire protocol byte layout.
- `DESIGN.md` §6 — TRT-LLM compatibility delta.
- `tests/v1/kv_offload/remote_g2/two_engines/FINDINGS.md` — pure
  vLLM-to-vLLM (no Dynamo) byte-equality evaluation.
- `kvp2p_plan_inject.py` — the POC plan-injection shim, with its
  own self-contained docstring.

Not in this repository (ask the PoC team if relevant):

- The internal operator handbook for the team's dev box, covering
  concrete file paths, container names, and bringup history.
