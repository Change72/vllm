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
| `transfer_handler.py` | `RemoteG2TransferHandler`: remote-read helper that `RemoteG2OffloadingWorker.submit_load` routes `RemoteG2LoadSpec → GPULoadStoreSpec` loads to. Issues batched NIXL READs per layer, synchronizes CUDA, reports `TransferResult`. |
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

A CUDA-13 dev image with NIXL/UCX is the simplest base. We
reproduced end-to-end on `vllm/vllm-openai:cu130-nightly`. Adjust for
your platform.

```bash
docker run -d --name kvp2p-demo --network=host --gpus all \
  --ipc=host --shm-size=8g \
  -v $PWD:/work \
  --entrypoint sleep vllm/vllm-openai:cu130-nightly infinity
```

**Note on `apt-get` inside this image.** Our environment had
`apt-get update` failing with persistent GPG `invalid signature`
errors (apt-key/gpgv chain misconfigured); `gpg --verify` of the same
InRelease against the same trusted keyring succeeded out-of-band, so
the issue was not the upstream signature itself. If you hit the same,
you will not be able to install build prerequisites via apt and will
need the binary-tarball fallbacks listed under 2.3.2(b) below.

#### 2.3.1 Install vLLM editable

Inside the container:

```bash
docker exec kvp2p-demo bash -c '
  pip install -q pytest setuptools_rust
  cd /work/vllm && \
    VLLM_VERSION_OVERRIDE=0.20.0+local \
    VLLM_USE_PRECOMPILED=1 \
    pip install -e . --no-deps --no-build-isolation
'
```

#### 2.3.2 Install Dynamo: choose one of two paths

**Path 2.3.2(a) — Prebuilt Dynamo from PyPI (fast, no Rust toolchain).**
This is what we used during M5 verification. It works for everything
**except** native Router plan emission; you will need the shim
described in §2.7 to inject plans on the Python side.

```bash
docker exec kvp2p-demo pip install ai-dynamo
```

**Path 2.3.2(b) — Source build of our Dynamo branch (slower, needs
several system packages; required to get a `dynamo._core` that
contains the Rust `remote_g2_plan.rs` code).** This rebuilds the Rust
crates that include `lib/kv-router/src/remote_g2_plan.rs`. **Note**
that compiling these crates is necessary but *not sufficient* for
native plan emission end-to-end — see §2.7 (a) for the additional
runtime configuration required, and §4.3 for the gap we observed
during the live reproduction.

**Build prerequisites the workspace requires.** All of these are
needed; without any one of them the cargo build fails at the named
crate.

| Build prerequisite | Failing crate | How to install (no-apt fallback we used) |
| ------------------ | ------------- | ---------------------------------------- |
| Rust toolchain **1.93.1** (the workspace's `rust-toolchain.toml` pins this exact version) | first cargo invocation | `curl https://sh.rustup.rs -sSf \| sh -s -- -y --default-toolchain stable --profile minimal --no-modify-path && /root/.cargo/bin/rustup install 1.93.1 --profile minimal --no-self-update` |
| `protoc` ≥ 25 | `etcd-client` (`.proto` compile) | Pull the Linux release zip from `https://github.com/protocolbuffers/protobuf/releases/download/v25.1/protoc-25.1-linux-x86_64.zip`, extract (`python3 -c 'import zipfile; zipfile.ZipFile("/tmp/p.zip").extractall("/opt/protoc")'`), chmod +x, put on PATH, export `PROTOC=/opt/protoc/bin/protoc` |
| `libclang` **with C system headers** (the pip `libclang` package alone ships only the .so and fails with `'stdbool.h' file not found`) | `nixl-sys` (bindgen wrapping C header) | Download a full LLVM release: `curl -L https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.6/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04.tar.xz -o /tmp/clang.tar.xz`, extract to `/opt/llvm`, then export `LIBCLANG_PATH=/opt/llvm/lib` and `BINDGEN_EXTRA_CLANG_ARGS=-I/opt/llvm/lib/clang/17/include`. The tarball is ~1 GB compressed, ~5 GB extracted — extract to a volume with space (e.g. the bind-mounted host disk, not the container overlay). |
| Pre-fetched Swagger-UI zip | `utoipa-swagger-ui` (build.rs fetches the v5.17.14 release) | If the build host's network is slow / restricted, build.rs fails with `folder ... does not exist` after a partial download. Pre-fetch: `curl -L https://github.com/swagger-api/swagger-ui/archive/refs/tags/v5.17.14.zip -o /opt/swagger.zip` and export `SWAGGER_UI_DOWNLOAD_URL=file:///opt/swagger.zip` |

Then build and install (we used `maturin build` then explicit
`pip install` on the wheel since `maturin develop` requires a venv
which the base image lacks):

```bash
docker exec kvp2p-demo bash -c '
  set -e
  export PATH=/root/.cargo/bin:$PATH
  export PROTOC=/opt/protoc/bin/protoc
  export LIBCLANG_PATH=/opt/llvm/lib
  export BINDGEN_EXTRA_CLANG_ARGS="-I/opt/llvm/lib/clang/17/include"
  export SWAGGER_UI_DOWNLOAD_URL=file:///opt/swagger.zip
  pip install -q maturin
  cd /work/dynamo/lib/bindings/python
  maturin build --release
  pip install --force-reinstall --no-deps \
    /work/dynamo/lib/bindings/python/target/wheels/ai_dynamo_runtime-*.whl
  cd /work/dynamo && pip install --no-deps -e .
'
```

The dynamo workspace has ~800 transitive crates. On our box (with a
warm cargo registry from a prior failed attempt) the actual cargo
build was ~2.5 minutes; from a completely cold registry expect 20–40
minutes of dependency download + compile. Confirm the wheel contains
`remote_g2_plan.rs` strings with `strings
/usr/local/lib/python3.12/dist-packages/dynamo/_core.abi3.so | grep
remote_g2_plan` — you should see the file path baked into the
binary.

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
- a `--kv-events-config` so that BlockStored events flow to the Router
- `KVP2P_PEER_SOCKETS` env var if you intend to use the §2.7 shim

A launcher (one per worker):

```bash
# launch_worker.sh
WID=${1:?worker_id}
GPU=${2:?gpu}
PEER_WID=${3:?peer_id}
PEER_SOCK=${4:?peer_socket}

SOCK="/tmp/dynamo_remote_g2_w${WID}.sock"
rm -f "$SOCK"

# CRITICAL: must use "*" (bind), not "127.0.0.1" (connect).
# vLLM's ZmqEventPublisher only bind()s when the endpoint string contains
# "*", "::", "ipc://", or "inproc://"; otherwise it connect()s. If both
# vLLM (publisher) and Dynamo (subscriber) connect, no one binds and
# all KV events are silently dropped.
EVENT_ENDPOINT="tcp://*:$((5560 + WID))"

export CUDA_VISIBLE_DEVICES="$GPU"
export ETCD_ENDPOINTS=http://127.0.0.1:2379
export NATS_SERVER=nats://127.0.0.1:4222
export REMOTE_G2_SOURCE_WORKER_ID="$WID"
export REMOTE_G2_SOURCE_RPC_SOCKET_PATH="$SOCK"
# Consumed by the §2.7 shim (kvp2p_plan_inject.py); harmless if you do
# not patch handlers.py with the shim.
export KVP2P_PEER_SOCKETS="${PEER_WID}=${PEER_SOCK}"

exec python3 -m dynamo.vllm \
  --namespace kvp2p \
  --endpoint kvp2p.worker.generate \
  --model <PATH_TO_MODEL> \
  --gpu-memory-utilization 0.2 \
  --max-model-len 8192 \
  --enforce-eager \
  --block-size 16 \
  --no-enable-prefix-caching \
  --max-num-seqs 4 \
  --kv-events-config "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"${EVENT_ENDPOINT}\"}" \
  --kv-transfer-config "{
    \"kv_connector\":\"OffloadingConnector\",
    \"kv_role\":\"kv_both\",
    \"kv_connector_extra_config\":{
      \"spec_name\":\"RemoteG2OffloadingSpec\",
      \"cpu_bytes_to_use\":17179869184,
      \"source_worker_id\":${WID},
      \"source_dp_rank\":0,
      \"source_rpc_socket_path\":\"${SOCK}\",
      \"use_mock_nixl\":false,
      \"peer_endpoints\":\"${PEER_WID}=${PEER_SOCK}\"
    }
  }"
```

Why each of the non-obvious flags matters — every one of these was a
debug session by itself:

- **`--gpu-memory-utilization 0.2` + `cpu_bytes_to_use: 17179869184`
  (16 GiB)**: the CPU pool must be *larger* than the GPU KV cache.
  vLLM's default would otherwise keep every block in GPU permanently
  and the offloading-to-CPU step that the whole PoC depends on never
  triggers. Sizing the GPU pool below the CPU pool forces eviction of
  GPU blocks into CPU.
- **`--no-enable-prefix-caching`**: counter-intuitive but mandatory
  for verification. The v1 scheduler computes the GPU prefix-cache
  prefix length first, then asks the OffloadingConnector to look up
  only the *suffix* that the prefix cache didn't cover. So a request
  whose prefix is fully covered by the GPU prefix cache never reaches
  `RemoteG2OffloadingManager.lookup()`, `_peek_plan()` is never
  consulted, and `plan_blocks_loaded` stays 0 no matter what plan the
  Router or the shim injected. This is the correct production
  behaviour (a GPU hit is strictly faster than a remote pull), but in
  a synthetic two-shot verification — same prompt twice — the second
  shot is a full GPU hit and Remote-G2 is invisible. Disabling prefix
  caching forces every request through the connector so the data
  plane can be observed deterministically.

  Composition with prefix caching in production is the partial-hit
  case: the prefix cache serves `[0, L_gpu)` from GPU; the
  OffloadingConnector serves `[L_gpu, L_total)` from the plan via
  NIXL READ. Both run for their respective ranges — there is no
  short-circuit, and KV-P2P fires exactly on the prefix range that
  the GPU cache missed, which is the regime that benefits from it.
- **`--kv-events-config '{...,"publisher":"zmq","endpoint":"tcp://*:PORT"}'`**:
  required for the OffloadingConnector to publish its BlockStored
  events. Dynamo's `create_kv_events_config` returns `None` (and
  silently sets `use_kv_events=False`) unless this config is
  explicitly provided. The `*` in the endpoint is what controls
  bind-vs-connect on vLLM's side as noted above.
- **`--enforce-eager`**: the compiled-graph path has not been audited
  against the lease-pin lifecycle. Drop only after adding a
  compiled-graph reset test.
- **`--max-num-seqs 4`**: keeps scheduling predictable during PoC
  bring-up.

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
  # Required for the Rust KV router to actually call
  # select_remote_g2_reuse_plan(). Without it the Router emits
  # RemoteKvReuseDecision::NoPlan { reason: Disabled } for every
  # request and remote_kv_reuse_plan never reaches extra_args.
  # Only meaningful for path 2.7(a); harmless when only the shim is in
  # use.
  export DYN_REMOTE_G2_REUSE_ENABLED=1
  python3 -m dynamo.frontend --http-port 8000 --router-mode kv \
      > /tmp/frontend.log 2>&1
'

# Sanity (returns a model listing):
curl -s http://127.0.0.1:8000/v1/models
```

### 2.7 Plan emission paths

A plan reaches vLLM as `sampling_params.extra_args["kv_transfer_params"]["remote_g2_plan"]`.
Two components can fill it in — the Dynamo Rust Router itself, or a
Python shim we patch into `dynamo/vllm/handlers.py`. They use the
*same downstream pipeline* (manager → source RPC → NIXL READ); they
only differ in **who decides which blocks to plan for**.

#### 2.7(a) Native — Router emits the plan (verified)

Olga's `oandreeva-kv-p2p-v1-followups` branch on `ai-dynamo/dynamo`
adds `lib/kv-router/src/remote_g2_plan.rs`, which gives the KV Router
the function `select_remote_g2_reuse_plan` and the wire shape
`RemoteKvReusePlan`. Our `feat/kv-p2p-vllm-bridge` branch is based on
this followups branch and adds the additional fixes needed to drive
the path end-to-end through the vLLM bridge.

End-to-end verified on a co-located 2-worker single-host
reproducer (see §4.1 for the verification numbers): Phase 2's
request reaches W2 with the Router-attached plan, W2 resolves it
against W1's source registry, W1 pins all 19 blocks, NIXL UCX pulls
the bytes, and W2 serves the prompt without writing it to its own
CPU pool. W1 pin/unpin balances 19/19 with zero failures.

The single-host predicate non-trigger case. The Router's
`select_remote_g2_reuse_plan` only emits when **target ≠ source AND
target has no device-tier prefix AND some other worker has a
CPU_PINNED prefix** for the same blocks. In a typical KV-router
deployment with prefix-aware routing, the Router naturally sends any
prefix-bearing request to whichever worker already has the prefix on
device — so `target == source` and the predicate short-circuits. To
observe the native path firing in a co-located test, force a
cross-worker route via Dynamo's `x-worker-instance-id` header (the
full reproducer is in §2.9 below).

Prerequisites required that the followups branch does not turn on by
default; the §2.5 launcher and §2.6 frontend already set these:

1. **`DYN_REMOTE_G2_REUSE_ENABLED=1`** in the environment of
   `dynamo.frontend` — gates `select_remote_g2_reuse_plan`.
2. **vLLM `--kv-events-config` with `enable_kv_cache_events=True`** on
   every worker — required for Dynamo's `KvEventPublisher` to forward
   BlockStored events to the Router. The endpoint string must start
   with `tcp://*:` (or `ipc://...` / `inproc://...`) so that vLLM
   `bind`s instead of `connect`s; otherwise both sides connect and no
   events flow.
3. **vLLM-side event medium override** —
   `RemoteG2OffloadingManager.medium = "CPU_PINNED"` (already
   committed). The Rust Router's `StorageTier::from_kv_medium` only
   maps `"CPU_PINNED"` / `"CPU_TIER1"` to `HostPinned`; the inherited
   `CPUOffloadingManager` default of `"CPU"` is silently classified
   as the default Device tier.
4. **Prefix caching enabled** on the worker (so KV events flow at
   all). Dynamo's `create_kv_events_config` short-circuits to
   "publish nothing" when `enable_prefix_caching=False`, which is
   the correct production setting anyway. The shim verification in
   §2.7(b) disables prefix caching to keep its two-shot test
   deterministic; the native path does not need that knob off because
   the cross-routing forced via `x-worker-instance-id` is what
   guarantees the target sees the prompt fresh.

The five fixes the native path needed end-to-end:

- **Dynamo `lower_tier_indexers.rs` — cross-worker host_pinned
  fallback.** The indexer's per-tier query extends the device chain
  per worker; a source whose own device tier still covers the
  prefix reports zero host-pinned hits even when it physically has
  the blocks. Added a parallel from-root(0) query per tier and
  merged by max-hits-per-worker. (Patch: `fix(kv-router):
  cross-worker host_pinned fallback in lower-tier query`.)
- **Dynamo `remote_g2_plan.rs` — target-anchored plan window.** The
  plan-emission predicate computed `start = source_device_match`,
  which collapses to `request_blocks` when the source still holds
  the prefix on device. Switched to `start = target_device_match`,
  with a `source_hp_start` term for chains that begin past zero.
- **Dynamo `handlers.py` — plan adapter.** The Router attaches the
  plan to `request.extra_args["remote_kv_reuse_plan"]`; vLLM expects
  `sampling_params.extra_args["kv_transfer_params"]["remote_g2_plan"]`.
  Added a translator in `build_sampling_params`. The Router's
  source/target identity is Dynamo's WorkerId (~64-bit lease/instance
  id); the vLLM source RPC and `peer_endpoints` map use the small
  integer `REMOTE_G2_SOURCE_WORKER_ID`. Identity-translate at the
  adapter (2-worker hardcoded for verification; the principled fix
  is a shared discovery layer per §2 of the cross-framework KV-P2P
  proposal).
- **vLLM `manager.py` — hash projection alignment.** The Router
  observes block hashes as `int.from_bytes(bytes, "big") & ((1<<64)-1)`
  (vLLM's `maybe_convert_block_hash`), which keeps the **last 8
  bytes** of multi-byte hashes. The previous
  `block_hash_to_router_int` took the **first 8 bytes**, so source
  registry keys and Router-side `kv_block_hashes` were two different
  projections of the same data — every native-plan resolve returned
  `missing`. Aligned the projection.
- **vLLM `data_model.py` — descriptor block_hash uses kv_hash.** The
  source's `resolve_and_lease` set the descriptor's `block_hash` to
  the Router's `identity_hash` (XXH3_64 of tokens), which the target
  side could not match against its own
  `block_hash_to_router_int(target_key)` lookup. Switched to
  `kv_hash` (the indexer's chain-walked vLLM projection); the shim
  path falls back to `identity_hash` automatically when
  `planned_kv_block_hashes` is empty, so it remains a no-op there.

The first three fixes live on the Dynamo `feat/kv-p2p-vllm-bridge`
branch; the last two live on the vLLM `feat/kv-p2p-remote-g2`
branch. Both branches are publicly visible on `linhu-nv`.

#### 2.7(b) Shim — Python-side unconditional plan injection (what we used to verify e2e)

A single Python file, `kvp2p_plan_inject.py`, queries each peer's
source registry stats over the source RPC and writes an "all-hashes"
plan into `sampling_params.extra_args["kv_transfer_params"]["remote_g2_plan"]`.
This unconditionally injects a plan on every request, bypassing the
Router-side predicate and producing the same load decisions as a
precise Router-emitted plan (the vLLM-side manager intersects the
plan's hashes with the current request's per-block hashes — only the
intersection triggers a NIXL READ).

The shim's audience: any environment that wants to verify the data
plane works without engineering a cross-worker routing scenario. It is
also the only realistic option when you are running against a
prebuilt Dynamo binary that lacks `remote_g2_plan.rs` (anything based
on `ai-dynamo/dynamo:main` as of 2026-06: PyPI wheels, prebuilt
containers).

**Patch location**: there is exactly one place to patch —
`build_sampling_params` in `dynamo/vllm/handlers.py` (the
non-OpenAI variant, around line 319). The earlier guidance pointed at
the `*_openai` variant too, but `dynamo.vllm` chat completions go
through the non-OpenAI builder; patching the OpenAI variant has no
effect for chat completions. Insert this block right before the
function's final `return sampling_params`:

```python
# === KV-P2P plan injection (router-shim, non-openai path) ===
if os.environ.get("KVP2P_PEER_SOCKETS"):
    try:
        from dynamo.vllm.kv_p2p.plan_inject_shim import maybe_inject_plan
        maybe_inject_plan(
            sampling_params,
            request_id=str(request.get("request_id", "unknown")),
        )
    except Exception as _exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "kvp2p plan inject failed: %s", _exc
        )
```

The shim file is published with our Dynamo branch at
`components/src/dynamo/vllm/kv_p2p/plan_inject_shim.py`. Its
docstring explains the wire protocol and the env-var contract
(`KVP2P_PEER_SOCKETS`, `REMOTE_G2_SOURCE_WORKER_ID`). The shim is a
no-op unless `KVP2P_PEER_SOCKETS` is set in the worker's
environment — production paths that prefer Router-emitted plans
simply leave that env var unset.

Removal once you no longer want the shim: delete the patch block
from `handlers.py`. The shim file itself can stay around — it does
nothing when its env var is unset.

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

### 2.9 End-to-end execution guide: native path with `x-worker-instance-id`

This is the verified end-to-end reproducer for the **native Router
plan path** in a co-located 2-worker setup. It assumes §2.1–§2.6 are
already running: source-built Dynamo on `feat/kv-p2p-vllm-bridge`,
vLLM on `feat/kv-p2p-remote-g2`, two workers up (W1 on GPU 1, W2 on
GPU 2), Dynamo Frontend on `:8001` with `DYN_REMOTE_G2_REUSE_ENABLED=1`.
For the native path use the launcher in this section (`launch_worker_xworker.sh`),
not the §2.5 shim-verification launcher — the differences are
prefix-caching is ON, `KVP2P_PEER_SOCKETS` is unset, and otherwise
the flag set is identical.

**Step 1 — verify the launcher does NOT export `KVP2P_PEER_SOCKETS`.**
The plan adapter in `dynamo.vllm.handlers.build_sampling_params`
handles the Router-emitted plan unconditionally; the opt-in shim
hook is gated on `KVP2P_PEER_SOCKETS` being set, so unset it for
the native run. Use this launcher (saved as
`/work/launch_worker_xworker.sh` in our test container; the only
delta vs. §2.5 is `--enable-prefix-caching` and the absence of the
shim env var):

```bash
#!/bin/bash
set -e
WORKER_ID="${1:?worker_id required}"
GPU="${2:?gpu required}"
ENDPOINT="${3:-generate}"
NAMESPACE="${4:-kvp2p}"

export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES="$GPU"
export ETCD_ENDPOINTS="http://127.0.0.1:2379"
export NATS_SERVER="nats://127.0.0.1:4222"
export REMOTE_G2_SOURCE_WORKER_ID="$WORKER_ID"

SOCKET="/tmp/dynamo_remote_g2_w${WORKER_ID}.sock"
rm -f "$SOCKET"
export REMOTE_G2_SOURCE_RPC_SOCKET_PATH="$SOCKET"

if [ "$WORKER_ID" = "1" ]; then PEER_WID=2; else PEER_WID=1; fi
PEER_SOCK="/tmp/dynamo_remote_g2_w${PEER_WID}.sock"

# Native path: do NOT export KVP2P_PEER_SOCKETS — the shim must stay
# dormant so we measure the native Router plan path exclusively.
unset KVP2P_PEER_SOCKETS

EVENT_ENDPOINT="tcp://*:$((5560 + WORKER_ID))"

exec python3 -m dynamo.vllm \
  --namespace "$NAMESPACE" \
  --endpoint "$NAMESPACE.worker.$ENDPOINT" \
  --model /raid/fly/model/Qwen3-8B \
  --gpu-memory-utilization 0.2 \
  --max-model-len 8192 \
  --enforce-eager \
  --block-size 16 \
  --enable-prefix-caching \
  --max-num-seqs 4 \
  --kv-events-config "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"${EVENT_ENDPOINT}\"}" \
  --kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"RemoteG2OffloadingSpec\",\"cpu_bytes_to_use\":17179869184,\"source_worker_id\":${WORKER_ID},\"source_dp_rank\":0,\"source_rpc_socket_path\":\"${SOCKET}\",\"use_mock_nixl\":false,\"peer_endpoints\":\"${PEER_WID}=${PEER_SOCK}\"}}"
```

**Step 2 — discover the Dynamo `worker_instance_id` for each
worker.** Dynamo's `x-worker-instance-id` header takes the 64-bit
lease/instance id assigned by Dynamo at registration time, which is
distinct from the small integer `REMOTE_G2_SOURCE_WORKER_ID` you set
in the launcher. Read it from etcd, then probe-confirm which is W1
vs W2 (instance IDs are assigned in startup order, but the safest
check is empirical):

```bash
# (a) list instance_id -> tcp port pairs from etcd
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 - <<'PY'
import etcd3, json, re
c = etcd3.client(host='127.0.0.1', port=2379)
for v, m in c.get_all():
    k = m.key.decode()
    if not k.startswith("v1/instances/kvp2p/worker/generate/"):
        continue
    meta = json.loads(v.decode())
    port = re.match(r"[\d\.]+:(\d+)/", meta["transport"]["tcp"]).group(1)
    print(meta["instance_id"], "->", port)
PY

# (b) for each instance_id from (a), send a 1-token probe and watch
#     which of /work/native_logs/w1.log /work/native_logs/w2.log grows
#     more. The one that grows more is OUR REMOTE_G2_SOURCE_WORKER_ID
#     for that instance_id.
probe() {
  W1_BEFORE=$(wc -l < /work/native_logs/w1.log)
  W2_BEFORE=$(wc -l < /work/native_logs/w2.log)
  curl -s -X POST http://127.0.0.1:8001/v1/completions \
    -H 'Content-Type: application/json' \
    -H "x-worker-instance-id: $1" \
    -d '{"model":"/raid/fly/model/Qwen3-8B","prompt":"hi","max_tokens":2,"temperature":0}' >/dev/null
  sleep 4
  W1_INC=$(($(wc -l < /work/native_logs/w1.log) - W1_BEFORE))
  W2_INC=$(($(wc -l < /work/native_logs/w2.log) - W2_BEFORE))
  [ "$W1_INC" -gt "$W2_INC" ] && echo "$1 -> W1" || echo "$1 -> W2"
}
probe <instance_id_from_(a)_first>
probe <instance_id_from_(a)_second>
```

Record the mapping; call the W1 id `$W1_INSTANCE` and the W2 id
`$W2_INSTANCE` for the rest of this guide.

**Step 3 — Phase 1: warm W1's CPU-pinned tier with the test prompt.**
Send a long prompt to W1 with `x-worker-instance-id: $W1_INSTANCE`
so the Frontend routes it directly. W1 will run prefill, offload
blocks to its CPU-pinned pool, and publish `BlockStored` events on
both Device (GPU prefix cache) and HostPinned tiers:

```bash
LONG_PROMPT='You are a careful, friendly senior software engineer reviewing a complex distributed-systems pull request. Below is a long context describing the system architecture: there are two vLLM workers running on the same host, each holding a host-pinned CPU KV cache, connected via NIXL UCX for KV transfer. The router is responsible for deciding which worker should serve each request and whether to inject a Remote-G2 reuse plan that tells the target worker to pull KV blocks from another worker'\''s CPU-pinned tier instead of recomputing prefill. Your job is to summarise this architecture and call out any subtle correctness concerns in the lease lifecycle, the pin/unpin balance, the eviction policy interaction, and the NIXL per-layer transfer shape. Please proceed with the review in three short paragraphs.'

curl -s -X POST http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -H "x-worker-instance-id: $W1_INSTANCE" \
  -d "{\"model\":\"/raid/fly/model/Qwen3-8B\",\"prompt\":\"$LONG_PROMPT $LONG_PROMPT\",\"max_tokens\":10,\"temperature\":0}"

sleep 8   # let BlockStored events settle into the Frontend indexer
```

Confirm W1's source registry got populated:

```bash
python3 - <<'PY'
from vllm.v1.kv_offload.remote_g2.target_client import TargetG2RpcClient
c = TargetG2RpcClient('/tmp/dynamo_remote_g2_w1.sock', timeout_ms=3000)
print(c.stats(sample_limit=0))
c.close()
PY
# Expect: descriptor_count: ~19, target_stats.plan_seen_count: 0
```

**Step 4 — Phase 2: send the same prompt to W2 with
`x-worker-instance-id: $W2_INSTANCE`.** The Frontend's KV Router
honours the header and bypasses prefix-aware routing, forcing the
request to W2. W2 has no device-tier prefix for this prompt
(W2 has never seen it), and W1 has it on CPU_PINNED — so the
predicate emits a plan, the adapter copies it into
`sampling_params.extra_args["kv_transfer_params"]["remote_g2_plan"]`,
W2's `RemoteG2OffloadingManager.lookup` peeks the plan, calls
`resolve_and_lease` over ZMQ to W1's source registry, W1 pins all
19 blocks, descriptors come back, and the per-layer NIXL UCX READ
moves the bytes to W2's GPU KV slots.

```bash
curl -s -X POST http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -H "x-worker-instance-id: $W2_INSTANCE" \
  -d "{\"model\":\"/raid/fly/model/Qwen3-8B\",\"prompt\":\"$LONG_PROMPT $LONG_PROMPT\",\"max_tokens\":10,\"temperature\":0}"

sleep 4
```

**Step 5 — confirm the native path fired end-to-end.** Pull stats
from both workers and check the canonical metrics:

```bash
python3 - <<'PY'
from vllm.v1.kv_offload.remote_g2.target_client import TargetG2RpcClient
for sock in ('/tmp/dynamo_remote_g2_w1.sock',
             '/tmp/dynamo_remote_g2_w2.sock'):
    c = TargetG2RpcClient(sock, timeout_ms=3000)
    print(sock, c.stats(sample_limit=0))
    c.close()
PY
```

Expected, on a successful native-path run:

| Metric | W1 (source) | W2 (target) |
|---|---|---|
| `descriptor_count` | 19 | 0 |
| `pin_count_total` / `unpin_count_total` | 19 / 19 | 0 / 0 |
| `pin_failures` | 0 | 0 |
| `target_stats.plan_seen_count` | 0 | 1 |
| `target_stats.plan_resolved_count` | 0 | 1 |
| `target_stats.plan_load_specs_emitted` | 0 | 1 |
| `target_stats.plan_blocks_loaded` | 0 | 19 |

The W2 `descriptor_count = 0` is the diagnostic to focus on: W2
served the entire prompt by pulling KV from W1 over NIXL and
**never wrote the prompt's blocks into its own CPU pool**. The plan
path completely short-circuited the prefill computation it would
otherwise have done.

If you also want to see the Router's plan emission in the Frontend
log:

```bash
grep -E "PROBE remote_g2_plan|host-pinned tier present" /work/native_logs/frontend.log | tail -5
```

The `host-pinned tier present` line should report
`host_pinned_hits={WorkerWithDpRank { worker_id: <W1_INSTANCE>, ... }: 19}`
and the `PROBE remote_g2_plan kv_block_hash chain` line should show
19 entries in both `requested_block_hashes` and `chain_kv_block_hashes`.

**Troubleshooting**:

- `plan_seen_count = 0` on W2 after Phase 2: either the Router did
  not emit (check the Frontend log for `host_pinned_hits` and the
  `PROBE remote_g2_plan` line), or the plan adapter in
  `handlers.py` did not run (check the worker log for the
  `native router plan adapted: ...` info line).
- `plan_resolved_count = 0` while `plan_seen_count = 1`: identity
  translation mismatch — check that the launcher sets
  `REMOTE_G2_SOURCE_WORKER_ID` and that the W1 instance id in the
  log matches what the adapter saw. The worker log will show
  `RemoteG2: no peer endpoint registered for worker_id=<dynamo-id>`
  if the adapter did not translate.
- `plan_blocks_loaded = 0` while `plan_resolved_count = 1`: hash
  format mismatch — confirm `block_hash_to_router_int` uses the
  full-bytes mask (not slice) and that `RemoteG2Descriptor.block_hash`
  is populated from `kv_hash` (not `identity_hash`). Both fixes are
  in the vLLM commits referenced in §2.7(a).

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

We validated the PoC along two independent strands:

* **Strand A** — a live run of the complete Dynamo + vLLM + KV-P2P
  stack, exercising the actual HTTP entry point all the way to NIXL
  READ on the data plane.
* **Strand B** — offline correctness evidence (unit tests + a
  no-Dynamo two-engine byte-equality eval) that pins down protocol
  invariants the live run alone cannot prove.

### 4.1 Live end-to-end run (Dynamo + vLLM + KV-P2P)

**Stack actually running.** Two co-located `dynamo.vllm` workers on
one host, Qwen3-8B, **real NIXL UCX** transport (not mock), real
Dynamo Frontend on `:8000` with `--router-mode kv` backed by real
NATS + etcd. The Dynamo runtime was built from source on
`feat/kv-p2p-vllm-bridge` (rebased onto
`oandreeva-kv-p2p-v1-followups`), so `remote_g2_plan.rs` is present
in the wheel and `select_remote_g2_reuse_plan` is reachable from the
KV-router hot path. Plan emission in this run was done by the POC
shim `kvp2p_plan_inject.py` (see §2.7(b)) — the rationale is in
§2.7(a): in a co-located 2-worker toy setup the Router's native
plan predicate intentionally short-circuits (`target == source` is
not a useful pull), so to actually move bytes over the wire we
inject from the shim. The native code path runs end-to-end up to
that predicate; the mechanical wiring (`enable_kv_cache_events`,
events on `tcp://*:556X`, `CPU_PINNED` medium, host-pinned
indexer populated) was verified independently in this same run.

**Request chain exercised.** Each request travelled the full path:

```
curl /v1/chat/completions
   → Dynamo Frontend HTTP
   → KV Router (selects target worker via prefix-aware routing)
   → dynamo.vllm handler on the target worker
   → handler patch invokes maybe_inject_plan(): ZMQ stats RPC to the
     peer worker's source registry → "all-hashes" RemoteKvReusePlan
     written into sampling_params.extra_args
   → vLLM EngineCore picks up the plan
   → RemoteG2OffloadingManager.lookup() peeks plan, calls
     TargetG2RpcClient.resolve_and_lease() over ZMQ to the peer
     worker's SourceG2RpcServer
   → peer's SourceG2DescriptorRegistry pins each requested block in
     its CPU pool's LRUCachePolicy and returns descriptors + lease_id
   → vLLM scheduler emits a RemoteG2LoadSpec → GPULoadStoreSpec
     transfer
   → RemoteG2TransferHandler issues per-layer NIXL UCX READ via the
     RawNixlRemoteG2Adapter; bytes land in the target worker's GPU
     KV slots
   → vLLM completes prefill on whatever the plan didn't cover and
     decodes
   → HTTP response with the chat completion
   → on request-finished, manager calls release_lease() → peer
     unpins the blocks
```

**Workload.** A small batch of long-prompt chat-completion requests
(each prompt long enough to fill several 16-token offload blocks)
against the Dynamo Frontend. All requests returned HTTP 200 with
valid completions.

**Observed metrics, pulled from each worker's `stats` RPC after the
run** (this is the canonical answer to "did the end-to-end stack
work?"):

| Metric | Worker 1 | Worker 2 |
|---|---|---|
| `pin_count_total` / `unpin_count_total` | 20 / 20 | 20 / 20 |
| `pin_failures` | 0 | 0 |
| `policy_registered` | True | True |
| `target_stats.plan_seen_count` | 5 | 5 |
| `target_stats.plan_resolved_count` | 5 | 5 |
| `target_stats.plan_blocks_loaded` | 20 | 20 |
| HTTP request failures / `Traceback` / `TransferResult success=False` | 0 | 0 |

How to read this:

* `plan_seen == plan_resolved` on both workers (5/5) → every plan
  that the shim injected was successfully resolved by the peer
  source registry (no `pin_failed`, no `wrong_owner`, no rejections).
* `plan_blocks_loaded` sums to **40** (20 per worker): forty KV
  blocks moved between the two workers over **real** NIXL UCX during
  the run.
* `pin / unpin` ratios 20/20 on both workers are exactly balanced →
  every lease that was granted got released; the source registry's
  working set is fully reclaimable.
* `pin_failures = 0` → no silent failure where the source registry
  found a descriptor but couldn't pin the underlying block.
* 0 stack traces and 0 transfer failures → the data plane and the
  HTTP control plane stayed healthy throughout.

Reproducing these exact numbers requires the launcher knobs
documented in §2.5 — in particular, `--no-enable-prefix-caching`,
`--gpu-memory-utilization 0.2` paired with a 16 GiB
`cpu_bytes_to_use`, and `tcp://*:556X` (bind) for `--kv-events-config`.
Without `--no-enable-prefix-caching`, the second wave of identical
prompts hits the GPU prefix cache and the vLLM scheduler short-circuits
`OffloadingConnector.lookup`, so `_peek_plan` is never called and
`plan_seen_count` stays 0 even with a perfectly valid plan in
`extra_args` — see §4.3 of `DESIGN.md` for the mechanism.

This is the answer to "did Dynamo + vLLM + KV-P2P end-to-end
actually work?" — yes, by the only meaningful definitions of "end to
end": HTTP requests succeeded, the plan path fired through every
component, and bytes verifiably moved over the real transport.

### 4.2 Offline correctness evidence

Two things the live run above cannot answer on its own:

1. *Does the plan-resolve protocol handle every failure mode cleanly,
   or does it just happen to work on healthy traffic?*
2. *When the target reuses the source's KV via NIXL instead of
   recomputing prefill, is the model output byte-for-byte identical?*

The unit-test surface and the two-engine byte-equality eval answer
both.

**Unit tests** (`tests/v1/kv_offload/remote_g2/`):

| Suite | What it pins down | # tests | Status |
|---|---|---|---|
| `test_e2e_smoke.py` | End-to-end load path with mock-NIXL; sanity-checks framework wiring | 4 | ✅ |
| `test_lease_pin.py` | Pin/unpin balance, TTL expiry, 8-thread / 500-iter concurrency, 1000-lease stress, no slow leak under mixed full/partial/miss workloads | 9 | ✅ |
| `test_plan_miss.py` | All rejection reasons from `resolve_and_lease`: full miss, partial-prefix miss, `wrong_source_worker`, `wrong_source_rank`, `unsupported_plan_version`, `plan_expired`, `invalid_plan`, empty plan, post-release cleanliness | 10 | ✅ |
| `test_failure_recovery.py` | NIXL fault injection (`fault_inject_every` knob): every-Nth READ raises → `TransferResult.success=False` → the **manager's** state machine recovers cleanly (lease released, resolve cache cleared, no stuck state across consecutive failures). Note: end-to-end propagation into the scheduler's `kv_load_failure_policy` is blocked by the upstream `assert transfer_result.success` in the offloading worker loop — see §4.3 status. | 7 | ✅ |

**30 / 30 pass.**

**Two-engine byte-equality eval** (no Dynamo involved):
`two_engines/orchestrate.py` drives two real `vllm.LLM` engines
directly:

1. Source engine runs every prompt and writes a reference output.
2. Target engine runs the same prompts under three configurations
   (cold-baseline, plan-driven cross-engine pull, hot-baseline).
3. Each target completion is compared byte-for-byte against the
   source's reference.

The plan-driven path produces output **byte-identical** to the
recompute path on the matching prompts — the strongest correctness
signal we have for "the KV cache pulled over NIXL is truly the same
KV as if we had recomputed". Detailed numbers in `FINDINGS.md`.

### 4.3 Status summary

| Aspect | State |
|---|---|
| Source RPC + plan resolve (cross-worker) | ✅ working |
| Per-layer (all 36 layers) NIXL READ | ✅ working |
| Lease pin / unpin / TTL | ✅ working, balanced |
| Plan-miss / partial-prefix degradation at the manager state machine | ✅ working, no leases leaked (validated by `test_failure_recovery.py`) |
| NIXL failure → vLLM `recompute` policy end-to-end | ⚠️ open. Our adapter correctly reports `TransferResult.success=False`, but the upstream offloading worker loop at `vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:273` runs `assert transfer_result.success` before the scheduler's `kv_load_failure_policy` can apply. Tracked as a separate proposal with the offloading-connector owners. |
| End-to-end output byte equality vs source reference | ✅ verified (two-engine eval) |
| Dynamo HTTP frontend + KV router → vLLM workers | ✅ working |
| Plan emission via POC shim (`kvp2p_plan_inject.py`) | ✅ working — the verified path that produced the 40-block numbers above. Also useful going forward as a deterministic injection knob for testing the data plane without depending on the Router's reuse predicate. |
| Native plan emission by the Router (`remote_g2_plan.rs`) | ✅ **verified end-to-end** on the `x-worker-instance-id` co-located reproducer (§2.9). Phase 2's request reaches W2 with the Router-attached plan, W2 resolves it against W1's source registry, W1 pins all 19 blocks, the per-layer NIXL UCX READ moves the bytes, and W2 serves the prompt **without writing it to its own CPU pool** (`W2.descriptor_count = 0`). W1 `pin/unpin = 19/19` balanced, `pin_failures = 0`. Required five paired fixes across the two repos — see §2.7(a) for the patch list. |
| Peer-endpoint discovery (target → source RPC) | ⚠️ POC uses static `peer_endpoints` config / `KVP2P_PEER_SOCKETS` env var. `RemoteG2OffloadingSpec.set_peer_endpoint()` is wired for dynamic injection but no discovery layer calls it on this branch. Planned follow-up: a Dynamo-etcd-backed registry that is symmetric across shim and native plan sources. |
| Mixed local + remote within one load batch | ⚠️ all-or-nothing per batch. `lookup()` composes correctly per-key (local hit precedes plan path), but `prepare_load()` falls back wholesale to the local path if any key in the batch is uncovered by the plan. Loses an opportunity (not correctness); tracked as a follow-up to split into separate `LoadStoreSpec`s. |
| Tensor parallelism | Single-TP only. `set_pool_layout` is called with `rank=0, num_workers=1`; registry, source RPC server, and NIXL agent are one-instance-per-process. Multi-TP wire shape (per-rank endpoints vs. one fan-out endpoint; per-rank pinned-memory layout) is open design work. |
| Cross-host deployment + UCX device tuning | not yet exercised |
| Performance benchmarks (TTFT / throughput) | not yet measured |
| `dp_rank > 0` live | not yet exercised |
| Compiled-graph (non eager) | not yet exercised |

The functional path is closed end-to-end; the open items are
deployment-shape, multi-rank/multi-TP scaling, and performance work,
plus the upstream `assert transfer_result.success` gap that has to be
resolved before failure handling can be claimed end-to-end.

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
