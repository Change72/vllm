#!/bin/bash
# Launcher for one TP=2 vLLM worker behind dynamo runtime, configured for the
# Remote-G2 (KV-P2P) end-to-end path.
#
# Required positional args:
#   $1 WORKER_ID   integer worker id (e.g. 1 or 2); used by
#                  REMOTE_G2_SOURCE_WORKER_ID and the canonical source RPC
#                  socket path /tmp/dynamo_remote_g2_w${WORKER_ID}.sock
#   $2 GPUS        comma-separated CUDA device list, e.g. "4,5"
#
# Optional positional args:
#   $3 ENDPOINT    dynamo endpoint name, default "generate"
#   $4 NAMESPACE   dynamo namespace, default "kvp2p"
#
# Environment overrides (all optional, defaults below are what the
# 2-worker × TP=2 Qwen3-8B repro doc was validated with):
#   MODEL_PATH         path to model weights (default /raid/fly/model/Qwen3-8B)
#   TP_SIZE            tensor-parallel size (default 2)
#   GPU_MEM_UTIL       --gpu-memory-utilization (default 0.2; H20-3e tuning)
#   CPU_POOL_BYTES     per-worker host-pinned offload pool size, bytes
#                      (default 16 GiB = 17179869184; reduce on smaller hosts)
#   MAX_MODEL_LEN      --max-model-len (default 8192)
#   MAX_NUM_SEQS       --max-num-seqs  (default 4; the §6 burst test
#                      assumes this to saturate W1 quickly)
#   BLOCK_SIZE         --block-size    (default 16)
#   EVENT_PORT_BASE    ZMQ port base for KV events; per-worker port is
#                      EVENT_PORT_BASE + WORKER_ID (default 5560 -> w1=5561, w2=5562)
#   ETCD_ENDPOINTS     default http://127.0.0.1:2379
#   NATS_SERVER        default nats://127.0.0.1:4222
#
set -e
WORKER_ID="${1:?worker_id required (e.g. 1 or 2)}"
GPUS="${2:?gpus required, e.g. 4,5}"
ENDPOINT="${3:-generate}"
NAMESPACE="${4:-kvp2p}"

MODEL_PATH="${MODEL_PATH:-/raid/fly/model/Qwen3-8B}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.2}"
CPU_POOL_BYTES="${CPU_POOL_BYTES:-17179869184}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
EVENT_PORT_BASE="${EVENT_PORT_BASE:-5560}"

export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES="$GPUS"
export ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-http://127.0.0.1:2379}"
export NATS_SERVER="${NATS_SERVER:-nats://127.0.0.1:4222}"
export REMOTE_G2_SOURCE_WORKER_ID="$WORKER_ID"

SOCKET="/tmp/dynamo_remote_g2_w${WORKER_ID}.sock"
# Per-rank sockets follow the convention {base}_tp{rank}.sock
# (see RemoteG2OffloadingSpec._rank_scoped_socket_path).
rm -f "${SOCKET}" "${SOCKET%.*}_tp0.sock" "${SOCKET%.*}_tp1.sock"
export REMOTE_G2_SOURCE_RPC_SOCKET_PATH="$SOCKET"

if [ "$WORKER_ID" = "1" ]; then
    PEER_WID=2
    PEER_SOCK="/tmp/dynamo_remote_g2_w2.sock"
else
    PEER_WID=1
    PEER_SOCK="/tmp/dynamo_remote_g2_w1.sock"
fi

# In the native-router-plan setup, the dynamo handlers.py shim path is a
# fallback that only fires when the native adapter did NOT inject
# extra_args["remote_kv_reuse_plan"]. We leave KVP2P_PEER_SOCKETS unset so
# the shim's peer-discovery path is also disabled — that way even on a
# prebuilt dynamo (no native remote_g2_plan.rs), there is no surprise plan
# injection. Set this if you specifically want the shim fallback to kick in.
unset KVP2P_PEER_SOCKETS

EVENT_ENDPOINT="tcp://*:$((EVENT_PORT_BASE + WORKER_ID))"

exec python3 -m dynamo.vllm \
  --namespace "$NAMESPACE" \
  --endpoint "$NAMESPACE.worker.$ENDPOINT" \
  --model "$MODEL_PATH" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --enforce-eager \
  --block-size "$BLOCK_SIZE" \
  --enable-prefix-caching \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --kv-events-config "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"${EVENT_ENDPOINT}\"}" \
  --kv-transfer-config "{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"RemoteG2OffloadingSpec\",\"cpu_bytes_to_use\":${CPU_POOL_BYTES},\"source_worker_id\":${WORKER_ID},\"source_dp_rank\":0,\"source_rpc_socket_path\":\"${SOCKET}\",\"use_mock_nixl\":false,\"peer_endpoints\":\"${PEER_WID}=${PEER_SOCK}\"}}"
