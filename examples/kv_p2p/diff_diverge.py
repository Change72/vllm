"""Re-run Phase 1 + 2 and show full diverging outputs for the
prompts that don't match."""
import json
import subprocess
import sys
import time

sys.path.insert(0, "/work/vllm-src")

import os
W1 = os.environ["W1_INSTANCE_ID"]
W2 = os.environ["W2_INSTANCE_ID"]
W1_SOCK = "/tmp/dynamo_remote_g2_w1.sock"
W2_SOCK = "/tmp/dynamo_remote_g2_w2.sock"
MODEL = "/raid/fly/model/Qwen3-8B"
MAX_TOKENS = 100
N_PROMPTS = 16

with open("/work/kvp2p_sharegpt_sample.json") as f:
    DATA = json.load(f)
PROMPTS = [it["prompt"] for it in DATA["items"][:N_PROMPTS]]


def post(prompt, instance, idx=0):
    with open(f"/tmp/payload_{idx}.json", "w") as f:
        f.write(json.dumps({
            "model": MODEL, "prompt": prompt,
            "max_tokens": MAX_TOKENS, "temperature": 0,
        }))
    out = subprocess.run([
        "curl", "-s", "--fail",
        "-X", "POST",
        "-H", "Content-Type: application/json",
        "-H", f"x-worker-instance-id: {instance}",
        "--data-binary", f"@/tmp/payload_{idx}.json",
        "http://127.0.0.1:8001/v1/completions",
    ], capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        raise RuntimeError(f"curl rc={out.returncode}: {out.stderr[:200]}")
    return json.loads(out.stdout)["choices"][0]["text"]


def stats(sock):
    from vllm.v1.kv_offload.remote_g2.target_client import TargetG2RpcClient
    c = TargetG2RpcClient(sock, timeout_ms=5000)
    try:
        return c.stats(sample_limit=0) or {}
    finally:
        c.close()


print("Phase 1: send all to W1 (already warm from prior runs)")
phase1 = [post(p, W1, idx=i) for i, p in enumerate(PROMPTS)]
print("Phase 2: send all to W2 (cross-instance)")
phase2 = [post(p, W2, idx=i) for i, p in enumerate(PROMPTS)]

print()
print("=" * 78)
for i in range(N_PROMPTS):
    match = phase1[i] == phase2[i]
    flag = "OK " if match else "DIFF"
    print(f"  [{i:02d}] {flag}")
    if not match:
        # Find first diverging char.
        diverge = 0
        while diverge < len(phase1[i]) and diverge < len(phase2[i]) and phase1[i][diverge] == phase2[i][diverge]:
            diverge += 1
        print(f"     diverge at char {diverge}")
        print(f"     W1: ...{phase1[i][max(0,diverge-30):diverge+80]!r}")
        print(f"     W2: ...{phase2[i][max(0,diverge-30):diverge+80]!r}")

n_match = sum(1 for a, b in zip(phase1, phase2) if a == b)
print(f"\n{n_match}/{N_PROMPTS} identical")
