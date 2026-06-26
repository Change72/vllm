"""End-to-end ShareGPT cross-instance KV reuse test.

Two phases:
  Phase 1 (warm):  send all N prompts to W1 -> W1 caches them in its
                   host-pinned CPU pool, publishes BlockStored events
                   so the dynamo router learns the prefix-hash trie.
  Phase 2 (reuse): send the SAME N prompts to W2 with x-worker-instance-id
                   pointing at W2 -> router injects a RemoteKvReusePlan
                   referencing W1, W2 NIXL-reads the KV from W1 host pool,
                   skips prefill, decodes.

Verifications:
  (a) Every Phase 1 vs Phase 2 output is byte-identical (greedy decode
      from byte-identical KV must reproduce the same tokens).
  (b) W2's plan counters strictly advance (one plan per prompt).
  (c) W1's pin == unpin (no lease leak) and pin_failures == 0.
  (d) Sample outputs are eyeballed in the script's last section so we
      can confirm the model is producing semantically reasonable answers.
"""
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
MAX_TOKENS = 80
N_PROMPTS = 16  # 16 distinct ShareGPT prompts sharing a system prefix

with open("/work/kvp2p_sharegpt_sample.json") as f:
    DATA = json.load(f)
PROMPTS = [it["prompt"] for it in DATA["items"][:N_PROMPTS]]
QUESTIONS = [it["user_question"][:120] + "..." for it in DATA["items"][:N_PROMPTS]]


def post(prompt, instance, max_tokens=MAX_TOKENS, idx=0):
    with open(f"/tmp/payload_{idx}.json", "w") as f:
        f.write(json.dumps({
            "model": MODEL, "prompt": prompt,
            "max_tokens": max_tokens, "temperature": 0,
        }))
    t0 = time.perf_counter()
    out = subprocess.run([
        "curl", "-s", "--fail",
        "-X", "POST",
        "-H", "Content-Type: application/json",
        "-H", f"x-worker-instance-id: {instance}",
        "--data-binary", f"@/tmp/payload_{idx}.json",
        "http://127.0.0.1:8001/v1/completions",
    ], capture_output=True, text=True, timeout=180)
    elapsed = time.perf_counter() - t0
    if out.returncode != 0:
        raise RuntimeError(f"curl rc={out.returncode}: {out.stderr[:200]}")
    return json.loads(out.stdout)["choices"][0]["text"], elapsed


def stats(sock):
    from vllm.v1.kv_offload.remote_g2.target_client import TargetG2RpcClient
    c = TargetG2RpcClient(sock, timeout_ms=5000)
    try:
        return c.stats(sample_limit=0) or {}
    finally:
        c.close()


def short(s):
    t = s.get("target_stats", {})
    return (f"desc={s.get('descriptor_count')} "
            f"p/u/f={s.get('pin_count_total')}/"
            f"{s.get('unpin_count_total')}/{s.get('pin_failures')} "
            f"seen={t.get('plan_seen_count')} "
            f"res={t.get('plan_resolved_count')} "
            f"loaded={t.get('plan_blocks_loaded')}")


def banner(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


banner("INITIAL")
print("W1:", short(stats(W1_SOCK)))
print("W2:", short(stats(W2_SOCK)))

banner(f"PHASE 1: warm W1 with {N_PROMPTS} ShareGPT prompts")
phase1_outputs = []
phase1_lats = []
for i, p in enumerate(PROMPTS):
    txt, e = post(p, W1, MAX_TOKENS, idx=i)
    phase1_outputs.append(txt)
    phase1_lats.append(e)
    print(f"  [{i:02d}] W1 {e:5.2f}s  {txt[:90]!r}")
# Sleep so router definitely sees all BlockStored events before phase 2.
time.sleep(5)
print("\nW1:", short(stats(W1_SOCK)))
print("W2:", short(stats(W2_SOCK)))

banner(f"PHASE 2: same prompts -> W2 with x-worker-instance-id (Remote-G2)")
phase2_outputs = []
phase2_lats = []
for i, p in enumerate(PROMPTS):
    txt, e = post(p, W2, MAX_TOKENS, idx=i)
    phase2_outputs.append(txt)
    phase2_lats.append(e)
    print(f"  [{i:02d}] W2 {e:5.2f}s  {txt[:90]!r}")
time.sleep(5)
s1 = stats(W1_SOCK)
s2 = stats(W2_SOCK)
print("\nW1:", short(s1))
print("W2:", short(s2))

banner("VERIFICATION")
n_match = sum(1 for a, b in zip(phase1_outputs, phase2_outputs) if a == b)
print(f"  outputs identical: {n_match}/{N_PROMPTS}")
plan_seen = s2.get("target_stats", {}).get("plan_seen_count", 0)
plan_resolved = s2.get("target_stats", {}).get("plan_resolved_count", 0)
plan_blocks_loaded = s2.get("target_stats", {}).get("plan_blocks_loaded", 0)
pin_w1 = s1.get("pin_count_total", 0)
unpin_w1 = s1.get("unpin_count_total", 0)
pin_fail_w1 = s1.get("pin_failures", 0)
print(f"  W2 plan: seen={plan_seen}, resolved={plan_resolved}, blocks_loaded={plan_blocks_loaded}")
print(f"  W1 pin/unpin/fail: {pin_w1}/{unpin_w1}/{pin_fail_w1}")
print(f"  Phase 1 mean latency: {sum(phase1_lats)/len(phase1_lats):.2f}s")
print(f"  Phase 2 mean latency: {sum(phase2_lats)/len(phase2_lats):.2f}s")

ok = (
    n_match == N_PROMPTS
    and plan_seen >= N_PROMPTS
    and plan_resolved >= N_PROMPTS
    and plan_blocks_loaded > 0
    and pin_w1 == unpin_w1
    and pin_w1 > 0
    and pin_fail_w1 == 0
)
print()
print("PASS" if ok else "FAIL")

banner("SAMPLE Q+A from Phase 2 (eyeball)")
for i in [0, 3, 7, 11, 15]:
    print(f"\n--- Q[{i}]: {QUESTIONS[i]}")
    print(f"--- A: {phase2_outputs[i][:300]}")
