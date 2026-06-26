"""Native-router plan test (no x-worker-instance-id).

Phase 1 (warm):   send N prompts to W1 sequentially WITH x-worker-instance-id=W1
                  so they cache on W1. No remote transfer happens since W1 is
                  the source -- shim plan during warming is harmless.

Phase 2 (burst):  send a concurrent burst of the SAME N prompts WITHOUT any
                  routing header. With W1 at max-num-seqs=4, the dynamo
                  kv-router should pick W2 for the overflow and -- because
                  W2's cache is empty but W1 has the prefix -- inject a
                  NATIVE RemoteKvReusePlan referencing W1.

Verifications:
  (a) At least one Phase 2 response was handled by W2 via Remote-G2 (look at
      W2 worker log for `prepare_load -> RemoteG2LoadSpec` lines; check the
      plan_id is NOT 'router-shim-unknown').
  (b) For Phase 2 requests handled by W2, the decoded text matches W1's
      ground-truth (Phase 1) output.
"""
import concurrent.futures
import json
import subprocess
import sys
import time
import re

sys.path.insert(0, "/work/vllm-src")

import os
W1 = os.environ["W1_INSTANCE_ID"]
W2 = os.environ["W2_INSTANCE_ID"]
W1_SOCK = "/tmp/dynamo_remote_g2_w1.sock"
W2_SOCK = "/tmp/dynamo_remote_g2_w2.sock"
MODEL = "/raid/fly/model/Qwen3-8B"
MAX_TOKENS = 60
N_PROMPTS = 16
BURST_SIZE = 16

with open("/work/kvp2p_sharegpt_sample.json") as f:
    DATA = json.load(f)
PROMPTS = [it["prompt"] for it in DATA["items"][:N_PROMPTS]]


def post(prompt, instance=None, idx=0):
    payload_path = f"/tmp/payload_{idx}.json"
    with open(payload_path, "w") as f:
        f.write(json.dumps({
            "model": MODEL, "prompt": prompt,
            "max_tokens": MAX_TOKENS, "temperature": 0,
        }))
    headers = ["-H", "Content-Type: application/json"]
    if instance is not None:
        headers += ["-H", f"x-worker-instance-id: {instance}"]
    t0 = time.perf_counter()
    out = subprocess.run([
        "curl", "-s", "--fail",
        "-X", "POST",
        *headers,
        "--data-binary", f"@{payload_path}",
        "http://127.0.0.1:8001/v1/completions",
    ], capture_output=True, text=True, timeout=180)
    elapsed = time.perf_counter() - t0
    if out.returncode != 0:
        return (None, elapsed, out.stderr[:200])
    body = json.loads(out.stdout)
    req_id = body.get("id", "")
    text = body["choices"][0]["text"]
    return (text, elapsed, req_id)


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

banner(f"PHASE 1: warm W1 sequentially with {N_PROMPTS} ShareGPT prompts (x-worker=W1)")
phase1 = []
for i, p in enumerate(PROMPTS):
    text, e, req_id = post(p, instance=W1, idx=i)
    if text is None:
        print(f"  [{i:02d}] ERROR: {req_id}")
        phase1.append("")
    else:
        phase1.append(text)
        print(f"  [{i:02d}] {e:5.2f}s  {text[:70]!r}")
time.sleep(8)
print()
print("W1:", short(stats(W1_SOCK)))
print("W2:", short(stats(W2_SOCK)))

banner(f"PHASE 2: CONCURRENT burst of same {BURST_SIZE} prompts, NO x-worker header")
phase2 = [None] * BURST_SIZE
t_start = time.time()
with concurrent.futures.ThreadPoolExecutor(BURST_SIZE) as ex:
    futures = {ex.submit(post, PROMPTS[i % N_PROMPTS], None, idx=100 + i): i
               for i in range(BURST_SIZE)}
    for fut in concurrent.futures.as_completed(futures):
        i = futures[fut]
        text, e, req_id = fut.result()
        phase2[i] = (text, e, req_id)
        flag = "ok" if text else "ERR"
        if text:
            print(f"  [{i:02d}] {flag} {e:5.2f}s req={req_id[5:13]} {text[:60]!r}")
        else:
            print(f"  [{i:02d}] {flag} {e:5.2f}s {req_id[:100]}")
print(f"\nburst wall-time {time.time() - t_start:.2f}s")
time.sleep(8)
s1 = stats(W1_SOCK)
s2 = stats(W2_SOCK)
print("W1:", short(s1))
print("W2:", short(s2))

banner("WORKER LOG: plan_id observed for Phase 2 requests")


def grep_log(path, pattern, n=200):
    out = subprocess.run(
        ["grep", "-aE", pattern, path],
        capture_output=True, text=True,
    )
    return out.stdout.splitlines()[-n:]


# Extract `RemoteG2: req X plan Y resolved: N descriptors` lines from worker logs.
plan_lines_w1 = grep_log("/tmp/w1.log", r"RemoteG2: req .+ plan .+ resolved:")
plan_lines_w2 = grep_log("/tmp/w2.log", r"RemoteG2: req .+ plan .+ resolved:")


def parse_plan(line):
    m = re.search(r"req (\S+).+plan (\S+) resolved: (\d+) descriptors", line)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


parsed_w1 = [p for p in (parse_plan(l) for l in plan_lines_w1) if p]
parsed_w2 = [p for p in (parse_plan(l) for l in plan_lines_w2) if p]
print(f"W1 saw {len(parsed_w1)} plan resolve events")
print(f"W2 saw {len(parsed_w2)} plan resolve events")
print()

# Filter to plan resolves seen during Phase 2 (last burst_size events on each).
phase2_req_ids = {p[2][5:] for p in phase2 if p and p[2]}  # strip 'cmpl-' prefix
print("Phase 2 req_ids (first 8 chars):",
      sorted({r[:8] for r in phase2_req_ids})[:8], "...")
print()

shim_count_w2 = 0
native_count_w2 = 0
for req_id, plan_id, n_desc in parsed_w2[-BURST_SIZE * 2:]:
    is_shim = "router-shim-unknown" in plan_id
    tag = "SHIM" if is_shim else "NATIVE"
    print(f"  W2  {tag:6s} req={req_id[:13]} plan={plan_id[:40]} desc={n_desc}")
    if is_shim:
        shim_count_w2 += 1
    else:
        native_count_w2 += 1

print()
print(f"W2 plan classification: native={native_count_w2}  shim={shim_count_w2}")

banner("OUTPUT CHECK")
n_match = 0
n_total_compared = 0
for i, item in enumerate(phase2):
    if item is None or item[0] is None:
        continue
    n_total_compared += 1
    ground = phase1[i % N_PROMPTS]
    if not ground:
        continue
    if item[0] == ground:
        n_match += 1
    else:
        # Find divergence point.
        diverge = 0
        while diverge < len(ground) and diverge < len(item[0]) and ground[diverge] == item[0][diverge]:
            diverge += 1
        print(f"  [{i:02d}] DIFF at char {diverge}")
        print(f"     gt:  ...{ground[max(0, diverge-20):diverge+60]!r}")
        print(f"     got: ...{item[0][max(0, diverge-20):diverge+60]!r}")

print()
print(f"identical: {n_match}/{n_total_compared}")
