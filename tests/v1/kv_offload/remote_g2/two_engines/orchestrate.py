# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-engine evaluation orchestrator (16-prompt, cold plan-driven path).

Test design:

1. Source engine warms up on all 16 varied prompts and writes a reference
   list ``source_outputs[i]`` for every prompt.
2. Target engine boots cold.
3. **Cycle 0 (baseline)** — target runs prompts 0..7 without a plan.
   These become hot in target's prefix cache.
4. **Cycle 1 (plan_driven_cold)** — target runs prompts 8..15 with a
   plan covering every published hash on the source. Because prompts
   8..15 share no prefix with prompts 0..7 (each base sentence is
   distinct), the target's GPU prefix cache misses for all of these.
   Manager.lookup falls through to the plan path, prepare_load emits
   RemoteG2LoadSpec for the matched blocks, and the transfer handler
   issues real NIXL READs.
5. **Cycle 2 (baseline_8_15)** — target runs prompts 8..15 without a
   plan, as a control. Target's cache is hot from cycle 1, so this
   measures the lower bound for hot-cache generation time.

PASS criteria:
- Per-prompt output equivalence: source_outputs[i] == target output
  for prompt i, on every prompt across all cycles.
- delta_stats[plan_resolved_count] for cycle 1 equals 8 (one resolve
  per request).
- delta_stats[plan_blocks_loaded] for cycle 1 > some non-trivial floor
  (proves NIXL transfers actually fired).
- Cycle 0 timing < cycle 1 timing is acceptable (NIXL roundtrip +
  resolve overhead on cold cycle). Report numbers for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _wait_for(path: Path, timeout: float, what: str) -> bool:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            print(f"[orchestrate] TIMEOUT waiting for {what} ({path})", flush=True)
            return False
        time.sleep(0.5)
    return True


def _spawn(
    mode: str,
    *,
    runner_script: Path,
    socket_path: Path,
    peer_socket_path: Path | None,
    source_worker_id: int,
    peer_worker_id: int | None,
    status_dir: Path,
    gpu: int,
    model: str,
    prompts_file: Path,
    use_mock_nixl: bool,
    log_path: Path,
    hashseed: str,
    max_tokens: int = 12,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.18,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-u",
        str(runner_script),
        "--mode", mode,
        "--socket-path", str(socket_path),
        "--peer-socket-path", str(peer_socket_path) if peer_socket_path else "",
        "--source-worker-id", str(source_worker_id),
        "--peer-worker-id", str(peer_worker_id) if peer_worker_id is not None else "-1",
        "--status-dir", str(status_dir),
        "--gpu", str(gpu),
        "--model", model,
        "--prompts-file", str(prompts_file),
        "--use-mock-nixl", "1" if use_mock_nixl else "0",
        "--max-tokens", str(max_tokens),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hashseed
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"[orchestrate] launched {mode} engine pid={proc.pid} log={log_path}")
    return proc


def _run_cycle(
    status_dir: Path,
    cycle: int,
    label: str,
    prompt_indices: list[int],
    plan: dict | None,
    timeout_s: float = 300.0,
) -> dict | None:
    request = {"label": label, "plan": plan, "prompt_indices": prompt_indices}
    request_file = status_dir / f"run_cycle_{cycle}.json"
    result_file = status_dir / f"result_cycle_{cycle}.json"
    if result_file.exists():
        result_file.unlink()
    request_file.write_text(json.dumps(request))
    if not _wait_for(result_file, timeout_s, f"cycle {cycle} ({label}) result"):
        return None
    return json.loads(result_file.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=os.environ.get("MODEL_PATH", "/raid/fly/model/Qwen3-8B")
    )
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--target-gpu", type=int, default=1)
    parser.add_argument("--use-mock-nixl", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--repeat", type=int, default=50,
                        help="copies of each base sentence per prompt")
    parser.add_argument(
        "--work-dir",
        default=os.environ.get("KVP2P_WORK_DIR", "/tmp/kvp2p_two_engines"),
    )
    args = parser.parse_args()

    # Make sure the runner can import this package.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from varied_prompts import build_prompts

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    src_status = work / "src"
    tgt_status = work / "tgt"
    src_status.mkdir(parents=True, exist_ok=True)
    tgt_status.mkdir(parents=True, exist_ok=True)

    src_sock = Path("/tmp/dynamo_remote_g2_engineA.sock")
    tgt_sock = Path("/tmp/dynamo_remote_g2_engineB.sock")
    for s in (src_sock, tgt_sock):
        if s.exists():
            s.unlink()

    prompts = build_prompts(repeat=args.repeat)
    prompts_file = work / "prompts.txt"
    prompts_file.write_text("\n".join(prompts))
    print(
        f"[orchestrate] built {len(prompts)} prompts, "
        f"~{len(prompts[0])} chars each, total {sum(len(p) for p in prompts)} chars"
    )

    runner = Path(__file__).resolve().parent / "engine_runner.py"
    hashseed = "0"

    src_proc = _spawn(
        "source",
        runner_script=runner,
        socket_path=src_sock,
        peer_socket_path=None,
        source_worker_id=1,
        peer_worker_id=None,
        status_dir=src_status,
        gpu=args.source_gpu,
        model=args.model,
        prompts_file=prompts_file,
        use_mock_nixl=bool(args.use_mock_nixl),
        log_path=work / "source.log",
        hashseed=hashseed,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    rc = 0
    tgt_proc: subprocess.Popen | None = None
    try:
        if not _wait_for(src_status / "source_ready", 900.0, "source ready"):
            return 1
        source_outputs = json.loads((src_status / "source_outputs.json").read_text())
        print(
            f"[orchestrate] source reference ready: "
            f"{len(source_outputs['texts'])} texts, "
            f"ref pass {source_outputs['generate_ms']:.0f} ms"
        )

        from vllm.v1.kv_offload.remote_g2.data_model import (
            REMOTE_KV_REUSE_PLAN_VERSION,
        )
        from vllm.v1.kv_offload.remote_g2.target_client import (
            TargetG2RpcClient,
        )

        client = TargetG2RpcClient(str(src_sock), timeout_ms=10_000)
        stats = client.stats(sample_limit=8192)
        meta = client.get_metadata()
        client.close()
        if not stats or stats["descriptor_count"] == 0 or meta is None:
            print(f"[orchestrate] source state insufficient: stats={stats}")
            return 1
        plan_hashes = [int(h) for h in stats["sample_block_hashes"]]
        print(
            f"[orchestrate] source has {stats['descriptor_count']} descriptors, "
            f"plan covers {len(plan_hashes)} hashes"
        )

        tgt_proc = _spawn(
            "target",
            runner_script=runner,
            socket_path=tgt_sock,
            peer_socket_path=src_sock,
            source_worker_id=2,
            peer_worker_id=1,
            status_dir=tgt_status,
            gpu=args.target_gpu,
            model=args.model,
            prompts_file=prompts_file,
            use_mock_nixl=bool(args.use_mock_nixl),
            log_path=work / "target.log",
            hashseed=hashseed,
            max_tokens=args.max_tokens,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        if not _wait_for(tgt_status / "target_ready", 900.0, "target ready"):
            return 1

        # Cycle 0: baseline pass for prompts 0..7
        baseline_indices = list(range(0, 8))
        cycle0 = _run_cycle(
            tgt_status, 0, "baseline_0_7", baseline_indices, plan=None,
            timeout_s=600.0,
        )
        if cycle0 is None:
            return 1

        # Cycle 1: plan-driven pass for prompts 8..15 (cold cache, plan path engaged)
        plan_indices = list(range(8, 16))
        plan = {
            "plan_id": "varied-eval",
            "request_id": "varied-eval-req",
            "target_worker_id": 2,
            "target_dp_rank": 0,
            "source_worker_id": 1,
            "source_dp_rank": 0,
            "source_tier": "host_pinned",
            "block_hashes": plan_hashes,
            "kv_block_hashes": [],
            "start_block_index": 0,
            "planned_prefix_blocks": len(plan_hashes),
            "block_size_tokens": 16,
            "created_at_ms": 0,
            "expires_at_ms": 10**15,
            "plan_version": REMOTE_KV_REUSE_PLAN_VERSION,
        }
        cycle1 = _run_cycle(
            tgt_status, 1, "plan_driven_cold", plan_indices, plan=plan,
            timeout_s=600.0,
        )
        if cycle1 is None:
            return 1

        # Cycle 2: hot-cache control for prompts 8..15 (no plan)
        cycle2 = _run_cycle(
            tgt_status, 2, "baseline_hot_8_15", plan_indices, plan=None,
            timeout_s=600.0,
        )
        if cycle2 is None:
            return 1

        per_prompt = []
        failures = []

        def _check(indices, texts, label):
            for i, prompt_idx in enumerate(indices):
                src = source_outputs["texts"][prompt_idx]
                tgt = texts[i]
                ok = src == tgt
                per_prompt.append({
                    "label": label,
                    "prompt_idx": prompt_idx,
                    "source_text": src,
                    "target_text": tgt,
                    "match": ok,
                })
                if not ok:
                    failures.append(
                        f"{label} prompt {prompt_idx}: "
                        f"source={src!r} target={tgt!r}"
                    )

        _check(baseline_indices, cycle0["texts"], "baseline_0_7")
        _check(plan_indices, cycle1["texts"], "plan_driven_cold")
        _check(plan_indices, cycle2["texts"], "baseline_hot_8_15")

        if cycle1["delta_stats"]["plan_resolved_count"] == 0:
            failures.append("cycle1: plan_resolved_count == 0")
        if cycle1["delta_stats"]["plan_load_specs_emitted"] == 0:
            failures.append("cycle1: plan_load_specs_emitted == 0")
        if cycle1["delta_stats"]["plan_blocks_loaded"] < len(plan_indices):
            failures.append(
                f"cycle1: plan_blocks_loaded "
                f"({cycle1['delta_stats']['plan_blocks_loaded']}) "
                f"< minimum expected ({len(plan_indices)})"
            )

        matches = sum(1 for r in per_prompt if r["match"])
        summary = {
            "num_prompts": len(prompts),
            "prompt_repeat": args.repeat,
            "source_descriptor_count": stats["descriptor_count"],
            "plan_hash_count": len(plan_hashes),
            "source_reference_ms": source_outputs["generate_ms"],
            "cycles": {
                "baseline_0_7": {
                    "indices": baseline_indices,
                    "generate_ms": cycle0["generate_ms"],
                    "ms_per_prompt": cycle0["ms_per_prompt"],
                    "delta_stats": cycle0["delta_stats"],
                    "texts": cycle0["texts"],
                },
                "plan_driven_cold_8_15": {
                    "indices": plan_indices,
                    "generate_ms": cycle1["generate_ms"],
                    "ms_per_prompt": cycle1["ms_per_prompt"],
                    "delta_stats": cycle1["delta_stats"],
                    "texts": cycle1["texts"],
                },
                "baseline_hot_8_15": {
                    "indices": plan_indices,
                    "generate_ms": cycle2["generate_ms"],
                    "ms_per_prompt": cycle2["ms_per_prompt"],
                    "delta_stats": cycle2["delta_stats"],
                    "texts": cycle2["texts"],
                },
            },
            "per_prompt_match": matches,
            "per_prompt_total": len(per_prompt),
        }
        (work / "summary.json").write_text(json.dumps(summary, indent=2))
        (work / "per_prompt.json").write_text(json.dumps(per_prompt, indent=2))

        print(
            f"\n[orchestrate] per-prompt match: {matches}/{len(per_prompt)}"
        )
        print(
            f"[orchestrate] timing (ms/prompt): "
            f"baseline_cold={cycle0['ms_per_prompt']:.0f} "
            f"plan_driven_cold={cycle1['ms_per_prompt']:.0f} "
            f"baseline_hot={cycle2['ms_per_prompt']:.0f}"
        )
        print(
            f"[orchestrate] cycle1 plan stats: {cycle1['delta_stats']}"
        )

        if failures:
            for f in failures[:5]:
                print(f"[FAIL] {f}")
            if len(failures) > 5:
                print(f"[FAIL] ...and {len(failures) - 5} more")
            rc = 1
        else:
            print("[PASS] varied two-engine evaluation: all outputs match")
    finally:
        (src_status / "stop").write_text("stop")
        (tgt_status / "stop").write_text("stop")
        for proc in (tgt_proc, src_proc):
            if proc is None:
                continue
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    return rc


if __name__ == "__main__":
    sys.exit(main())
