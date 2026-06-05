# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine runner used by the two-engine evaluation.

Source mode: warms up by running all prompts in the prompts file once
per warmup round, then captures one final reference pass and writes
``source_outputs.json`` (an indexed list of generated texts per prompt
index). Idles until told to stop.

Target mode: per-cycle protocol. Each cycle the orchestrator drops a
``run_cycle_<n>.json`` file containing ``{label, prompt_indices,
plan}``. The runner reads the indexed prompts from the prompts file,
runs them through ``llm.generate`` (with the plan injected via
``kv_transfer_params`` when present), captures the texts, polls the
target-side plan counters via ZMQ on the local engine's source RPC
socket, and writes ``result_cycle_<n>.json``. Repeats until ``stop``.

Both modes set ``PYTHONHASHSEED`` deterministically via the launching
orchestrator so block hashes match across engines.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["source", "target"], required=True)
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--peer-socket-path", default="")
    parser.add_argument("--source-worker-id", type=int, required=True)
    parser.add_argument("--peer-worker-id", type=int, default=-1)
    parser.add_argument("--status-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--use-mock-nixl", type=int, default=0)
    parser.add_argument("--source-warmup-rounds", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.18)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=12)
    return parser.parse_args()


def _write_status(status_dir: Path, name: str, payload) -> None:
    path = status_dir / name
    if isinstance(payload, str):
        path.write_text(payload)
    else:
        path.write_text(json.dumps(payload, indent=2))


def _read_target_stats(socket_path: str) -> dict:
    from vllm.v1.kv_offload.remote_g2.target_client import TargetG2RpcClient

    client = TargetG2RpcClient(socket_path, timeout_ms=5_000)
    try:
        stats = client.stats(sample_limit=1) or {}
        return dict(stats.get("target_stats", {}))
    finally:
        client.close()


def main() -> int:
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    status_dir = Path(args.status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    prompts = Path(args.prompts_file).read_text().splitlines()
    prompts = [p for p in prompts if p.strip()]
    print(f"[runner {args.mode}] loaded {len(prompts)} prompts", flush=True)

    try:
        os.unlink(args.socket_path)
    except FileNotFoundError:
        pass

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    extra: dict = {
        "spec_name": "RemoteG2OffloadingSpec",
        "cpu_bytes_to_use": int(2 * 1024**3),
        "source_worker_id": args.source_worker_id,
        "source_dp_rank": 0,
        "source_rpc_socket_path": args.socket_path,
        "use_mock_nixl": bool(args.use_mock_nixl),
    }
    if args.mode == "target" and args.peer_socket_path and args.peer_worker_id >= 0:
        extra["peer_endpoints"] = f"{args.peer_worker_id}={args.peer_socket_path}"

    kv_cfg = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra,
    )

    print(f"[runner {args.mode}] booting LLM, gpu={args.gpu}", flush=True)
    llm = LLM(
        model=args.model,
        kv_transfer_config=kv_cfg,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        block_size=16,
        enable_prefix_caching=True,
        max_num_seqs=4,
    )
    print(f"[runner {args.mode}] LLM ready, socket={args.socket_path}", flush=True)

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    if args.mode == "source":
        # Warmup with the FULL prompt set so the host pool fills with
        # entries for every prompt (the registry needs hashes from
        # prompts 0..N-1 so target plan-driven loads can resolve).
        for r in range(args.source_warmup_rounds):
            t0 = time.perf_counter()
            llm.generate(prompts, sampling, use_tqdm=False)
            print(
                f"[runner source] warmup round {r} (all 16 prompts) "
                f"{(time.perf_counter() - t0) * 1000:.0f} ms",
                flush=True,
            )
        # Reference pass — match the target's batch composition so the
        # source / target output comparison is apples-to-apples
        # (vLLM v1's async + chunked-prefill scheduling makes the
        # first generated token under temperature=0 depend on the
        # batch contents; running 8-at-a-time on source mirrors what
        # the target does per cycle).
        half = len(prompts) // 2
        t0 = time.perf_counter()
        out_first = llm.generate(prompts[:half], sampling, use_tqdm=False)
        out_second = llm.generate(prompts[half:], sampling, use_tqdm=False)
        ref_ms = (time.perf_counter() - t0) * 1000
        reference_texts = (
            [o.outputs[0].text for o in out_first]
            + [o.outputs[0].text for o in out_second]
        )
        _write_status(
            status_dir,
            "source_outputs.json",
            {"texts": reference_texts, "generate_ms": ref_ms},
        )
        print(
            f"[runner source] reference (2x{half}-prompt batches) "
            f"{ref_ms:.0f} ms; idling",
            flush=True,
        )
        _write_status(status_dir, "source_ready", "ok")
        while not (status_dir / "stop").exists():
            time.sleep(0.2)
        return 0

    # target mode: cycle loop.
    _write_status(status_dir, "target_ready", "ok")
    print(f"[runner target] entering cycle loop", flush=True)

    cycle = 0
    while not (status_dir / "stop").exists():
        cycle_req = status_dir / f"run_cycle_{cycle}.json"
        if not cycle_req.exists():
            time.sleep(0.2)
            continue

        spec = json.loads(cycle_req.read_text())
        label = spec.get("label", f"cycle{cycle}")
        plan = spec.get("plan")
        indices = spec.get("prompt_indices") or list(range(len(prompts)))
        cycle_prompts = [prompts[i] for i in indices]

        if plan is not None:
            sampling_call = SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                extra_args={"kv_transfer_params": {"remote_g2_plan": plan}},
            )
        else:
            sampling_call = sampling

        pre = _read_target_stats(args.socket_path)
        t0 = time.perf_counter()
        outputs = llm.generate(cycle_prompts, sampling_call, use_tqdm=False)
        dt_ms = (time.perf_counter() - t0) * 1000
        post = _read_target_stats(args.socket_path)
        delta = {
            k: int(post.get(k, 0)) - int(pre.get(k, 0))
            for k in (
                "plan_seen_count",
                "plan_resolved_count",
                "plan_load_specs_emitted",
                "plan_blocks_loaded",
            )
        }
        texts = [o.outputs[0].text for o in outputs]
        result = {
            "label": label,
            "prompt_indices": indices,
            "generate_ms": dt_ms,
            "ms_per_prompt": dt_ms / max(len(cycle_prompts), 1),
            "texts": texts,
            "delta_stats": delta,
            "post_stats": {k: int(post.get(k, 0)) for k in delta},
        }
        _write_status(status_dir, f"result_cycle_{cycle}.json", result)
        print(
            f"[runner target] cycle {cycle} ({label}): "
            f"{len(cycle_prompts)} prompts in {dt_ms:.0f} ms "
            f"({dt_ms / max(len(cycle_prompts), 1):.0f} ms/prompt), "
            f"delta={delta}",
            flush=True,
        )
        cycle_req.unlink()
        cycle += 1

    return 0


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except Exception:
        pass
    sys.exit(main())
