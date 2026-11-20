# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["apc", "session", "session-apc"], required=True
    )
    parser.add_argument("--regime", choices=["resident", "pressure"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--eviction-policy", choices=["whole", "prefix"], default="whole"
    )
    parser.add_argument("--kv-cache-mib", type=int)
    parser.add_argument("--shared-root", action="store_true")
    parser.add_argument("--scenario", choices=["shared_prefix"])
    args = parser.parse_args()
    from vllm import LLM, SamplingParams

    if args.mode == "apc" and args.eviction_policy != "whole":
        parser.error("--eviction-policy applies only to session mode")
    if args.shared_root and (args.mode == "apc" or args.regime != "resident"):
        parser.error("--shared-root requires resident session mode")
    if args.scenario and args.regime != "resident":
        parser.error("--scenario requires resident mode")
    budget = args.kv_cache_mib or (12288 if args.regime == "resident" else 3072)
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        scheduler_cls="session_perf_hooks."
        + ("APCMetricsScheduler" if args.mode == "apc" else "SessionMetricsScheduler"),
        enable_prefix_caching=args.mode != "session",
        async_scheduling=False,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_model_len=17024,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        kv_cache_memory_bytes=budget * 1024 * 1024,
        gpu_memory_utilization=0.04,
        block_size=16,
        seed=0,
        enable_trace_replay=True,
        attention_config={"backend": "TRITON_ATTN"},
        additional_config={"session_cache": {"eviction_policy": args.eviction_policy}},
    )
    unit = llm.get_tokenizer().encode(
        "Inspect the repository, run the tests, and summarize the results. ",
        add_special_tokens=False,
    )
    source = (unit * (16384 // len(unit) + 1))[:16384]
    completion = source[:16]
    snap = lambda reset=False: llm.llm_engine.engine_core.call_utility(
        "session_perf_snapshot", reset
    )
    scenarios = (
        ["serial_continue", "shared_prefix", "batch_continue"]
        if args.regime == "resident"
        else ["alternating_pressure"]
    )
    if args.scenario:
        scenarios = [args.scenario]
    records = []
    for scenario in scenarios:
        for repeat in range(-1, 2):
            assert llm.reset_prefix_cache()
            histories = {}
            versions = {}

            def run(
                ids, deltas, fork_from=None, histories=histories, versions=versions
            ):
                prompts = []
                params = []
                logical = []
                for sid, delta in zip(ids, deltas):
                    history = histories.setdefault(sid, [])
                    history.extend(delta)
                    version = versions.get(sid, 0)
                    prompts.append(
                        {"prompt_token_ids": history if args.mode == "apc" else delta}
                    )
                    params.append(
                        SamplingParams(
                            temperature=0,
                            max_tokens=16,
                            ignore_eos=True,
                            trace_decode_token_ids=completion,
                            extra_args={
                                "session_cache": {"id": sid, "version": version}
                            }
                            if args.mode != "apc"
                            else None,
                        )
                    )
                    logical.append(len(history))
                start = time.perf_counter()
                if fork_from is not None:
                    llm.fork_session(
                        fork_from,
                        ids[0],
                        version=versions[fork_from],
                        num_tokens=len(source),
                    )
                fork_seconds = time.perf_counter() - start if fork_from else 0.0
                result = llm.generate(prompts, params, use_tqdm=False)
                elapsed = time.perf_counter() - start
                row = {
                    "seconds": elapsed,
                    "logical_prompt_tokens": sum(logical),
                    "cached_tokens": 0,
                    "turns": len(ids),
                    "output_tokens": 16 * len(ids),
                    "hit_turns": 0,
                    "fork_seconds": fork_seconds,
                }
                for sid, output in zip(ids, result):
                    assert output.outputs[0].token_ids == completion
                    histories[sid].extend(completion)
                    versions[sid] = versions.get(sid, 0) + 1
                    row["cached_tokens"] += output.num_cached_tokens
                    row["hit_turns"] += int(output.num_cached_tokens > 0)
                return row

            ids = [f"{scenario}-{repeat}-{i}" for i in range(4)]
            rows = []
            if scenario == "serial_continue":
                run(ids[:1], [source])
                snap(True)
                for _ in range(3):
                    rows.append(run(ids[:1], [source[:64]]))
            elif scenario == "shared_prefix":
                snap(True)
                for i, sid in enumerate(ids):
                    delta = [100 + i] * 64
                    if args.shared_root and i:
                        histories[sid] = source.copy()
                        rows.append(run([sid], [delta], fork_from=ids[0]))
                    else:
                        rows.append(run([sid], [source + delta]))
            elif scenario == "batch_continue":
                for i, sid in enumerate(ids):
                    run([sid], [[100 + i] + source[1:]])
                snap(True)
                for _ in range(3):
                    rows.append(run(ids, [source[:64]] * 4))
            else:
                ids = ids[:2]
                for i, sid in enumerate(ids):
                    run([sid], [[100 + i] + source[1:]])
                snap(True)
                for _ in range(2):
                    for sid in ids:
                        rows.append(run([sid], [source[:64]]))
            metrics = snap()
            if repeat >= 0:
                totals = {key: sum(row[key] for row in rows) for key in rows[0]}
                totals.update(
                    {
                        "token_reuse_rate": totals["cached_tokens"]
                        / totals["logical_prompt_tokens"],
                        "request_hit_rate": totals["hit_turns"] / totals["turns"],
                        "turns_per_second": totals["turns"] / totals["seconds"],
                        "output_tokens_per_second": totals["output_tokens"]
                        / totals["seconds"],
                        "recomputed_prompt_tokens": totals["logical_prompt_tokens"]
                        - totals["cached_tokens"],
                    }
                )
                record = {
                    "scenario": scenario,
                    "repeat": repeat,
                    "totals": totals,
                    "rows": rows,
                    "engine": metrics,
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            "scenario": scenario,
                            "repeat": repeat,
                            **totals,
                            "block_evictions": metrics["block_evictions"],
                            "session_evictions": metrics["session_evictions"],
                            "session_trims": metrics["session_trims"],
                            "session_evicted_blocks": metrics["session_evicted_blocks"],
                        }
                    ),
                    flush=True,
                )
            if args.mode != "apc":
                for sid in histories:
                    assert llm.close_session(sid)
    Path(args.output).write_text(
        json.dumps(
            {
                "config": vars(args),
                "kv_cache_mib": budget,
                "note": (
                    "Small fixed-work diagnostic, eager and sync, no SD/Mooncake. "
                    "No cProfile/torch profiler; lightweight counters enabled. "
                    "Excludes setup turns except shared-prefix cold fanout. "
                    "Two measured trajectories after one warmup, "
                    "not confidence intervals."
                ),
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
