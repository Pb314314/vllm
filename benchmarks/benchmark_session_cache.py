# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-work single-session replay; reports turn latency, not streaming TTFT.

Run APC and session modes in separate processes with identical arguments.
Outputs are forced to the same token sequence to hold future contexts constant.
This diagnostic does not model concurrent sessions or real tool timing.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("apc", "session"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--delta-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kv-cache-mib", type=int, default=1024)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.context_len, args.delta_len, args.output_len, args.turns, args.repeats)
        <= 0
    ):
        parser.error("Lengths, turns and repeats must be positive")
    max_len = args.context_len + args.turns * (args.delta_len + args.output_len)
    llm = LLM(
        model=args.model,
        enable_prefix_caching=args.mode == "apc",
        scheduler_cls=(
            "vllm.v1.core.sched.session_scheduler.SessionScheduler"
            if args.mode == "session"
            else None
        ),
        async_scheduling=False,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_model_len=max_len,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        kv_cache_memory_bytes=args.kv_cache_mib * 1024 * 1024,
        gpu_memory_utilization=0.04,
        block_size=16,
        seed=0,
        enable_trace_replay=True,
        attention_config={"backend": args.attention_backend},
    )
    tokenizer = llm.get_tokenizer()
    source = tokenizer.encode(
        "Inspect the repository, run the tests, and summarize the results. " * max_len,
        add_special_tokens=False,
    )
    initial = source[: args.context_len]
    delta = source[: args.delta_len]
    completion = source[: args.output_len]
    # Warm the same prefill and decode shapes; exclude this trajectory.
    records = []
    for repeat in range(-1, args.repeats):
        assert llm.reset_prefix_cache()
        history: list[int] = []
        for turn in range(args.turns):
            new_tokens = initial if turn == 0 else delta
            history.extend(new_tokens)
            extra = (
                {"session_cache": {"id": f"replay-{repeat}", "version": turn}}
                if args.mode == "session"
                else None
            )
            params = SamplingParams(
                temperature=0,
                max_tokens=args.output_len,
                ignore_eos=True,
                trace_decode_token_ids=completion,
                extra_args=extra,
            )
            start = time.perf_counter()
            result = llm.generate(
                {"prompt_token_ids": new_tokens if args.mode == "session" else history},
                params,
                use_tqdm=False,
            )[0]
            elapsed = time.perf_counter() - start
            assert result.outputs[0].token_ids == completion
            if repeat >= 0:
                records.append(
                    dict(
                        repeat=repeat,
                        turn=turn,
                        context_tokens=len(history),
                        cached_tokens=result.num_cached_tokens,
                        elapsed_ms=elapsed * 1000,
                    )
                )
            history.extend(completion)
        if args.mode == "session":
            assert llm.close_session(f"replay-{repeat}")
    continuations = [r["elapsed_ms"] for r in records if r["turn"] > 0]
    report = {
        "config": {**vars(args), "output_json": str(args.output_json)},
        "metric": "offline LLM.generate wall time including all output tokens",
        "continuation_median_ms": (
            statistics.median(continuations) if continuations else None
        ),
        "turns": records,
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "turns"}, indent=2))


if __name__ == "__main__":
    main()
