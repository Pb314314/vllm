# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate continuation admission, steady decode, and turn completion CPU cost.

Uses real synchronous schedulers and fake GPU outputs. No worker or model forward.
Run from the repository with PYTHONPATH=. and test dependencies installed.
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

from tests.v1.core.utils import create_scheduler
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.session_scheduler import SessionScheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request


def run(mode, batch, context, warmup, steps):
    max_len = context + 64 + 16 + warmup + steps + 64
    scheduler = create_scheduler(
        model="Qwen/Qwen3-0.6B",
        max_num_seqs=batch,
        max_model_len=max_len,
        max_num_batched_tokens=batch * max_len,
        num_blocks=batch * ((max_len + 15) // 16) + 32,
        enable_prefix_caching=mode == "apc",
        scheduler_cls=SessionScheduler if mode == "session" else None,
    )
    hasher = get_request_block_hasher(16, sha256) if mode == "apc" else None
    histories = [[100 + i] + [7] * (context - 1) for i in range(batch)]

    def request(i, version, max_tokens):
        tokens = histories[i] if version == 0 or mode == "apc" else [8] * 64
        return Request(
            f"{version}-{i}",
            tokens,
            SamplingParams(
                max_tokens=max_tokens,
                ignore_eos=True,
                extra_args={"session_cache": {"id": str(i), "version": version}}
                if mode == "session"
                else None,
            ),
            None,
            block_hasher=hasher,
        )

    for i in range(batch):
        scheduler.add_request(request(i, 0, 16))
    for _ in range(16):
        output = scheduler.schedule()
        ids = list(output.num_scheduled_tokens)
        assert len(ids) == batch
        scheduler.update_from_output(
            output, ModelRunnerOutput(req_ids=ids, sampled_token_ids=[[9] for _ in ids])
        )
    assert scheduler.get_num_unfinished_requests() == 0
    for history in histories:
        history.extend([9] * 16 + [8] * 64)

    start = time.perf_counter_ns()
    requests = [request(i, 1, warmup + steps + 2) for i in range(batch)]
    constructed = time.perf_counter_ns()
    for req in requests:
        scheduler.add_request(req)
    admitted = time.perf_counter_ns()
    output = scheduler.schedule()
    scheduled = time.perf_counter_ns()
    ids = list(output.num_scheduled_tokens)
    assert len(ids) == batch
    prefill_tokens = sum(output.num_scheduled_tokens.values())
    gpu = ModelRunnerOutput(req_ids=ids, sampled_token_ids=[[9] for _ in ids])
    update_start = time.perf_counter_ns()
    scheduler.update_from_output(output, gpu)
    updated = time.perf_counter_ns()
    admission = {
        "construct_us": (constructed - start) / 1000,
        "add_us": (admitted - constructed) / 1000,
        "schedule_us": (scheduled - admitted) / 1000,
        "update_us": (updated - update_start) / 1000,
        "scheduled_tokens": prefill_tokens,
    }
    samples = []
    for step in range(warmup + steps + 1):
        before = requests[0].num_computed_tokens
        sample_start = time.perf_counter_ns()
        output = scheduler.schedule()
        schedule_end = time.perf_counter_ns()
        assert len(output.num_scheduled_tokens) == batch
        assert sum(output.num_scheduled_tokens.values()) == batch
        gpu = ModelRunnerOutput(req_ids=ids, sampled_token_ids=[[9] for _ in ids])
        update_start = time.perf_counter_ns()
        scheduler.update_from_output(output, gpu)
        update_end = time.perf_counter_ns()
        sample = {
            "schedule_us": (schedule_end - sample_start) / 1000,
            "update_us": (update_end - update_start) / 1000,
            "allocate_boundary": before % 16 == 0,
            "hash_boundary": (before + 2) % 16 == 0,
        }
        sample["total_us"] = sample["schedule_us"] + sample["update_us"]
        if warmup <= step < warmup + steps:
            samples.append(sample)
        elif step == warmup + steps:
            completion = sample
    assert scheduler.get_num_unfinished_requests() == 0
    if mode == "session":
        assert scheduler.kv_cache_manager.num_session_evictions == 0
        for i in range(batch):
            assert scheduler.close_session(str(i))
    median = {
        key: statistics.median(s[key] for s in samples)
        for key in ["schedule_us", "update_us", "total_us"]
    }
    ordinary = [
        s for s in samples if not s["allocate_boundary"] and not s["hash_boundary"]
    ]
    boundary = [s for s in samples if s["allocate_boundary"] or s["hash_boundary"]]
    return {
        "mode": mode,
        "batch": batch,
        "context": context,
        "admission": admission,
        "decode_median": median,
        "decode_mean_total_us": statistics.mean(s["total_us"] for s in samples),
        "ordinary_median_total_us": statistics.median(s["total_us"] for s in ordinary),
        "boundary_median_total_us": statistics.median(s["total_us"] for s in boundary),
        "completion": completion,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    init_none_hash(sha256)
    records = []
    for batch in args.batches:
        for repeat in range(args.rounds):
            modes = ["apc", "session"] if repeat % 2 == 0 else ["session", "apc"]
            for mode in modes:
                gc.collect()
                record = run(mode, batch, args.context, args.warmup, args.steps)
                record["round"] = repeat
                records.append(record)
                print(
                    json.dumps({k: v for k, v in record.items() if k != "samples"}),
                    flush=True,
                )
    args.output.write_text(
        json.dumps(
            {
                "config": {**vars(args), "output": str(args.output)},
                "note": (
                    "Sync, same-process alternated rounds, real scheduler with "
                    "fake GPU outputs. No cProfile, worker, IPC, SD, or Mooncake. "
                    "Homogeneous request positions align block boundaries. "
                    "Not a reproduction of PR63934 async M4 timings."
                ),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
