# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure dependent Chat turns using the standard serving benchmark client.

Run against a separately started server. Repeat with the turn-prefix flag off
and on, keeping APC, GPU capacity, and all other server settings unchanged.
The output includes full request histories for an identical-input replay.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import aiohttp
import numpy as np
import psutil
import requests
from prometheus_client.parser import text_string_to_metric_families

from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    async_request_openai_chat_completions,
)


def metrics(url):
    response = requests.get(f"{url}/metrics", timeout=30)
    response.raise_for_status()
    result = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            if sample.name.startswith(
                (
                    "vllm:prefix_cache_",
                    "vllm:turn_prefix_cache_",
                    "vllm:request_success_",
                )
            ) and not sample.name.endswith("_created"):
                result[sample.name] = result.get(sample.name, 0) + sample.value
    return result


async def workload(args, replay=None):
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as client:

        async def conversation(index):
            history = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Work item {index}. "
                    + " hello" * args.context
                    + "\nWrite a list of animals.",
                },
            ]
            rows, inputs = [], []
            for turn in range(args.turns):
                messages = replay[index][turn] if replay else copy.deepcopy(history)
                inputs.append(messages)
                output = await async_request_openai_chat_completions(
                    RequestFuncInput(
                        prompt="",
                        chat_messages=messages,
                        api_url=f"{args.url}/v1/chat/completions",
                        prompt_len=0,
                        output_len=args.output_tokens,
                        model=args.model,
                        ignore_eos=True,
                        extra_body={"temperature": 0},
                    ),
                    client,
                )
                assert output.success, output.error
                rows.append({"conversation": index, "turn": turn, **asdict(output)})
                history += [
                    {"role": "assistant", "content": output.generated_text},
                    {"role": "user", "content": "Continue the list with more animals."},
                ]
            return rows, inputs

        return await asyncio.gather(*(conversation(i) for i in range(args.sessions)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay = (
        json.loads(args.replay.read_text())["rounds"][0]["inputs"]
        if args.replay
        else None
    )
    if replay:
        assert len(replay) == args.sessions
        assert all(len(turns) == args.turns for turns in replay)

    def cpu_seconds():
        if args.server_pid is None:
            return 0.0
        root = psutil.Process(args.server_pid)
        return sum(
            p.cpu_times().user + p.cpu_times().system
            for p in [root, *root.children(recursive=True)]
        )

    # Warm model kernels, tokenization, and the HTTP connection path. The first
    # round is excluded from measurements but still drains scheduler metrics.
    records = []
    for run in range(args.rounds + 1):
        reset = requests.post(f"{args.url}/reset_prefix_cache", timeout=30)
        reset.raise_for_status()
        before = metrics(args.url)
        cpu_before = cpu_seconds()
        start = time.perf_counter()
        results = asyncio.run(workload(args, replay))
        duration = time.perf_counter() - start
        cpu_duration = cpu_seconds() - cpu_before
        rows = [row for conversation, _ in results for row in conversation]
        # Scheduler metrics are delivered periodically; wait for this run's
        # successful requests, excluding this polling time from throughput.
        deadline = time.monotonic() + 30
        while True:
            after = metrics(args.url)
            completed = after.get("vllm:request_success_total", 0)
            if completed >= before.get("vllm:request_success_total", 0) + len(rows):
                break
            assert time.monotonic() < deadline, after
            time.sleep(0.05)
        record = {
            "duration_s": duration,
            "server_cpu_s": cpu_duration,
            "requests_per_s": len(rows) / duration,
            "ttft_ms_p50_p95_p99": np.percentile(
                [row["ttft"] * 1000 for row in rows], [50, 95, 99]
            ).tolist(),
            "metrics_before": before,
            "metrics_after": after,
            "requests": rows,
            "inputs": [inputs for _, inputs in results],
        }
        record["inputs_sha256"] = hashlib.sha256(
            json.dumps(record["inputs"], sort_keys=True).encode()
        ).hexdigest()
        if run == 0:
            continue
        records.append(record)
        print(
            json.dumps(
                {k: v for k, v in record.items() if k not in ("requests", "inputs")}
            ),
            flush=True,
        )
    args.output.write_text(
        json.dumps({"config": vars(args), "rounds": records}, default=str, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
