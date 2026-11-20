# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only session continuation diagnostic; requires test dependencies.

Measures Request construction and add/schedule/update separately. GPU output is
synthetic. This excludes frontend tokenization, IPC, workers, and model execution;
it is not an isolated hash-match benchmark or a serving latency measurement.

Run from the repository with PYTHONPATH=. and VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
No model positions are executed; extended lengths only size scheduler metadata.
"""

import argparse
import gc
import json
import statistics
import time

from tests.v1.core.utils import create_scheduler
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.session_scheduler import SessionScheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default="/tmp/session-scheduler-results.json")
    args = parser.parse_args()
    init_none_hash(sha256)
    results = []
    for context in (16384, 65536, 131072):
        for mode in ("apc", "session"):
            prep_times, schedule_times = [], []
            run_medians = []
            for _repeat in range(3):
                scheduler = create_scheduler(
                    model="Qwen/Qwen3-0.6B",
                    enable_prefix_caching=mode == "apc",
                    scheduler_cls=SessionScheduler if mode == "session" else None,
                    max_model_len=context + 4096,
                    max_num_batched_tokens=context + 4096,
                    num_blocks=(context + 4096) // 16 * 2 + 1,
                )
                hasher = get_request_block_hasher(16, sha256) if mode == "apc" else None
                history = [7] * context
                run_times = []
                for turn in range(21):
                    delta = history if turn == 0 else [8] * 64
                    if turn:
                        history.extend(delta)
                    params = SamplingParams(
                        max_tokens=1,
                        ignore_eos=True,
                        extra_args={"session_cache": {"id": "agent", "version": turn}}
                        if mode == "session"
                        else None,
                    )
                    start = time.perf_counter()
                    request = Request(
                        str(turn),
                        history if mode == "apc" else delta,
                        params,
                        None,
                        block_hasher=hasher,
                    )
                    prepared = time.perf_counter()
                    scheduler.add_request(request)
                    output = scheduler.schedule()
                    assert len(output.num_scheduled_tokens) == 1
                    scheduler.update_from_output(
                        output,
                        ModelRunnerOutput(req_ids=[str(turn)], sampled_token_ids=[[9]]),
                    )
                    end = time.perf_counter()
                    if turn:
                        prep_times.append((prepared - start) * 1000)
                        schedule_times.append((end - prepared) * 1000)
                        run_times.append((end - start) * 1000)
                    history.append(9)
                run_medians.append(statistics.median(run_times))
                if mode == "session":
                    scheduler.close_session("agent")
                del scheduler
                gc.collect()
            result = dict(
                context=context,
                mode=mode,
                request_creation_hash_median_ms=statistics.median(prep_times),
                scheduler_median_ms=statistics.median(schedule_times),
                total_per_run_median_ms=run_medians,
                turns_per_run=20,
                runs=3,
            )
            results.append(result)
            print(json.dumps(result), flush=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
