# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recheck key GPU effects with randomized interleaved A/B/C trials."""

import argparse
import hashlib
import json
import os
import random
import socket
import statistics
import subprocess
import time
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from pipeline_bench import (
    MODELS,
    correctness,
    distribution,
    init_workspace_manager,
    make_inputs,
    plan,
    runner,
    telemetry,
)

import vllm


def append(path, record):
    with path.open("a") as output:
        output.write(json.dumps(record) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    init_workspace_manager(torch.device("cuda"))
    metadata = dict(
        hostname=socket.gethostname(),
        gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        git_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        vllm_python_root=str(Path(vllm.__file__).parent),
        executable=os.sys.executable,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        argv=os.sys.argv,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pipeline_source_sha256=hashlib.sha256(
            Path(__file__).with_name("pipeline_bench.py").read_bytes()
        ).hexdigest(),
        graph_pool_policy="one persistent private pool per arm; replay serially",
    )
    for model in ("v4", "v41-c1"):
        heads, width, compression, _ = MODELS[model]
        for context in (8192, 32768):
            x = make_inputs(
                args.rows, heads, width, compression, context, "mxfp4", 8, 512
            )
            case = dict(
                model=model,
                rows=args.rows,
                width=width,
                heads=heads,
                raw_context=context,
                q_len=8,
                precision="mxfp4",
                topk=512,
                max_seq_len=width * compression,
            )
            graphs = {}
            checks = {}
            for variant in ("single-auto", "single-ds", "final-512"):
                backend, chunks = plan(args.rows, width, variant)
                fn = runner(x, backend, chunks, 512, width * compression, "none")
                check = correctness(x, fn, chunks, 512, "none")
                assert not any(
                    check[name]
                    for name in (
                        "chunked_logit_mismatch_rows",
                        "nonfinite_valid_logits",
                        "invalid_index_rows",
                        "duplicate_index_rows",
                        "exact_topk_value_mismatch_rows",
                    )
                ), check
                checks[variant] = check
                for _ in range(3):
                    fn()
                torch.accelerator.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()
                graph.replay()
                torch.accelerator.synchronize()
                graphs[variant] = graph
            for round_index in range(2):
                medians = {}
                for trial in range(args.trials):
                    order = [
                        (variant, cache)
                        for variant in graphs
                        for cache in ("warm", "cold")
                    ]
                    random.Random(context + trial + round_index * 101).shuffle(order)
                    for order_index, (variant, cache) in enumerate(order):
                        record = dict(
                            metadata,
                            case=case,
                            variant=variant,
                            round=round_index,
                            trial=trial,
                            order_index=order_index,
                            cache=cache,
                            started=time.time(),
                            telemetry_before=telemetry(),
                            correctness=checks[variant],
                        )
                        samples = bench_gpu_time_with_cupti(
                            graphs[variant].replay,
                            dry_run_iters=5,
                            repeat_iters=100,
                            use_cuda_graph=False,
                            cold_l2_cache=cache == "cold",
                        )
                        timing = distribution([s * 1000 for s in samples])
                        medians.setdefault((variant, cache), []).append(
                            timing["median_us"]
                        )
                        record.update(
                            status="ok",
                            record_kind="trial",
                            timing=timing,
                            telemetry_after=telemetry(),
                        )
                        append(args.out, record)
                summary = {}
                for (variant, cache), values in medians.items():
                    summary[f"{variant}/{cache}"] = dict(
                        median_us=statistics.median(values),
                        medians_us=values,
                        cv=statistics.pstdev(values) / statistics.mean(values),
                    )
                noisy = any(v["cv"] > 0.03 for v in summary.values())
                append(
                    args.out,
                    dict(
                        metadata,
                        case=case,
                        round=round_index,
                        record_kind="summary",
                        summary=summary,
                        status="noisy" if noisy else "ok",
                    ),
                )
                print(case, round_index, "noisy" if noisy else "ok", flush=True)
                if not noisy:
                    break
            del graph, graphs, fn, x
            torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
