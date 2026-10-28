# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check real PR budget helpers after changing lengths within captured graphs."""

import argparse
import gc
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import torch
from pipeline_bench import (
    MIB,
    MODELS,
    correctness,
    get_num_sms,
    get_paged_mqa_logits_metadata,
    indexer_module,
    init_workspace_manager,
    make_inputs,
    runner,
)

import vllm


def append(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    fields = (
        "model",
        "rows",
        "candidate",
        "requested_backend",
        "budget_mib",
        "replay_raw_context",
    )
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            record = json.loads(line)
            if record.get("status") == "ok":
                completed.add(tuple(record[field] for field in fields))
    contexts = (32768, 8192, 1536, 131072)
    init_workspace_manager(torch.device("cuda"))
    metadata = dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pipeline_sha256=hashlib.sha256(
            Path(__file__).with_name("pipeline_bench.py").read_bytes()
        ).hexdigest(),
        vllm_root=str(Path(vllm.__file__).parent),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        hostname=socket.gethostname(),
        maximum_allocated_raw_context=131072,
        capture_raw_context=32768,
    )
    for model in MODELS:
        heads, width, compression, _ = MODELS[model]
        for rows in (256, 4096):
            x = make_inputs(rows, heads, width, compression, 131072, "mxfp4", 8, 512)
            offsets = torch.arange(rows, device="cuda", dtype=torch.int32) % 8 - 7

            def stage_lengths(context, x=x, offsets=offsets, compression=compression):
                x["ctx"].copy_(((context + offsets) // compression)[:, None])
                # The production outer metadata builder also updates this bank
                # outside the model graph, including unchanged whole-row calls.
                x["schedule"].copy_(
                    get_paged_mqa_logits_metadata(
                        x["ctx"], 64, get_num_sms(), indices=x["indices"]
                    )
                )
                x["max_valid"] = int(x["ctx"].max())

            for candidate in ("none", "select") if model == "v41-c1" else ("none",):
                for requested_backend in ("auto", "deep_select"):
                    for budget in (512, 1024, 2048):
                        case_key = (model, rows, candidate, requested_backend, budget)
                        if all((*case_key, c) in completed for c in contexts):
                            continue
                        stage_lengths(32768)
                        backend, ranges = indexer_module.varlen_decode_plan(
                            rows, width, requested_backend, budget * MIB
                        )
                        chunks = ranges or [(0, rows)]
                        fn = runner(
                            x,
                            backend,
                            chunks,
                            512,
                            width * compression,
                            candidate,
                            production_helper=(
                                indexer_module._varlen_decode_rows if ranges else None
                            ),
                        )
                        for _ in range(3):
                            fn()
                        torch.accelerator.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            fn()
                        for context in contexts:
                            if (*case_key, context) in completed:
                                continue
                            stage_lengths(context)
                            check = correctness(x, graph.replay, chunks, 512, candidate)
                            invalid = any(
                                check[name]
                                for name in (
                                    "chunked_logit_mismatch_rows",
                                    "nonfinite_valid_logits",
                                    "invalid_index_rows",
                                    "duplicate_index_rows",
                                    "exact_topk_value_mismatch_rows",
                                )
                            ) or any(check["candidate_check"].values())
                            append(
                                args.out,
                                dict(
                                    metadata,
                                    status="error" if invalid else "ok",
                                    model=model,
                                    rows=rows,
                                    width=width,
                                    requested_backend=requested_backend,
                                    actual_backend=backend,
                                    budget_mib=budget,
                                    chunks=chunks,
                                    candidate=candidate,
                                    replay_raw_context=context,
                                    correctness=check,
                                    unix_time=time.time(),
                                ),
                            )
                            if invalid:
                                raise AssertionError(check)
                        del graph, fn
                        gc.collect()
                        torch.accelerator.empty_cache()
            del stage_lengths, x
            gc.collect()
            torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
