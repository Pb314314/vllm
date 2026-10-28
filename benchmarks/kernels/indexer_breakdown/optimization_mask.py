# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-only BF16 mask ablation with GPU-bounded valid-prefix stores."""

import hashlib
import json
import os
import random
import statistics
import subprocess
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from optimization_schedule import save
from pipeline_bench import (
    distribution,
    get_indexer_topk,
    get_num_sms,
    get_paged_mqa_logits_metadata,
    init_workspace_manager,
    logits_for_rows,
    make_inputs,
    profile_kernels,
    runner,
    telemetry,
)

from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    _candidate_flags_kernel,
)
from vllm.triton_utils import tl, triton


@triton.jit
def prefix_mask(logits, ends, flags, stride: tl.constexpr, width: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    end = tl.minimum(tl.load(ends + row), width)
    offsets = tl.arange(0, 256)
    for tile in range(tl.program_id(1), tl.cdiv(end, 256), tl.num_programs(1)):
        cols = tile * 256 + offsets
        keep = tl.load(flags + row * (width // 8 + 1) + cols // 8, cols < end, other=0)
        tl.store(
            logits + row * stride + cols, -float("inf"), (cols < end) & (keep == 0)
        )


def main():
    path = Path(os.environ["OPTIMIZATION_RESULTS"])
    init_workspace_manager(torch.device("cuda"))
    rows, width = 1024, 1048576
    programs = int(os.environ.get("MASK_PROGRAMS", "32"))
    x = make_inputs(rows, 32, width, 1, 32768, "mxfp4", 8, 512)
    chunks = [(i, i + 256) for i in range(0, rows, 256)]
    base = runner(x, "deep_select", chunks, 512, width, "mask")
    candidates = torch.arange(2048, device="cuda", dtype=torch.int32) * 4
    base.candidates.copy_(candidates[None].expand(rows, -1))

    def candidate():
        for start, end in chunks:
            schedule = get_paged_mqa_logits_metadata(
                x["ctx"][start:end],
                64,
                get_num_sms(),
                indices=x["indices"][start:end],
            )
            logits = logits_for_rows(x, start, end, schedule)
            flags = torch.empty(
                (end - start, width // 8 + 1), device="cuda", dtype=torch.uint8
            )
            _candidate_flags_kernel[(end - start,)](
                base.candidates[start:end],
                None,
                flags,
                *base.candidates.stride(),
                0,
                width,
                width // 8,
                8,
                2048,
                False,
                1,
            )
            prefix_mask[(end - start, programs)](
                logits, x["ctx"][start:end], flags, logits.stride(0), width
            )
            get_indexer_topk("deep_select")(
                logits, x["ctx"][start:end], 1, x["out"][start:end], 512, width
            )

    metadata = dict(
        git_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        rows=rows,
        capacity=width,
        topk=512,
        heads=32,
        precision="mxfp4",
        chunk_rows=256,
        mask_programs_per_row=programs,
        host_max_seq_len=width,
        scope="decode start=0, BF16 DeepSelect end-aware; flags unchanged",
    )
    graphs = {}
    for name, fn in (("full_capacity_mask", base), ("gpu_prefix_mask", candidate)):
        for _ in range(3):
            fn()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs[name] = graph
        save(
            path,
            dict(
                metadata,
                record_kind="kernel_profile",
                arm=name,
                kernels=profile_kernels(fn, 3),
                status="ok",
            ),
        )
    for raw_context in (32768, 8192):
        lens = torch.tensor(
            [raw_context - 7 + i % 8 for i in range(rows)],
            device="cuda",
            dtype=torch.int32,
        )[:, None]
        x["ctx"].copy_(lens)
        x["schedule"] = get_paged_mqa_logits_metadata(
            x["ctx"], 64, get_num_sms(), indices=x["indices"]
        )
        full = logits_for_rows(x, 0, rows, x["schedule"])
        scores = full[:, :32768].clone()
        del full
        cols = torch.arange(32768, device="cuda")[None]
        scores.masked_fill_((cols >= lens) | ((cols // 8) % 4 != 0), -float("inf"))
        reference = scores.float().topk(512, dim=-1).values
        checks = {}
        for name, graph in graphs.items():
            graph.replay()
            selected = x["out"].long()
            values = scores.gather(1, selected.clamp(0, 32767)).float()
            check = dict(
                value_mismatch_rows=int(
                    (values.sort(1, descending=True).values != reference).any(1).sum()
                ),
                invalid_rows=int(((selected < 0) | (selected >= lens)).any(1).sum()),
                duplicate_rows=int(
                    (selected.sort(1).values.diff(dim=1) == 0).any(1).sum()
                ),
            )
            assert not any(check.values()), check
            checks[name] = check
        samples = {}
        for trial in range(3):
            order = [(name, cache) for name in graphs for cache in ("warm", "cold")]
            random.Random(53 + trial).shuffle(order)
            for name, cache in order:
                values = bench_gpu_time_with_cupti(
                    graphs[name].replay,
                    dry_run_iters=5,
                    repeat_iters=100,
                    use_cuda_graph=False,
                    cold_l2_cache=cache == "cold",
                )
                timing = distribution([v * 1000 for v in values])
                samples.setdefault((name, cache), []).append(timing["median_us"])
                save(
                    path,
                    dict(
                        metadata,
                        record_kind="trial",
                        arm=name,
                        raw_context=raw_context,
                        cache=cache,
                        trial=trial,
                        timing=timing,
                        correctness=checks[name],
                        telemetry_after=telemetry(),
                        status="ok",
                    ),
                )
        result = {
            f"{name}/{cache}": dict(
                median_us=statistics.median(values),
                medians_us=values,
                cv=statistics.pstdev(values) / statistics.mean(values),
            )
            for (name, cache), values in samples.items()
        }
        save(
            path,
            dict(
                metadata,
                record_kind="summary",
                raw_context=raw_context,
                timings=result,
                status="ok",
            ),
        )
        print(raw_context, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
