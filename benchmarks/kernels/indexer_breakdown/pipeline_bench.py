# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real SM100 paged-indexer pipeline and row-chunk ablations.

Run from an installed checkout with a reserved CUDA GPU. Results are appended
after each variant; --resume skips successful keys. Shapes describe per-rank
inputs, not API concurrency. No model or product source is patched.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import socket
import statistics
import subprocess
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

import vllm
import vllm.model_executor.layers.sparse_attn_indexer as indexer_module
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
    select_candidate_blocks,
)
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.model_executor.layers.sparse_attn_indexer import (
    as_fp32_logits,
    kv_cache_as_quant_view,
)
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
)
from vllm.v1.worker.workspace import init_workspace_manager

MIB = 1024**2
MODELS = {
    "v4": (64, 262144, 4, 21),
    "v41-c2": (32, 524288, 2, 3),
    "v41-c1": (32, 1048576, 1, 1),
}


def csv_ints(value):
    return [int(x) for x in value.split(",")]


def telemetry():
    try:
        value = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,clocks.sm,temperature.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        return value.splitlines()
    except Exception as error:
        return {"error": str(error)}


def distribution(values):
    values = sorted(values)
    return {
        "median_us": statistics.median(values),
        "p10_us": values[int((len(values) - 1) * 0.1)],
        "p90_us": values[int((len(values) - 1) * 0.9)],
        "count": len(values),
    }


def make_inputs(rows, heads, width, compression, context, dtype, q_len, topk):
    device, dim, page = "cuda", 128, 64
    requests = math.ceil(rows / q_len)
    per_req_rows = [min(q_len, rows - i * q_len) for i in range(requests)]
    lengths = []
    for count in per_req_rows:
        lengths.extend((context - count + j + 1) // compression for j in range(count))
    if min(lengths) < 0 or max(lengths) > width:
        raise ValueError("Context/query lengths exceed the configured capacity")
    pages_per_req = math.ceil(max(lengths) / page)
    num_pages = requests * pages_per_req
    generator = torch.Generator(device=device).manual_seed(1741)
    if dtype == "mxfp4":
        cache = torch.empty((num_pages, page * 68), device=device, dtype=torch.uint8)
        cache[:, : page * 64] = torch.randint(
            0,
            256,
            (num_pages, page * 64),
            generator=generator,
            device=device,
            dtype=torch.uint8,
        )
        cache[:, page * 64 :] = 127
        cache = cache.view(num_pages, page, 68)
        cache = kv_cache_as_quant_view(cache, dim, True)
        q = torch.randint(
            0,
            256,
            (rows, 1, heads, 64),
            device=device,
            generator=generator,
            dtype=torch.uint8,
        ).view(torch.int8)
    else:
        values = torch.randn(
            (num_pages, page, dim),
            device=device,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(torch.float8_e4m3fn)
        cache = torch.empty((num_pages, page * 132), device=device, dtype=torch.uint8)
        cache[:, : page * 128] = values.view(num_pages, page * dim).view(torch.uint8)
        cache[:, page * 128 :] = 127
        cache = cache.view(num_pages, page, 1, 132)
        del values
        q = torch.randn(
            (rows, 1, heads, dim),
            device=device,
            generator=generator,
            dtype=torch.bfloat16,
        ).to(torch.float8_e4m3fn)
    scales = torch.full((rows, 1, heads), 0x7F7F7F7F, device=device, dtype=torch.int32)
    weights = (torch.randn((rows, heads), device=device, generator=generator) * 0.1).to(
        torch.bfloat16
    )
    request_table = (
        torch.randperm(num_pages, device=device, generator=generator)
        .to(torch.int32)
        .view(requests, pages_per_req)
    )
    indices = torch.arange(requests, device=device, dtype=torch.int32)
    counts = torch.tensor(per_req_rows, device=device)
    indices = indices.repeat_interleave(counts)
    table = request_table.repeat_interleave(counts, dim=0).contiguous()
    seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)[:, None]
    schedule = get_paged_mqa_logits_metadata(
        seq_lens, page, get_num_sms(), indices=indices
    )
    return dict(
        q=q,
        qs=scales,
        weights=weights,
        kv=cache,
        ctx=seq_lens,
        bt=table,
        indices=indices,
        schedule=schedule,
        out=torch.full((rows, topk), -1, device=device, dtype=torch.int32),
        width=width,
        max_valid=max(lengths),
        valid_scores=sum(lengths),
    )


def plan(rows, width, name):
    if name == "single-auto":
        return "auto", [(0, rows)]
    if name == "single-ds":
        return "deep_select", [(0, rows)]
    if name == "single-persistent":
        return "persistent", [(0, rows)]
    family, budget = name.rsplit("-", 1)
    budget = int(budget) * MIB
    backend = {
        "ds": "deep_select",
        "prototype": "deep_select",
        "auto": "auto",
        "final": "auto",
        "persistent": "persistent",
    }[family]
    if family == "final":
        if rows * width * 6 <= budget:
            return "auto", [(0, rows)]
        backend = "deep_select"
    row_bytes = width * (2 if backend == "deep_select" else 6)
    count = max(16, budget // row_bytes // 16 * 16)
    return backend, [(i, min(i + count, rows)) for i in range(0, rows, count)]


def logits_for_rows(x, start, end, schedule):
    return fp8_fp4_paged_mqa_logits(
        (x["q"][start:end], x["qs"][start:end]),
        x["kv"],
        x["weights"][start:end],
        x["ctx"][start:end],
        x["bt"][start:end],
        schedule,
        max_model_len=x["width"],
        indices=x["indices"][start:end],
    )


def runner(
    x,
    backend,
    chunks,
    topk,
    host_max,
    candidate,
    prebuilt=False,
    production_helper=None,
):
    metadata = None
    if prebuilt:
        metadata = [
            get_paged_mqa_logits_metadata(
                x["ctx"][s:e],
                x["kv"].shape[1],
                get_num_sms(),
                indices=x["indices"][s:e],
            )
            for s, e in chunks
        ]
    if candidate != "none":
        number = 2048
        candidates = torch.arange(number, device="cuda", dtype=torch.int32)
        candidates = candidates[None].expand(x["q"].shape[0], -1).contiguous()
        candidates.masked_fill_(
            candidates >= torch.div(x["ctx"] + 7, 8, rounding_mode="floor"), -1
        )
    else:
        candidates = None
    decode = SimpleNamespace(
        seq_lens=x["ctx"],
        indices=x["indices"],
        block_table=x["bt"],
        schedule_metadata=x["schedule"],
    )

    def run():
        if production_helper is not None:
            for start, end in chunks:
                production_helper(
                    x["q"],
                    x["qs"],
                    x["weights"],
                    x["kv"],
                    decode,
                    x["width"],
                    host_max,
                    backend,
                    topk,
                    x["out"],
                    candidates,
                    8,
                    candidate == "select",
                    start,
                    end,
                )
            return
        for index, (start, end) in enumerate(chunks):
            ctx = x["ctx"][start:end]
            if metadata is not None:
                schedule = metadata[index]
            elif len(chunks) == 1:
                schedule = x["schedule"]
            else:
                schedule = get_paged_mqa_logits_metadata(
                    ctx,
                    x["kv"].shape[1],
                    get_num_sms(),
                    indices=x["indices"][start:end],
                )
            logits = logits_for_rows(x, start, end, schedule)
            if backend != "deep_select":
                logits = as_fp32_logits(logits, ctx.view(-1))
            if candidate == "mask":
                apply_candidate_mask(
                    logits, None, ctx.view(-1), candidates[start:end], 8, 1
                )
            elif candidate == "select":
                select_candidate_blocks(
                    logits,
                    None,
                    ctx.view(-1),
                    candidates.shape[1],
                    8,
                    candidates[start:end],
                    1,
                )
            get_indexer_topk(backend)(
                logits, ctx, 1, x["out"][start:end, :topk], topk, host_max
            )
            del logits

    run.candidates = candidates
    return run


def correctness(x, run, chunks, topk, candidate):
    full = logits_for_rows(x, 0, x["q"].shape[0], x["schedule"])
    logits_stride = list(full.stride())
    valid = full[:, : x["max_valid"]].clone()
    del full
    bad_logits = 0
    for start, end in chunks:
        metadata = get_paged_mqa_logits_metadata(
            x["ctx"][start:end],
            x["kv"].shape[1],
            get_num_sms(),
            indices=x["indices"][start:end],
        )
        chunk = logits_for_rows(x, start, end, metadata)
        mask = torch.arange(x["max_valid"], device="cuda")[None] < x["ctx"][start:end]
        bad_logits += int(
            ((chunk[:, : x["max_valid"]] != valid[start:end]) & mask).any(1).sum()
        )
        del chunk, mask
    run()
    selected = x["out"].long()
    valid_count = x["ctx"].view(-1).clamp(max=topk)
    in_range = (selected >= 0) & (selected < x["ctx"])
    bad_indices = int((in_range.sum(1) != valid_count).sum())
    sorted_indices = selected.sort(1).values
    duplicates = int(
        (
            (sorted_indices[:, 1:] == sorted_indices[:, :-1])
            & (sorted_indices[:, 1:] >= 0)
        )
        .any(1)
        .sum()
    )
    cols = torch.arange(x["max_valid"], device="cuda")[None]
    nonfinite = int((~valid.isfinite() & (cols < x["ctx"])).sum())
    valid.masked_fill_(cols >= x["ctx"], -float("inf"))
    if candidate == "mask":
        valid.masked_fill_(
            cols >= min(2048, math.ceil(x["max_valid"] / 8)) * 8, -float("inf")
        )
    padded = torch.nn.functional.pad(
        valid.float(), (0, max(0, topk - valid.shape[1])), value=-float("inf")
    )
    reference, ref_indices = padded.topk(topk, dim=1)
    values = valid.gather(1, selected.clamp(0, x["max_valid"] - 1))
    values.masked_fill_(selected < 0, -float("inf"))
    bad_values = int(
        (values.float().sort(1, descending=True).values != reference).any(1).sum()
    )
    ref_indices.masked_fill_(reference == -float("inf"), -1)
    ref_indices = ref_indices.sort(1).values
    bad_sets = int((sorted_indices != ref_indices).any(1).sum())
    candidate_check = {}
    if candidate == "select":
        blocks = run.candidates.long()
        lengths = torch.div(x["ctx"] + 7, 8, rounding_mode="floor")
        block_scores = (
            torch.nn.functional.pad(
                valid.float(), (0, (-valid.shape[1]) % 8), value=-float("inf")
            )
            .view(valid.shape[0], -1, 8)
            .amax(-1)
        )
        newest = (lengths - 1).clamp(min=0)
        block_scores.scatter_(1, newest.long(), float("inf"))
        selected_scores = block_scores.gather(
            1, blocks.clamp(0, block_scores.shape[1] - 1)
        )
        selected_scores.masked_fill_(blocks < 0, -float("inf"))
        expected_scores = (
            torch.nn.functional.pad(
                block_scores,
                (0, max(0, 2048 - block_scores.shape[1])),
                value=-float("inf"),
            )
            .topk(2048, dim=1)
            .values
        )
        sorted_blocks = blocks.sort(1).values
        candidate_check = dict(
            invalid_rows=int(
                (
                    ((blocks >= 0) & (blocks < lengths)).sum(1)
                    != lengths.view(-1).clamp(max=2048)
                ).sum()
            ),
            duplicate_rows=int(
                (
                    (sorted_blocks[:, 1:] == sorted_blocks[:, :-1])
                    & (sorted_blocks[:, 1:] >= 0)
                )
                .any(1)
                .sum()
            ),
            selected_block_value_mismatch_rows=int(
                (selected_scores.sort(1, descending=True).values != expected_scores)
                .any(1)
                .sum()
            ),
        )
    return dict(
        chunked_logit_mismatch_rows=bad_logits,
        nonfinite_valid_logits=nonfinite,
        invalid_index_rows=bad_indices,
        duplicate_index_rows=duplicates,
        exact_topk_value_mismatch_rows=bad_values,
        exact_topk_index_set_mismatch_rows=bad_sets,
        rows=selected.shape[0],
        reference="torch.topk on one unchunked BF16 call",
        bf16_output_stride=logits_stride,
        candidate_check=candidate_check,
    )


def profile_kernels(fn, count):
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.accelerator.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(count):
            graph.replay()
        torch.accelerator.synchronize()
    kernels = defaultdict(lambda: {"count": 0, "total_us": 0.0})
    for event in prof.events():
        if "CUDA" in str(event.device_type):
            kernels[event.name]["count"] += 1
            kernels[event.name]["total_us"] += event.device_time_total
    return [
        dict(
            name=name,
            count_per_call=entry["count"] / count,
            sum_us_per_call=entry["total_us"] / count,
        )
        for name, entry in kernels.items()
    ]


def memory(fn):
    gc.collect()
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    before = torch.accelerator.memory_allocated()
    before_reserved = torch.accelerator.memory_reserved()
    torch.accelerator.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool):
        fn()
    torch.accelerator.synchronize()
    result = dict(
        baseline_allocated_bytes=before,
        baseline_reserved_bytes=before_reserved,
        peak_extra_bytes=torch.accelerator.max_memory_allocated() - before,
        live_extra_bytes=torch.accelerator.memory_allocated() - before,
        reserved_extra_bytes=torch.accelerator.memory_reserved() - before_reserved,
    )
    del graph, pool
    gc.collect()
    torch.accelerator.empty_cache()
    return result


def launch_details(fn, repeats):
    source = Path(__file__).parent.parent / "benchmark_indexer_decode_topk_breakdown.py"
    spec = importlib.util.spec_from_file_location("topk_details", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.accelerator.synchronize()
    result = module.cupti_kernel_details(graph.replay, repeats, None)
    result["helper_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result


def benchmark(fn, args):
    timings = {}
    for cold in args.cache.split(","):
        attempts = []
        for _attempt in range(3):
            trials = []
            for _ in range(args.trials):
                samples = bench_gpu_time_with_cupti(
                    fn,
                    dry_run_iters=5,
                    repeat_iters=args.repeat,
                    use_cuda_graph=True,
                    cold_l2_cache=cold == "cold",
                )
                trials.append(distribution([v * 1000 for v in samples]))
            medians = [t["median_us"] for t in trials]
            cv = statistics.pstdev(medians) / statistics.mean(medians)
            attempts.append(dict(trials=trials, trial_cv=cv))
            if cv <= 0.03:
                break
        timings[cold] = dict(
            trials=trials,
            median_us=statistics.median(medians),
            trial_cv=cv,
            retry_attempts=attempts,
            noisy=cv > 0.03,
        )
    if args.eager:
        for _ in range(args.eager_warmup):
            fn()
        torch.accelerator.synchronize()
        host, wall = [], []
        for _ in range(args.eager_reps):
            torch.accelerator.synchronize()
            t0 = time.perf_counter_ns()
            for _ in range(args.eager_layers):
                fn()
            t1 = time.perf_counter_ns()
            torch.accelerator.synchronize()
            t2 = time.perf_counter_ns()
            host.append((t1 - t0) / 1000 / args.eager_layers)
            wall.append((t2 - t0) / 1000 / args.eager_layers)
        timings["eager"] = dict(
            host=distribution(host),
            wall=distribution(wall),
            consecutive_layers=args.eager_layers,
            warmup_calls=args.eager_warmup,
        )
    return timings


def stage_timings(x, backend, chunks, topk, host_max, args):
    """Time isolated stages for each distinct chunk size, outside the pipeline."""
    result = []
    seen = set()
    for start, end in chunks:
        if end - start in seen:
            continue
        seen.add(end - start)
        ctx = x["ctx"][start:end]

        def schedule_fn(ctx=ctx, start=start, end=end):
            return get_paged_mqa_logits_metadata(
                ctx,
                x["kv"].shape[1],
                get_num_sms(),
                indices=x["indices"][start:end],
            )

        schedule = schedule_fn()

        def logits_fn(start=start, end=end, schedule=schedule):
            return logits_for_rows(x, start, end, schedule)

        logits = logits_fn()
        topk_logits = (
            logits if backend == "deep_select" else as_fp32_logits(logits, ctx.view(-1))
        )
        fns = {"schedule": schedule_fn, "logits": logits_fn}
        if backend != "deep_select":
            fns["upcast"] = lambda logits=logits, ctx=ctx: as_fp32_logits(
                logits, ctx.view(-1)
            )
        if args.candidate != "none":
            number = 2048
            candidates = torch.arange(number, device="cuda", dtype=torch.int32)
            candidates = candidates[None].expand(end - start, -1).contiguous()
            candidates.masked_fill_(
                candidates >= torch.div(ctx + 7, 8, rounding_mode="floor"), -1
            )
            if args.candidate == "mask":
                candidate_fn = (
                    lambda logits=topk_logits, ctx=ctx, candidates=candidates: (
                        apply_candidate_mask(
                            logits, None, ctx.view(-1), candidates, 8, 1
                        )
                    )
                )
            else:

                def candidate_fn(
                    logits=topk_logits, ctx=ctx, candidates=candidates, number=number
                ):
                    return select_candidate_blocks(
                        logits, None, ctx.view(-1), number, 8, candidates, 1
                    )

            candidate_fn()
            fns["candidate_" + args.candidate] = candidate_fn
        fns["topk"] = lambda topk_logits=topk_logits, ctx=ctx, start=start, end=end: (
            get_indexer_topk(backend)(
                topk_logits, ctx, 1, x["out"][start:end, :topk], topk, host_max
            )
        )
        entry = dict(
            rows=end - start,
            occurrences=sum(e - s == end - start for s, e in chunks),
            stages={},
        )
        for name, fn in fns.items():
            samples = bench_gpu_time_with_cupti(
                fn,
                dry_run_iters=3,
                repeat_iters=args.repeat,
                use_cuda_graph=True,
                cold_l2_cache=False,
            )
            entry["stages"][name] = distribution([v * 1000 for v in samples])
        result.append(entry)
        del topk_logits, logits, schedule, fns
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", default="v4,v41-c2,v41-c1")
    parser.add_argument(
        "--rows", type=csv_ints, default=csv_ints("8,64,65,256,512,1024,4096")
    )
    parser.add_argument(
        "--contexts", type=csv_ints, default=csv_ints("1536,8192,32768")
    )
    parser.add_argument("--dtypes", default="mxfp4,fp8")
    parser.add_argument("--q-len", type=int, default=8)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument(
        "--variants",
        default="single-auto,single-ds,final-512,prototype-1024,auto-512,ds-256,ds-512,ds-1024,ds-2048,prebuilt-512",
    )
    parser.add_argument(
        "--candidate", choices=["none", "mask", "select"], default="none"
    )
    parser.add_argument("--cache", default="warm,cold")
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--eager-reps", type=int, default=10)
    parser.add_argument("--eager-warmup", type=int, default=0)
    parser.add_argument("--eager-layers", type=int, default=1)
    parser.add_argument("--profile-replays", type=int, default=5)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--launch-details", action="store_true")
    parser.add_argument("--details-repeats", type=int, default=5)
    parser.add_argument(
        "--production-helper",
        action="store_true",
        help="Use the active checkout's varlen planner/helper for final-* arms",
    )
    parser.add_argument("--stages", action="store_true")
    parser.add_argument(
        "--max-seq-mode", choices=["capture", "actual"], default="capture"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if int(importlib.metadata.version("cupti-python").split(".")[0]) < 13:
        raise RuntimeError("CUPTI >=13 is required; do not silently fall back")
    from cupti import cupti  # noqa: F401

    init_workspace_manager(torch.device("cuda"))
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    native_root = Path(vllm.__file__).parent
    native_artifacts = {}
    for pattern in (
        "_C_stable_libtorch*.so",
        "_deepselect_C*.so",
        "third_party/deep_gemm/_C_extension*.so",
    ):
        for artifact in native_root.glob(pattern):
            native_artifacts[str(artifact.relative_to(native_root))] = {
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "bytes": artifact.stat().st_size,
            }
    metadata = dict(
        git_sha=git_sha,
        vllm_python_root=str(native_root),
        native_artifacts=native_artifacts,
        production_helper_for_final=args.production_helper,
        hostname=socket.gethostname(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(
            torch.cuda.get_device_properties(
                torch.accelerator.current_device_index()
            ).uuid
        ),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cupti_python=importlib.metadata.version("cupti-python"),
        flashinfer=importlib.metadata.version("flashinfer-python"),
        executable=os.sys.executable,
        argv=os.sys.argv,
        source_file=__file__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            value = json.loads(line)
            if value.get("status") == "ok":
                completed.add(value["key"])
    for model in args.models.split(","):
        heads, width, compression, model_layers = MODELS[model]
        for dtype in args.dtypes.split(","):
            for context in args.contexts:
                for rows in args.rows:
                    case = dict(
                        model=model,
                        dtype=dtype,
                        rows=rows,
                        heads=heads,
                        head_dim=128,
                        width=width,
                        compression=compression,
                        context_tokens=context,
                        q_len=args.q_len,
                        requests=math.ceil(rows / args.q_len),
                        topk=args.topk,
                        candidate=args.candidate,
                        max_seq_mode=args.max_seq_mode,
                    )
                    variants = args.variants.split(",")
                    random.Random(rows + context).shuffle(variants)
                    x = None
                    try:
                        x = make_inputs(
                            rows,
                            heads,
                            width,
                            compression,
                            context,
                            dtype,
                            args.q_len,
                            args.topk,
                        )
                        for variant in variants:
                            key = json.dumps(
                                dict(
                                    {k: v for k, v in case.items() if k != "requests"},
                                    variant=variant,
                                ),
                                sort_keys=True,
                            )
                            if key in completed:
                                continue
                            record = dict(
                                metadata,
                                case=case,
                                variant=variant,
                                key=key,
                                started=time.time(),
                                telemetry_before=telemetry(),
                            )
                            try:
                                mapped = (
                                    "ds-512" if variant == "prebuilt-512" else variant
                                )
                                backend, chunks = plan(rows, width, mapped)
                                production_helper = None
                                if args.production_helper and variant.startswith(
                                    "final-"
                                ):
                                    backend, production_chunks = (
                                        indexer_module.varlen_decode_plan(
                                            rows,
                                            width,
                                            "auto",
                                            int(variant.rsplit("-", 1)[1]) * MIB,
                                        )
                                    )
                                    if production_chunks:
                                        chunks = production_chunks
                                        production_helper = (
                                            indexer_module._varlen_decode_rows
                                        )
                                run = runner(
                                    x,
                                    backend,
                                    chunks,
                                    args.topk,
                                    width * compression
                                    if args.max_seq_mode == "capture"
                                    else context,
                                    args.candidate,
                                    prebuilt=variant == "prebuilt-512",
                                    production_helper=production_helper,
                                )
                                check = correctness(
                                    x, run, chunks, args.topk, args.candidate
                                )
                                record.update(
                                    backend=backend, chunks=chunks, correctness=check
                                )
                                if (
                                    check["chunked_logit_mismatch_rows"]
                                    or check["nonfinite_valid_logits"]
                                    or check["invalid_index_rows"]
                                    or check["duplicate_index_rows"]
                                ):
                                    raise AssertionError(check)
                                if check["exact_topk_value_mismatch_rows"]:
                                    raise AssertionError(check)
                                if any(check["candidate_check"].values()):
                                    raise AssertionError(check)
                                record.update(
                                    backend=backend,
                                    chunks=chunks,
                                    correctness=check,
                                    model_indexer_layers=model_layers,
                                    capacity_elements=rows * width,
                                    valid_elements=x["valid_scores"],
                                    input_k_cache_bytes=x["kv"].numel(),
                                    input_layout={
                                        name: dict(
                                            shape=list(x[name].shape),
                                            stride=list(x[name].stride()),
                                            dtype=str(x[name].dtype),
                                        )
                                        for name in (
                                            "q",
                                            "qs",
                                            "kv",
                                            "bt",
                                            "ctx",
                                            "indices",
                                            "weights",
                                        )
                                    },
                                )
                                record["memory"] = memory(run)
                                record["timings"] = benchmark(run, args)
                                if args.launch_details:
                                    record["launch_details"] = launch_details(
                                        run, args.details_repeats
                                    )
                                if args.stages:
                                    record["isolated_stages_warm"] = stage_timings(
                                        x,
                                        backend,
                                        chunks,
                                        args.topk,
                                        width * compression
                                        if args.max_seq_mode == "capture"
                                        else context,
                                        args,
                                    )
                                if not args.no_profile:
                                    try:
                                        record["cuda_kernels"] = profile_kernels(
                                            run, args.profile_replays
                                        )
                                    except Exception as error:
                                        record["profile_error"] = repr(error)
                                record["status"] = "ok"
                                del run
                            except Exception as error:
                                record.update(
                                    status="error",
                                    error=repr(error),
                                    traceback=traceback.format_exc(),
                                )
                            record.update(
                                elapsed_s=time.time() - record["started"],
                                telemetry_after=telemetry(),
                            )
                            with args.out.open("a") as output:
                                output.write(json.dumps(record) + "\n")
                                output.flush()
                                os.fsync(output.fileno())
                            print(
                                json.dumps(
                                    {
                                        k: record[k]
                                        for k in (
                                            "case",
                                            "variant",
                                            "status",
                                            "elapsed_s",
                                        )
                                    }
                                ),
                                flush=True,
                            )
                            gc.collect()
                            torch.accelerator.empty_cache()
                    except Exception as error:
                        with args.out.open("a") as output:
                            output.write(
                                json.dumps(
                                    dict(
                                        metadata,
                                        case=case,
                                        status="input_error",
                                        error=repr(error),
                                        traceback=traceback.format_exc(),
                                    )
                                )
                                + "\n"
                            )
                    finally:
                        del x
                        gc.collect()
                        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
