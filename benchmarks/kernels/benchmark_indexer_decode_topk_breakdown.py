# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Resumable SM100 indexer top-k experiments, starting from BF16 logits.

Lengths and capacity are measured in logits columns (compressed positions),
not uncompressed model tokens. A record is one independently timed trial.
The input BF16 logits, final indices and shared radix workspace are excluded
from additional-allocation measurements. An FP32-only arm additionally treats
its prepared FP32 input as an input, rather than charging it to the arm.

Example:
    .venv/bin/python benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py \
        --output benchmarks/kernels/indexer_breakdown/results/topk_probe.jsonl \
        --preset probe --resume

"""

import argparse
import bisect
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import random
import socket
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

# Snapshot provenance before the potentially slow CUDA-dependent imports.
_SCRIPT_SOURCE = Path(__file__).read_bytes()

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.indexer_topk import (
    RADIX_TOPK_WORKSPACE_SIZE,
    get_indexer_topk,
)
from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    is_workspace_manager_initialized,
)

SCHEMES = (
    "DS-bf16",
    "UP+COOP",
    "UP+PERS",
    "UP+DS-fp32",
    "UP+FI",
    "UP+PERROW",
    "COOP-fp32-only",
    "PERS-fp32-only",
    "UP-only",
)
BACKENDS = {
    "DS-bf16": "deep_select",
    "UP+COOP": "cooperative",
    "UP+PERS": "persistent",
    "UP+DS-fp32": "deep_select",
    "UP+FI": "flashinfer",
    "UP+PERROW": "per_row",
    "COOP-fp32-only": "cooperative",
    "PERS-fp32-only": "persistent",
}
SPEC_FIELDS = (
    "rows",
    "capacity",
    "stride",
    "topk",
    "length_profile",
    "distribution",
    "scheme",
    "execution",
    "cache",
    "bound_kind",
    "max_seq_len",
    "seed",
    "repeat",
    "query_len",
    "compress_ratio",
    "logits_source_sha256",
    "file_random_control",
)
SELECTION_ERRORS = (
    "wrong_count_rows",
    "wrong_sentinel_rows",
    "out_of_bounds_rows",
    "duplicate_rows",
    "value_multiset_mismatch_rows",
)


class InvalidSelectionError(RuntimeError):
    """A warmup produced incorrect selection; do not accept timing."""

    def __init__(self, check: dict):
        super().__init__(f"Invalid selected values or indices: {check}")
        self.check = check


def configuration_key(record: dict) -> tuple:
    return tuple(record.get(field) for field in SPEC_FIELDS)


def command_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return str(exc)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def telemetry() -> dict:
    query = "uuid,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw"
    return {
        "nvidia_smi": command_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
        )
    }


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "median": statistics.median(values),
        "p10": percentile(values, 0.1),
        "p90": percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
        "samples": len(values),
    }


def cupti_kernel_details(
    runner: Callable, repeats: int, flush: torch.Tensor | None
) -> dict:
    """Expose per-kernel activities using FlashInfer's CUPTI tracing method.

    FlashInfer's public helper returns only the first-to-last GPU span. This
    separate trace retains kernel names/durations and their sum, allowing a
    host-induced gap inside an eager sequence to be distinguished from work.
    """
    from cupti import cupti

    kernels, launches, intervals = [], [], []
    configurations = {}

    def completed(activities):
        for activity in activities:
            if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                fields = (
                    "grid_x",
                    "grid_y",
                    "grid_z",
                    "block_x",
                    "block_y",
                    "block_z",
                    "cluster_x",
                    "cluster_y",
                    "cluster_z",
                    "registers_per_thread",
                    "static_shared_memory",
                    "dynamic_shared_memory",
                )
                configurations[str(activity.name)] = {
                    field: int(getattr(activity, field))
                    for field in fields
                    if getattr(activity, field, None) is not None
                }
            if activity.kind in (
                cupti.ActivityKind.CONCURRENT_KERNEL,
                cupti.ActivityKind.MEMCPY,
                cupti.ActivityKind.MEMSET,
            ):
                kernels.append(
                    (
                        str(activity.name)
                        if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL
                        else str(activity.kind),
                        activity.start,
                        activity.end,
                        activity.correlation_id,
                    )
                )
            elif activity.kind in (
                cupti.ActivityKind.RUNTIME,
                cupti.ActivityKind.DRIVER,
            ):
                launches.append((activity.start, activity.correlation_id))

    kinds = (
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.MEMCPY,
        cupti.ActivityKind.MEMSET,
    )
    try:
        for kind in kinds:
            cupti.activity_enable(kind)
        cupti.activity_register_callbacks(lambda: (8 * 1024 * 1024, 0), completed)
        for _ in range(repeats):
            if flush is not None:
                flush.zero_()
            torch.accelerator.synchronize()
            start = cupti.get_timestamp()
            runner()
            end = cupti.get_timestamp()
            torch.accelerator.synchronize()
            intervals.append((start, end))
        cupti.activity_flush_all(0)
    finally:
        for kind in kinds:
            cupti.activity_disable(kind)
        cupti.finalize()

    launches.sort()
    starts = [item[0] for item in launches]
    by_id = defaultdict(list)
    for kernel in kernels:
        by_id[kernel[3]].append(kernel)
    by_name = defaultdict(list)
    spans, sums, counts = [], [], []
    for start, end in intervals:
        lo, hi = bisect.bisect_left(starts, start), bisect.bisect_right(starts, end)
        ids = {correlation for _, correlation in launches[lo:hi]}
        selected = [kernel for key in ids for kernel in by_id[key]]
        if not selected:
            raise RuntimeError("CUPTI returned no kernels for a timed invocation")
        spans.append((max(k[2] for k in selected) - min(k[1] for k in selected)) / 1000)
        sums.append(sum((k[2] - k[1]) / 1000 for k in selected))
        counts.append(len(selected))
        for name, begin, finish, _ in selected:
            by_name[name].append((finish - begin) / 1000)
    return {
        "gpu_span_us": summarize(spans),
        "gpu_kernel_sum_us": summarize(sums),
        "kernels_per_invocation": summarize(counts),
        "kernels": {
            name: {
                **summarize(times),
                "calls_per_invocation": len(times) / repeats,
                "launch_config": configurations.get(name, {}),
            }
            for name, times in by_name.items()
        },
    }


def make_lengths(
    rows: int,
    width: int,
    profile: str,
    seed: int,
    q: int,
    compress_ratio: int,
    logits_file: Path | None = None,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if profile == "file":
        if logits_file is None:
            raise ValueError("The file length profile requires --logits-file")
        loaded = torch.load(logits_file, map_location="cpu", weights_only=True)
        lengths = loaded["seq_lens"].flatten().to(torch.int32)
        if lengths.max().item() > min(width, loaded["logits"].shape[1]):
            raise ValueError("Saved effective lengths exceed available logits")
        lengths = lengths.repeat(math.ceil(rows / lengths.numel()))[:rows]
    elif profile in ("short", "mid", "long", "full"):
        maximum = min(
            width, {"short": 4096, "mid": 32768, "long": 131072, "full": width}[profile]
        )
        lengths = torch.full((rows,), maximum, dtype=torch.int32)
    elif profile == "ragged":
        low = min(1024, width)
        lengths = (
            torch.exp(
                torch.rand(rows, generator=gen) * math.log(width / low) + math.log(low)
            )
            .to(torch.int32)
            .clamp_(1, width)
        )
    elif profile == "spec":
        reqs = (rows + q - 1) // q
        low = min(1024, width)
        bases = torch.randint(low, width + 1, (reqs,), generator=gen)
        # Expand in original-token coordinates and then compress, just as the
        # SM100 metadata builder does. Compressed bounds need not advance by 1.
        offsets = torch.arange(q) - q + 1
        lengths = (
            ((bases[:, None] * compress_ratio + offsets).clamp_min(0) // compress_ratio)
            .flatten()[:rows]
            .to(torch.int32)
        )
    elif profile == "padding":
        lengths = torch.full((rows,), min(width, 32768), dtype=torch.int32)
        lengths[::7] = 0
        lengths[1::11] = min(width, 128)
    else:
        raise ValueError(profile)
    return lengths.to(device="cuda")


def make_input(
    rows: int,
    width: int,
    padding: int,
    distribution: str,
    seed: int,
    logits_file: Path | None,
) -> torch.Tensor:
    torch.manual_seed(seed)
    logits = torch.empty_strided(
        (rows, width), (width + padding, 1), dtype=torch.bfloat16, device="cuda"
    )
    if logits_file is not None:
        loaded = torch.load(logits_file, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            loaded = loaded["logits"]
        columns = min(loaded.shape[1], width)
        logits.fill_(-float("inf"))
        for start in range(0, rows, loaded.shape[0]):
            count = min(loaded.shape[0], rows - start)
            logits[start : start + count, :columns].copy_(loaded[:count, :columns])
    else:
        logits.normal_()
        if distribution == "ties":
            logits.mul_(2).round_()
    return logits


def make_scheme(
    scheme: str,
    logits: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    k: int,
    bound: int,
):
    fp32 = as_fp32_logits(logits, lengths) if "fp32-only" in scheme else None
    backend = BACKENDS.get(scheme)

    def run():
        if scheme == "UP-only":
            return as_fp32_logits(logits, lengths)
        scores = (
            fp32
            if fp32 is not None
            else as_fp32_logits(logits, lengths)
            if scheme.startswith("UP+")
            else logits
        )
        get_indexer_topk(backend)(scores, lengths[:, None], 1, out, k, bound)
        return out

    return run, fp32


def correctness(logits, lengths, out, k: int, check_rows: int) -> dict:
    rows, width = logits.shape
    expected_count = torch.minimum(lengths, torch.full_like(lengths, k))
    index_valid = (out >= 0) & (out < lengths[:, None])
    count_bad = (index_valid.sum(1) != expected_count).sum().item()
    sentinel_bad = ((out == -1).sum(1) != k - expected_count).sum().item()
    oob = ((out < -1) | (out >= lengths[:, None])).any(1).sum().item()
    ordered = out.sort(dim=1).values
    duplicate = (
        ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0))
        .any(1)
        .sum()
        .item()
    )
    selected_rows = (
        torch.linspace(0, rows - 1, min(rows, check_rows), device="cuda")
        .round()
        .long()
        .unique()
    )
    values = logits[selected_rows].float()
    ends = lengths[selected_rows]
    columns = torch.arange(width, device="cuda")[None]
    values.masked_fill_(columns >= ends[:, None], -float("inf"))
    ref_values, ref_indices = values.topk(k, dim=1)
    ref_indices = torch.where(
        torch.arange(k, device="cuda")[None] < ends[:, None], ref_indices, -1
    )
    selected = out[selected_rows]
    selected_values = values.gather(1, selected.long().clamp(0, width - 1))
    selected_values = torch.where(selected >= 0, selected_values, -float("inf"))
    value_equal = (selected_values.sort(1, descending=True).values == ref_values).all(1)
    set_equal = (selected.sort(1).values == ref_indices.sort(1).values).all(1)
    return {
        "all_rows_bounds_checked": rows,
        "reference_rows_checked": selected_rows.numel(),
        "reference_is_sampled": selected_rows.numel() != rows,
        "wrong_count_rows": int(count_bad),
        "wrong_sentinel_rows": int(sentinel_bad),
        "out_of_bounds_rows": int(oob),
        "duplicate_rows": int(duplicate),
        "value_multiset_mismatch_rows": int((~value_equal).sum().item()),
        "index_set_mismatch_rows": int((~set_equal).sum().item()),
        "ties_only_different_index_rows": int((value_equal & ~set_equal).sum().item()),
    }


def host_timing(runner: Callable, repeats: int, flush: torch.Tensor | None) -> dict:
    enqueue, wall = [], []
    for _ in range(repeats):
        if flush is not None:
            flush.zero_()
        torch.accelerator.synchronize()
        start = time.perf_counter_ns()
        runner()
        submitted = time.perf_counter_ns()
        torch.accelerator.synchronize()
        finished = time.perf_counter_ns()
        enqueue.append((submitted - start) / 1000)
        wall.append((finished - start) / 1000)
    return {"host_enqueue_us": summarize(enqueue), "wall_us": summarize(wall)}


def measure(scheme, logits, lengths, out, k, bound, mode, cache, args):
    run, fp32 = make_scheme(scheme, logits, lengths, out, k, bound)
    for _ in range(3):
        run()
    torch.accelerator.synchronize()
    check = (
        None
        if scheme == "UP-only"
        else correctness(logits, lengths, out, k, args.check_rows)
    )
    if check and any(check[field] for field in SELECTION_ERRORS):
        raise InvalidSelectionError(check)
    torch.accelerator.synchronize()
    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats()
    allocated_before = torch.accelerator.memory_allocated()
    reserved_before = torch.accelerator.memory_reserved()
    graph, pool = None, None
    if mode == "graph":
        pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            run()
        runner = graph.replay
    else:
        run()
        runner = run
    torch.accelerator.synchronize()
    memory = {
        "additional_peak_allocated_bytes": (
            torch.accelerator.max_memory_allocated() - allocated_before
        ),
        "reserved_delta_bytes": torch.accelerator.memory_reserved() - reserved_before,
        "graph_pool_id": list(pool) if pool is not None else None,
        "prepared_fp32_input_bytes": (
            fp32.untyped_storage().nbytes() if fp32 is not None else 0
        ),
        "shared_radix_workspace_bytes": RADIX_TOPK_WORKSPACE_SIZE,
        "scheme_uses_shared_radix_workspace": BACKENDS.get(scheme)
        in ("cooperative", "persistent"),
        "input_bf16_storage_bytes": logits.untyped_storage().nbytes(),
        "final_indices_bytes": out.numel() * out.element_size(),
    }
    cold = cache == "cold"
    public_flush_kwargs = {}
    if cold:
        l2_size = torch.cuda.get_device_properties(0).L2_cache_size
        public_flush_kwargs = {
            "l2_flush": True,
            "l2_flush_size_mb": math.ceil(2 * l2_size / (1024 * 1024)),
        }
    samples = [
        value * 1000
        for value in bench_gpu_time_with_cupti(
            runner,
            use_cuda_graph=False,
            cold_l2_cache=cold,
            dry_run_iters=3,
            repeat_iters=args.repeat,
            **public_flush_kwargs,
        )
    ]
    result = {
        "gpu_span_us": summarize(samples),
        "samples_us": samples,
        "memory": memory,
        "correctness": check,
    }
    # The public FlashInfer helper above measures the real graph replay when
    # mode=graph: runner is already a captured graph, so don't capture again.
    flush = None
    if cold:
        l2 = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
        flush = torch.empty(
            max(128 * 1024 * 1024, 2 * l2), dtype=torch.uint8, device="cuda"
        )
    result.update(host_timing(runner, args.host_repeat, flush))
    if args.kernel_details:
        result["kernel_breakdown"] = cupti_kernel_details(
            runner, args.detail_repeat, flush
        )
    if args.replay_check and mode == "graph" and scheme != "UP-only":
        result["replay_correctness"] = correctness(
            logits, lengths, out, k, args.check_rows
        )
        saved_lengths = lengths.clone()
        try:
            changed_lengths = saved_lengths // 2
            changed_lengths[::3] = 0
            changed_lengths[1::3] = changed_lengths[1::3].clamp_max(k // 2)
            lengths.copy_(changed_lengths)
            runner()
            torch.accelerator.synchronize()
            result["changed_lengths_replay_correctness"] = correctness(
                logits, lengths, out, k, args.check_rows
            )
        finally:
            lengths.copy_(saved_lengths)
            runner()
            torch.accelerator.synchronize()
        del saved_lengths, changed_lengths
    del runner, run, graph, pool, fp32, flush
    torch.accelerator.synchronize()
    gc.collect()
    torch.accelerator.empty_cache()
    return result


def append_record(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sentinel(args, metadata: dict, count: int, baseline: float | None) -> float:
    """Track drift with a constant workload; leave rerun policy to analysis."""
    logits = make_input(64, 262144, 512, "normal", 123, None)
    lengths = make_lengths(64, 262144, "long", 123, 1, 1)
    out = torch.empty((64, 2048), device="cuda", dtype=torch.int32)
    result = measure(
        "UP+PERS", logits, lengths, out, 2048, 262144, "graph", "warm", args
    )
    observed = result["gpu_span_us"]["median"]
    baseline = baseline or observed
    append_record(
        args.output,
        {
            **metadata,
            **result,
            "status": "ok",
            "record_type": "sentinel",
            "rows": 64,
            "capacity": 262144,
            "topk": 2048,
            "length_profile": "long",
            "scheme": "UP+PERS",
            "execution": "graph",
            "cache": "warm",
            "bound_kind": "capture",
            "max_seq_len": 262144,
            "repeat": args.repeat,
            "ordinal_after_record": count,
            "unix_time": time.time(),
            "drift_from_first_fraction": observed / baseline - 1,
            "drift_exceeds_3pct": abs(observed / baseline - 1) > 0.03,
            "telemetry_after": telemetry(),
        },
    )
    del logits, lengths, out
    gc.collect()
    torch.accelerator.empty_cache()
    return baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=Path,
        nargs="*",
        default=[],
        help="Reuse exact keys from same-GPU/same-native result files; no data copied",
    )
    parser.add_argument("--preset", choices=["probe", "core", "manual"], default="core")
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        help="Total query rows R, including any graph padding",
    )
    parser.add_argument(
        "--capacities",
        type=int,
        nargs="+",
        help="Logical logits columns W; separate from effective row length",
    )
    parser.add_argument("--top-k", type=int, nargs="+", default=[512])
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=["short", "mid", "long", "full", "ragged", "spec", "padding", "file"],
        default=["short", "long", "ragged"],
        help="Effective row lengths; short/mid/long are 4K/32K/128K",
    )
    parser.add_argument(
        "--distributions", nargs="+", choices=["normal", "ties"], default=["normal"]
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=SCHEMES,
        default=[
            "DS-bf16",
            "UP+COOP",
            "UP+PERS",
            "UP+DS-fp32",
            "COOP-fp32-only",
            "PERS-fp32-only",
            "UP-only",
        ],
    )
    parser.add_argument(
        "--modes",
        choices=["eager", "graph"],
        nargs="+",
        default=["graph"],
        help="graph captures the whole top-k scheme, excluding DeepGEMM",
    )
    parser.add_argument(
        "--caches",
        choices=["warm", "cold"],
        nargs="+",
        default=["warm"],
        help="warm repeats inputs; cold flushes >=2xL2 outside timing",
    )
    parser.add_argument(
        "--bounds",
        choices=["actual", "raw_actual", "capture"],
        nargs="+",
        default=["actual", "capture"],
        help=(
            "actual: compressed max; raw_actual: uncompressed max; "
            "capture: fixed host max baked into the CUDA graph"
        ),
    )
    parser.add_argument(
        "--capture-max-seq-len",
        type=int,
        help="Configured uncompressed model limit for FULL capture",
    )
    parser.add_argument(
        "--raw-max-seq-len",
        type=int,
        help="Exact raw host bound; floor-compression loses remainders",
    )
    parser.add_argument(
        "--stride-padding",
        type=int,
        help="Override padding; default 1024 at W512K, otherwise 512",
    )
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--host-repeat", type=int, default=10)
    parser.add_argument("--detail-repeat", type=int, default=100)
    parser.add_argument("--trial-offset", type=int, default=0)
    parser.add_argument("--rerun-unstable", action="store_true")
    parser.add_argument("--cv-threshold", type=float, default=0.03)
    parser.add_argument(
        "--replay-check",
        action="store_true",
        help="Validate after many replays and in-place length updates",
    )
    parser.add_argument("--sentinel-every", type=int, default=200)
    parser.add_argument("--prune-unused-bounds", action="store_true")
    parser.add_argument(
        "--kernel-details", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--check-rows",
        type=int,
        default=128,
        help="Reference-check all rows up to this count; evenly sample larger batches",
    )
    parser.add_argument("--query-len", type=int, default=8)
    parser.add_argument("--compress-ratio", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=3 * 3600,
        help="Stop between input shapes after this wall-time budget",
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--logits-file",
        type=Path,
        help="torch.save dict: logits, seq_lens, optional metadata",
    )
    parser.add_argument("--file-random-control", action="store_true")
    parser.add_argument(
        "--skip-cases",
        type=Path,
        help="JSONL match dictionaries for explicitly unsafe layouts",
    )
    parser.add_argument(
        "--log-attempts", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.repeat < 1 or args.trials < 1:
        parser.error("repeat and trials must be positive")
    if args.query_len < 1 or args.compress_ratio < 1:
        parser.error("query-len and compress-ratio must be positive")
    if args.logits_file and args.profiles != ["file"]:
        parser.error("Saved-prefix logits require --profiles file")
    if args.file_random_control and args.logits_file is None:
        parser.error("--file-random-control requires --logits-file")
    if args.stride_padding is not None and args.stride_padding < 0:
        parser.error("stride-padding must be nonnegative")
    if args.preset == "probe":
        rows = args.rows or [1, 64, 65, 85, 256]
        capacities = args.capacities or [1048576]
    else:
        rows = args.rows or [
            1,
            2,
            4,
            8,
            16,
            32,
            48,
            60,
            63,
            64,
            65,
            66,
            72,
            80,
            85,
            86,
            96,
            128,
            160,
            170,
            171,
            192,
            256,
            320,
            384,
            512,
            1024,
            2048,
            4096,
            8192,
        ]
        capacities = args.capacities or [1048576, 524288, 262144, 163840, 65536]
    if any(width < max(args.top_k) for width in capacities):
        parser.error("capacity must be at least top-k")
    cupti_version = package_version("cupti-python")
    if cupti_version is None or int(cupti_version.split(".")[0]) < 13:
        raise RuntimeError("cupti-python is required; refusing silent event fallback")
    from cupti import cupti as required_cupti

    required_cupti.get_timestamp()
    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda", 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    existing = []
    if args.resume and args.output.exists():
        for line in args.output.read_text().splitlines():
            try:
                entry = json.loads(line)
                if entry.get("status") in (
                    "ok",
                    "unsupported",
                    "unsafe_skipped",
                    "invalid_selection",
                ):
                    done.add(entry["key"])
                    if (
                        entry["status"] == "ok"
                        and not (
                            entry.get("cache") == "cold"
                            and not entry.get("public_cold_flush_bytes")
                        )
                        and not any(
                            (entry.get(check) or {}).get(field, 0)
                            for check in (
                                "correctness",
                                "replay_correctness",
                                "changed_lengths_replay_correctness",
                            )
                            for field in SELECTION_ERRORS
                        )
                    ):
                        existing.append(entry)
            except (ValueError, KeyError):
                pass
    elif args.output.exists():
        raise FileExistsError("Output exists; pass --resume or choose a new path")
    reused_sources = []
    for source in args.resume_from:
        raw = source.read_bytes()
        reused = 0
        for line in raw.splitlines():
            try:
                entry = json.loads(line)
                if (
                    entry.get("status") in ("ok", "unsupported")
                    and entry.get("hostname") == socket.gethostname()
                    and entry.get("cuda_visible_devices")
                    == os.getenv("CUDA_VISIBLE_DEVICES")
                    and entry.get("key")
                ):
                    done.add(entry["key"])
                    reused += 1
            except ValueError:
                pass
        reused_sources.append(
            {
                "path": str(source),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "eligible_keys": reused,
            }
        )
    skip_rules = (
        [json.loads(line) for line in args.skip_cases.read_text().splitlines()]
        if args.skip_cases
        else []
    )
    unstable = None
    if args.rerun_unstable:
        groups = defaultdict(list)
        for entry in existing:
            groups[configuration_key(entry)].append(entry["gpu_span_us"]["median"])
        unstable = {
            key
            for key, values in groups.items()
            if len(values) >= 3
            and statistics.pstdev(values) / statistics.mean(values) > args.cv_threshold
        }
        args.trial_offset = max((entry["trial"] for entry in existing), default=-1) + 1
        print(
            f"CV rerun: {len(unstable)} configurations, "
            f"trial offset {args.trial_offset}",
            flush=True,
        )
        if not unstable:
            return
    props = torch.cuda.get_device_properties(0)
    logits_source_sha256 = (
        hashlib.sha256(args.logits_file.read_bytes()).hexdigest()
        if args.logits_file
        else None
    )
    native_root = Path(vllm.__file__).parent
    native_artifacts = {}
    for pattern in ("_C_stable_libtorch*.so", "_deepselect_C*.so"):
        for artifact in native_root.glob(pattern):
            native_artifacts[artifact.name] = {
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "bytes": artifact.stat().st_size,
            }
    metadata = {
        "git_sha": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": command_output(["git", "status", "--short"]),
        "vllm_python_root": str(native_root),
        "native_artifacts": native_artifacts,
        "script_sha256": hashlib.sha256(_SCRIPT_SOURCE).hexdigest(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gpu": props.name,
        "gpu_uuid": str(props.uuid) if hasattr(props, "uuid") else None,
        "compute_capability": [props.major, props.minor],
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": package_version("flashinfer-python"),
        "cupti_python": package_version("cupti-python"),
        "timing_method": "flashinfer CUPTI activity first-to-last GPU span",
        "repeat_kernel_details": args.detail_repeat,
        "repeat_host": args.host_repeat,
        "replay_check_enabled": args.replay_check,
        "stride_padding_override": args.stride_padding,
        "raw_max_seq_len_override": args.raw_max_seq_len,
        "reused_result_sources": reused_sources,
        "warm_definition": "repeated same-input replay; not producer-warmed L2",
        "cold_flush_bytes": 2 * props.L2_cache_size,
        "public_cold_flush_bytes": (
            math.ceil(2 * props.L2_cache_size / (1024 * 1024)) * 1024 * 1024
        ),
        "rerun_unstable": args.rerun_unstable,
    }
    if args.logits_file is not None:
        loaded = torch.load(args.logits_file, map_location="cpu", weights_only=True)
        file_metadata = dict(loaded.get("metadata", {}))
        for field in ("family", "layer", "full_shape", "full_stride"):
            if field in loaded:
                file_metadata[field] = loaded[field]
        if "full_shape" in file_metadata:
            file_metadata.setdefault("capacity", file_metadata["full_shape"][1])
        if "full_stride" in file_metadata:
            file_metadata.setdefault(
                "original_logits_stride", file_metadata["full_stride"]
            )
        metadata["input_file_metadata"] = file_metadata
    source_archive = args.output.parent / "topk_runner_sources.jsonl"
    source_key = metadata["script_sha256"]
    if not source_archive.exists() or source_key not in source_archive.read_text():
        append_record(
            source_archive,
            {
                "status": "metadata",
                "record_type": "runner_source",
                "script_sha256": source_key,
                "source": _SCRIPT_SOURCE.decode(),
            },
        )
    start_time, count = time.monotonic(), 0
    sentinel_baseline = None
    shapes = list(
        itertools.product(
            rows, capacities, args.top_k, args.profiles, args.distributions
        )
    )
    for shape_index, (row_count, capacity, k, profile, distribution) in enumerate(
        shapes
    ):
        if time.monotonic() - start_time >= args.max_seconds:
            break
        padding = args.stride_padding
        file_metadata = metadata.get("input_file_metadata", {})
        if padding is None:
            padding = 1024 if capacity == 524288 else 512
            if (
                file_metadata.get("capacity") == capacity
                and "original_logits_stride" in file_metadata
            ):
                padding = int(file_metadata["original_logits_stride"][0]) - capacity
        logits = make_input(
            row_count,
            capacity,
            padding,
            distribution,
            args.seed,
            None if args.file_random_control else args.logits_file,
        )
        lengths = make_lengths(
            row_count,
            capacity,
            profile,
            args.seed,
            args.query_len,
            args.compress_ratio,
            args.logits_file,
        )
        out = torch.empty((row_count, k), device="cuda", dtype=torch.int32)
        actual_max = int(lengths.max().item())
        bounds = {
            "actual": actual_max,
            "raw_actual": int(
                args.raw_max_seq_len
                or file_metadata.get(
                    "raw_context_tokens", actual_max * args.compress_ratio
                )
            ),
            "capture": args.capture_max_seq_len or capacity,
        }
        valid_elements = int(lengths.sum().item())
        for trial in range(args.trial_offset, args.trial_offset + args.trials):
            combinations = list(
                itertools.product(args.schemes, args.modes, args.caches, args.bounds)
            )
            random.Random(args.seed + trial + shape_index * 100).shuffle(combinations)
            for order, (scheme, mode, cache, bound_kind) in enumerate(combinations):
                if args.prune_unused_bounds and not args.rerun_unstable:
                    if bound_kind != "actual" and BACKENDS.get(scheme) != "persistent":
                        continue
                    if bound_kind == "raw_actual" and bounds[bound_kind] == actual_max:
                        continue
                    if bound_kind == "capture" and (
                        bounds[bound_kind] == actual_max
                        or (
                            "raw_actual" in args.bounds
                            and bounds[bound_kind] == bounds["raw_actual"]
                        )
                    ):
                        continue
                spec = {
                    "rows": row_count,
                    "capacity": capacity,
                    "stride": logits.stride(0),
                    "topk": k,
                    "length_profile": profile,
                    "distribution": distribution,
                    "scheme": scheme,
                    "execution": mode,
                    "cache": cache,
                    "bound_kind": bound_kind,
                    "trial": trial,
                    "max_seq_len": bounds[bound_kind],
                    "seed": args.seed,
                    "repeat": args.repeat,
                    "query_len": args.query_len,
                    "compress_ratio": args.compress_ratio,
                }
                if logits_source_sha256 is not None:
                    spec["logits_source_sha256"] = logits_source_sha256
                    spec["file_random_control"] = args.file_random_control
                if unstable is not None and configuration_key(spec) not in unstable:
                    continue
                key = hashlib.sha256(
                    json.dumps(spec, sort_keys=True).encode()
                ).hexdigest()
                if key in done:
                    continue
                record = {
                    **metadata,
                    **spec,
                    "key": key,
                    "order_within_trial": order,
                    "unix_time": time.time(),
                    "telemetry_before": telemetry(),
                    "actual_max_row_length": actual_max,
                    "valid_elements": valid_elements,
                    "mean_row_length": valid_elements / row_count,
                    "upcast_useful_traffic_bytes": 6 * valid_elements,
                    "request_count_if_uniform": row_count / args.query_len,
                    "input_source": str(args.logits_file)
                    if args.logits_file
                    else "synthetic",
                    "input_values": "randn_matched_saved_lengths"
                    if args.file_random_control
                    else "saved"
                    if args.logits_file
                    else distribution,
                }
                if args.log_attempts:
                    append_record(
                        args.output.with_name(args.output.stem + "_attempts.jsonl"),
                        {
                            "status": "attempt",
                            "record_type": "case_attempt",
                            "output_file": str(args.output),
                            "key": key,
                            "git_sha": metadata["git_sha"],
                            "script_sha256": metadata["script_sha256"],
                            "hostname": metadata["hostname"],
                            "pid": metadata["pid"],
                            "cuda_visible_devices": metadata["cuda_visible_devices"],
                            "unix_time": record["unix_time"],
                            "spec": spec,
                            "input_source": record["input_source"],
                        },
                    )
                fatal_cuda_context = False
                matching_rule = next(
                    (
                        rule
                        for rule in skip_rules
                        if all(
                            spec.get(field) == value
                            for field, value in rule["match"].items()
                        )
                    ),
                    None,
                )
                try:
                    if matching_rule is not None:
                        record.update(status="unsafe_skipped", skip_rule=matching_rule)
                    else:
                        if "COOP" in scheme and row_count > 64:
                            raise ValueError("cooperative supports at most 64 rows")
                        result = measure(
                            scheme,
                            logits,
                            lengths,
                            out,
                            k,
                            spec["max_seq_len"],
                            mode,
                            cache,
                            args,
                        )
                        record.update(status="ok", **result)
                except (RuntimeError, ValueError, ImportError, AttributeError) as exc:
                    message = str(exc)
                    unsupported = (
                        "supports at most 64" in message
                        or "was requested, but" in message
                        or "is not importable" in message
                    )
                    record.update(
                        status="unsupported" if unsupported else "error",
                        error_type=type(exc).__name__,
                        error=message,
                    )
                    if isinstance(exc, InvalidSelectionError):
                        record.update(status="invalid_selection", correctness=exc.check)
                    fatal_cuda_context = any(
                        text in message.lower()
                        for text in (
                            "illegal memory access",
                            "device-side assert",
                            "unspecified launch failure",
                        )
                    )
                    record["fatal_cuda_context"] = fatal_cuda_context
                    if not fatal_cuda_context:
                        gc.collect()
                        try:
                            torch.accelerator.empty_cache()
                        except RuntimeError as cleanup_error:
                            record["cleanup_error"] = str(cleanup_error)
                record["telemetry_after"] = telemetry()
                record["elapsed_s"] = time.monotonic() - start_time
                append_record(args.output, record)
                count += 1
                if (
                    not fatal_cuda_context
                    and args.sentinel_every
                    and count % args.sentinel_every == 0
                ):
                    sentinel_baseline = sentinel(
                        args, metadata, count, sentinel_baseline
                    )
                print(
                    json.dumps(
                        {
                            k: record.get(k)
                            for k in (
                                "status",
                                "rows",
                                "capacity",
                                "length_profile",
                                "scheme",
                                "execution",
                                "cache",
                                "bound_kind",
                                "trial",
                                "gpu_span_us",
                                "error",
                                "elapsed_s",
                            )
                        }
                    ),
                    flush=True,
                )
                if fatal_cuda_context:
                    raise SystemExit(2)
                if record["status"] in ("ok", "unsupported", "unsafe_skipped"):
                    done.add(key)
                if (
                    args.max_records is not None and count >= args.max_records
                ) or time.monotonic() - start_time >= args.max_seconds:
                    return
        del logits, lengths, out
        gc.collect()
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
