# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure real indexer-op live memory and CUDA-graph allocator ownership.

This is an operator reproduction, not a full-model memory measurement. The
breakable arm uses vLLM's actual eager-break implementation around the indexer.
Timing numbers are intentionally omitted: snapshots perturb host execution.
"""

import argparse
import contextlib
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
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
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    is_workspace_manager_initialized,
)


def emit(path, row):
    with Path(path).open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def allocator_summary():
    pools = {}
    segments = torch.cuda.memory_snapshot()
    for seg in segments:
        pool = str(seg.get("segment_pool_id", seg.get("owner_private_pool_id")))
        entry = pools.setdefault(
            pool, {"reserved": 0, "allocated": 0, "active": 0, "segments": []}
        )
        entry["reserved"] += seg["total_size"]
        entry["allocated"] += seg.get("allocated_size", 0)
        entry["active"] += seg.get("active_size", 0)
        if seg["total_size"] >= 64 * 1024 * 1024:
            blocks = []
            address = seg["address"]
            for block in seg["blocks"]:
                if block["size"] >= 64 * 1024 * 1024:
                    blocks.append(
                        dict(
                            address=block.get("address", address),
                            size=block["size"],
                            requested_size=block.get("requested_size"),
                            state=block["state"],
                            frames=block.get("frames", [])[-8:],
                        )
                    )
                address += block["size"]
            entry["segments"].append(
                dict(address=seg["address"], size=seg["total_size"], blocks=blocks)
            )
    return dict(
        allocated=torch.accelerator.memory_allocated(),
        reserved=torch.accelerator.memory_reserved(),
        peak_allocated=torch.accelerator.max_memory_allocated(),
        peak_reserved=torch.accelerator.max_memory_reserved(),
        pools=pools,
    )


def make_inputs(rows, actual_len, heads, topk):
    assert rows % 8 == 0
    generator = torch.Generator(device="cuda").manual_seed(7)
    requests, page, dim = rows // 8, 64, 128
    pages_per_req = (actual_len + page - 1) // page
    blocks = requests * pages_per_req
    # Each physical page has all quantized token values followed by scales.
    raw = torch.empty((blocks + 1, page * 68), dtype=torch.uint8, device="cuda")
    raw[:, : page * 64] = torch.randint(
        0,
        256,
        (blocks + 1, page * 64),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    raw[:, page * 64 :] = 127
    raw = raw.view(blocks + 1, page, 68)
    cache = kv_cache_as_quant_view(raw, dim, True)
    tables = torch.arange(blocks, device="cuda", dtype=torch.int32)
    tables = tables.view(requests, pages_per_req).repeat_interleave(8, dim=0)
    indices = torch.arange(requests, device="cuda", dtype=torch.int32)
    indices = indices.repeat_interleave(8)
    # Lengths are already in compressed-cache positions, like kernel inputs.
    lens = torch.full((requests, 8), actual_len, dtype=torch.int32, device="cuda")
    lens -= 7 - torch.arange(8, dtype=torch.int32, device="cuda")
    lens = lens.reshape(-1, 1)
    q = torch.randint(
        0,
        256,
        (rows, 1, heads, dim // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    ).view(torch.int8)
    scale = torch.full((rows, 1, heads), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
    weights = (
        torch.randn(
            rows, heads, dtype=torch.bfloat16, device="cuda", generator=generator
        )
        * 0.01
    )
    output = torch.full((rows, topk), -1, dtype=torch.int32, device="cuda")
    return dict(
        q=q,
        scale=scale,
        cache=cache,
        tables=tables,
        indices=indices,
        lens=lens,
        weights=weights,
        output=output,
    )


def run_case(args, rows, width, variant, mode):
    gc.collect()
    torch.accelerator.empty_cache()
    actual_len = min(args.actual_len, width)
    x = make_inputs(rows, actual_len, args.heads, args.topk)
    if args.candidate_mode != "none":
        x["candidates"] = torch.arange(2048, device="cuda", dtype=torch.int32)
        x["candidates"] = x["candidates"].repeat(rows, 1)
    backend = "deep_select" if "deep_select" in variant else "auto"
    row_bytes = width * (2 if backend == "deep_select" else 6)
    chunk_rows = rows
    if variant.startswith("chunk_"):
        chunk_rows = max(16, args.budget_mib * 1024**2 // row_bytes // 16 * 16)
    bounds = [(s, min(s + chunk_rows, rows)) for s in range(0, rows, chunk_rows)]
    sms = get_num_sms()
    full_schedule = get_paged_mqa_logits_metadata(
        x["lens"], 64, sms, indices=x["indices"]
    )
    marker = torch.zeros(1, device="cuda")
    observations = []
    capture_kind = "warmup"
    trace_allocations = False
    key = f"r{rows}_w{width}_{variant}_{mode}"
    if args.candidate_mode != "none":
        key += "_candidate_" + args.candidate_mode
    snapshots = []

    def snapshot(stage):
        path = args.snapshot_dir / f"{key}_{stage}.pickle"
        torch.cuda.memory._dump_snapshot(str(path))
        snapshots.append(str(path))

    def pipeline():
        for start, end in bounds:
            lens, idx = x["lens"][start:end], x["indices"][start:end]
            schedule = full_schedule
            if len(bounds) > 1:
                schedule = get_paged_mqa_logits_metadata(lens, 64, sms, indices=idx)
            logits = fp8_fp4_paged_mqa_logits(
                (x["q"][start:end], x["scale"][start:end]),
                x["cache"],
                x["weights"][start:end],
                lens,
                x["tables"][start:end],
                schedule,
                width,
                indices=idx,
            )
            original = logits
            if backend != "deep_select":
                logits = as_fp32_logits(logits, lens.view(-1))
            if trace_allocations:
                observations.append(
                    dict(
                        stage=capture_kind,
                        start=start,
                        end=end,
                        stream_capturing=torch.cuda.is_current_stream_capturing(),
                        bf16_ptr=original.data_ptr(),
                        bf16_dtype=str(original.dtype),
                        bf16_shape=list(original.shape),
                        bf16_stride=list(original.stride()),
                        bf16_storage_bytes=original.untyped_storage().nbytes(),
                        consumer_ptr=logits.data_ptr(),
                        consumer_dtype=str(logits.dtype),
                        consumer_storage_bytes=logits.untyped_storage().nbytes(),
                        memory=allocator_summary(),
                    )
                )
                if start == 0 and args.snapshots:
                    snapshot(capture_kind + "_conversion_overlap")
            # Do not retain the BF16 Python object beyond its natural upcast
            # overlap; the snapshot above observes that overlap explicitly.
            del original
            if args.candidate_mode == "source":
                select_candidate_blocks(
                    logits, None, lens.flatten(), 2048, 8, x["candidates"][start:end], 1
                )
            elif args.candidate_mode == "mask":
                apply_candidate_mask(
                    logits, None, lens.flatten(), x["candidates"][start:end], 8, 1
                )
            get_indexer_topk(backend)(
                logits,
                lens,
                1,
                x["output"][start:end],
                args.topk,
                width * args.compression,
            )
            del logits

    @eager_break_during_capture
    def eager_indexer():
        pipeline()

    def wrapped_model_region():
        marker.add_(1)
        eager_indexer()
        marker.add_(1)

    def selected_values():
        scores = fp8_fp4_paged_mqa_logits(
            (x["q"], x["scale"]),
            x["cache"],
            x["weights"],
            x["lens"],
            x["tables"],
            full_schedule,
            width,
            indices=x["indices"],
        )
        selected = x["output"].long()
        values = scores.gather(1, selected.clamp_min(0))
        values = torch.where(selected >= 0, values, float("-inf"))
        valid = torch.arange(actual_len, device="cuda")[None, :] < x["lens"]
        nonfinite = int((~torch.isfinite(scores[:, :actual_len]) & valid).sum())
        return values.sort(dim=1).values, nonfinite

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            pipeline()
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()
    # Warm-up output is copied once for graph replay correctness, outside the
    # measured allocation window. Compare indices from the identical backend.
    reference = x["output"].clone()
    reference_values, nonfinite = selected_values()
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    torch.cuda.memory._record_memory_history(max_entries=100000)
    torch.accelerator.reset_peak_memory_stats()
    before = allocator_summary()
    pool = torch.cuda.graph_pool_handle()
    graph = None
    capture_kind = "capture" if mode != "eager" else "eager"
    trace_allocations = True
    if mode == "full":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            pipeline()
    elif mode == "breakable":
        with torch.cuda.stream(stream):
            graph = BreakableCUDAGraphCapture(pool=pool)
            with graph:
                wrapped_model_region()
    else:
        pipeline()
    torch.accelerator.synchronize()
    after_capture = allocator_summary()
    if args.snapshots:
        snapshot("after_capture")
    trace_allocations = mode == "breakable"
    capture_kind = "replay"
    if graph is not None:
        for _ in range(2):
            graph.replay()
    else:
        pipeline()
    torch.accelerator.synchronize()
    after_replay = allocator_summary()
    equal = torch.equal(reference, x["output"])
    if args.snapshots:
        snapshot("after_replay")
    # BF16 ties can change indices without changing the selected score set.
    trace_allocations = False
    replay_values, replay_nonfinite = selected_values()
    values_equal = torch.equal(reference_values, replay_values)
    indices_valid = bool(((x["output"] >= 0) & (x["output"] < x["lens"])).all())
    indices_unique = bool((x["output"].sort(dim=1).values.diff(dim=1) != 0).all())
    result = dict(
        case_key=key,
        kind="operator_memory",
        source_commit=args.source_commit,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        host=os.uname().nodename,
        pid=os.getpid(),
        time=time.time(),
        gpu=torch.cuda.get_device_name(),
        rows=rows,
        width=width,
        actual_compressed_len=actual_len,
        input_layout="planar_unit_ue8m0_scales",
        valid_score_nonfinite_count=nonfinite,
        replay_valid_score_nonfinite_count=replay_nonfinite,
        max_seq_len_host=width * args.compression,
        compression=args.compression,
        candidate_mode=args.candidate_mode,
        heads=args.heads,
        head_dim=128,
        topk=args.topk,
        qk_dtype="mxfp4",
        variant=variant,
        backend=backend,
        mode=mode,
        chunk_rows=chunk_rows,
        chunks=len(bounds),
        pool_id=pool,
        graph_segments=getattr(graph, "num_graphs", None),
        eager_breaks=getattr(graph, "num_eager_breaks", None),
        identical_backend_replay_equal=equal,
        before=before,
        replay_selected_values_equal=values_equal,
        indices_valid=indices_valid,
        indices_unique=indices_unique,
        after_capture=after_capture,
        after_replay=after_replay,
        observations=observations,
        snapshots=snapshots,
    )
    del graph
    gc.collect()
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    result["after_graph_release"] = allocator_summary()
    torch.cuda.memory._record_memory_history(enabled=None)
    emit(args.out, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("case_key", "chunks", "identical_backend_replay_equal")
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--rows", default="512,1024,2048,4096")
    parser.add_argument("--widths", default="65536,262144")
    parser.add_argument(
        "--variants", default="auto,deep_select,chunk_auto,chunk_deep_select"
    )
    parser.add_argument("--modes", default="eager,full,breakable")
    parser.add_argument("--actual-len", type=int, default=2048)
    parser.add_argument("--compression", type=int, default=4)
    parser.add_argument(
        "--candidate-mode", choices=["none", "source", "mask"], default="none"
    )
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--budget-mib", type=int, default=512)
    parser.add_argument("--snapshots", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    args.source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda"))
    done = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            with contextlib.suppress(ValueError, KeyError):
                done.add(json.loads(line)["case_key"])
    for rows in map(int, args.rows.split(",")):
        for width in map(int, args.widths.split(",")):
            for variant in args.variants.split(","):
                for mode in args.modes.split(","):
                    key = f"r{rows}_w{width}_{variant}_{mode}"
                    if args.candidate_mode != "none":
                        key += "_candidate_" + args.candidate_mode
                    if key in done:
                        continue
                    try:
                        run_case(args, rows, width, variant, mode)
                    except Exception as exc:
                        emit(
                            args.out,
                            dict(case_key=key, error=repr(exc), time=time.time()),
                        )
                        raise


if __name__ == "__main__":
    main()
