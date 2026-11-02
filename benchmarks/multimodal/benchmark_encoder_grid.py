# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure Qwen3-VL encoder capture grids against whole-batch eager.

All policies share a captured superset to isolate dispatch latency. Measure
deployment memory separately with only the selected capture grid installed.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from benchmarks.multimodal.benchmark_encoder_breakable import make_batching_cases

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def grid_sizes(cap: int, family: str) -> list[int]:
    sizes = [64, 128, 256, 512]
    if family == "powers":
        sizes += [1024, 2048, 4096, 8192, 16384]
    elif family == "quarter":
        for lower in (512, 1024, 2048, 4096, 8192):
            sizes += [lower + lower // 4 * i for i in range(1, 5)]
    elif family in ("linear", "hybrid"):
        sizes += list(range(768, 2049, 256))
        sizes += list(range(2560, 8193, 512))
        sizes += list(range(9216, 16385, 1024))
        if family == "hybrid":
            sizes += [640, 896]
    elif family == "uniform":
        sizes += list(range(768, 16385, 256))
    else:
        raise ValueError(family)
    return sorted({s for s in sizes if s <= cap} | {cap})


def make_cases(suite: str) -> list[dict]:
    cases = []

    if suite == "video":
        for index, grids in enumerate(
            (
                [[2, 32, 32]],
                [[4, 32, 32], [2, 32, 16]],
                [[8, 32, 32]] * 4,
                [[16, 32, 64]] * 2,
                [[1, 32, 32], [3, 32, 48]],
            )
        ):
            cases.append(
                dict(
                    name=f"video-{index}",
                    tokens=[t * h * w // 4 for t, h, w in grids],
                    grid=grids,
                    seed=7103 + index,
                    modality="video",
                )
            )
        return cases

    def add(name, tokens, seed):
        grids = []
        for i, n in enumerate(tokens):
            h = math.isqrt(n)
            while n % h:
                h -= 1
            w = n // h
            if i % 2:
                h, w = w, h
            grids.append([1, h * 2, w * 2])
        cases.append(dict(name=name, tokens=tokens, grid=grids, seed=seed))

    if suite == "items":
        for count in (1, 2, 32, 64):
            add(f"items-{count}", [64] * count, 7313)
        return cases + make_cases("edges")

    if suite == "sweep":
        for total in (
            256,
            512,
            768,
            1024,
            1536,
            2048,
            2304,
            2560,
            3072,
            3584,
            4096,
            5120,
            6144,
            7168,
            8192,
            10240,
            12288,
            16384,
        ):
            add(f"single-{total}", [total], 103)
            add(f"small-{total}", [256] * (total // 256), 103)
            tail_size = 128 if total <= 8192 else 256
            add(
                f"skewed-{total}",
                [total // 2] + [tail_size] * (total // 2 // tail_size),
                103,
            )
        return cases
    if suite == "previous":
        return make_batching_cases()
    if suite == "edges":
        for count in (128, 129, 256):
            add(f"many-{count}x64", [64] * count, 3307)
        add("mixed-65-items", [4096] + [64] * 64, 3307)
        add("mixed-129-items", [8192] + [64] * 128, 3307)
        return cases
    if suite == "irregular":
        rng = random.Random(2807)
        for limit in (1024, 4096, 8192, 16384):
            for index in range(8):
                tokens = []
                remaining = limit
                while remaining >= 64 and len(tokens) < 32:
                    h = rng.randint(8, min(64, remaining // 8))
                    w = rng.randint(8, min(64, remaining // h))
                    tokens.append(h * w)
                    remaining -= h * w
                    if index % 3 == 0:
                        break
                rng.shuffle(tokens)
                add(f"irregular-{limit}-{index}", tokens, 2807 + index)
        return cases
    if suite != "heldout":
        raise ValueError(suite)
    rng = random.Random(1907)
    totals = [1984, 2112, 4032, 4160, 6080, 6208, 8128, 8256, 16320]
    totals += rng.sample(list(range(256, 16385, 64)), 23)
    palette = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    for index, total in enumerate(totals):
        remaining, tokens = total, []
        while remaining:
            slots = 64 - len(tokens)
            candidates = [
                n
                for n in palette
                if max(64, remaining - (slots - 1) * 4096) <= n <= remaining
            ]
            n = remaining if not candidates or slots == 1 else rng.choice(candidates)
            tokens.append(n)
            remaining -= n
        rng.shuffle(tokens)
        add(f"heldout-{index}-{total}", tokens, 1907 + index)
    return cases


def configurations():
    result = [
        dict(name="old-default-8K", sizes=grid_sizes(8192, "powers"), policy="split")
    ]
    for family in ("powers", "quarter", "linear", "uniform", "hybrid"):
        for cap in (1024, 2048, 4096, 6144, 8192, 12288, 16384):
            result.append(
                dict(
                    name=f"{family}-{cap}",
                    sizes=grid_sizes(cap, family),
                    policy="eager",
                )
            )
    result.append(
        dict(name="hybrid-8192-split", sizes=grid_sizes(8192, "hybrid"), policy="split")
    )
    return result


def get_plan(manager, specs, config):
    from unittest.mock import patch

    calls = []
    manager.path_token_budgets = {"default": config["sizes"]}
    manager.oversize_policy = config["policy"]

    def graph(mm, budget, **kwargs):
        calls.append(("graph", tuple(mm["indices"]), budget))
        return object()

    def eager(mm, **kwargs):
        calls.append(("eager", tuple(mm["indices"]), None))
        return object()

    def postprocess(outputs, indices, sizes, dest, **kwargs):
        dest.update((i, None) for i in indices)

    with (
        patch.object(
            manager, "_get_item_specs", lambda mm: [specs[i] for i in mm["indices"]]
        ),
        patch.object(
            manager, "_select_items", lambda mm, indices: ({"indices": indices}, ())
        ),
        patch.object(manager, "_run_budget_graph", graph),
        patch.object(manager.model, "encoder_eager_forward", eager),
        patch.object(manager.model, "postprocess_encoder_output", postprocess),
    ):
        manager._execute_local({"indices": list(range(len(specs)))})
    return tuple(calls)


def run_grid_benchmark(worker, options):
    import gc
    import os
    import socket
    import time
    import uuid

    import torch

    from vllm.distributed.parallel_state import graph_capture
    from vllm.multimodal.encoder_budget import MultiModalBudget
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    model = worker.model_runner.get_model()
    config = worker.vllm_config
    device = next(model.parameters()).device
    cases = make_cases(options["suite"])
    if options.get("only_cases"):
        names = set(options["only_cases"].split(","))
        cases = [c for c in cases if c["name"] in names]
        if names != {c["name"] for c in cases}:
            raise ValueError("Unknown case name")
    variants = configurations()
    if options.get("configs"):
        names = set(options["configs"].split(","))
        variants = [v for v in variants if v["name"] in names]
        if names != {v["name"] for v in variants}:
            raise ValueError("Unknown grid configuration")
    max_items = options.get("max_items", 64)
    for variant in variants:
        variant["sizes"] = [s for s in variant["sizes"] if s >= max_items]
    sizes = sorted(
        {n for v in variants for n in v["sizes"]}
        | (
            set()
            if options["memory_only"]
            else {sum(c["tokens"]) for c in cases if sum(c["tokens"]) >= max_items}
        )
    )
    config.compilation_config.encoder_cudagraph_token_budgets = sizes
    config.compilation_config.encoder_cudagraph_max_vision_items_per_batch = max_items
    if options["suite"] == "video":
        config.compilation_config.encoder_cudagraph_max_frames_per_batch = 64
    manager = EncoderCudaGraphManager(config, device, torch.bfloat16, model)
    budget = MultiModalBudget(
        config, worker.model_runner.mm_registry, enable_cache=False
    )
    pool = torch.cuda.graph_pool_handle()
    boot = str(uuid.uuid4())

    def save(row):
        with open(options["output"], "a") as f:
            f.write(json.dumps(dict(boot=boot, **row)) + "\n")

    def memory():
        torch.accelerator.synchronize()
        free, total = torch.accelerator.get_memory_info()
        return dict(
            device_used=total - free,
            allocated=torch.accelerator.memory_allocated(),
            reserved=torch.accelerator.memory_reserved(),
        )

    gc.collect()
    torch.accelerator.empty_cache()
    before = memory()
    started = time.perf_counter()
    with torch.inference_mode(), graph_capture(device=device):
        manager.capture(pool)
    after = memory()
    segments = [
        s
        for s in torch.cuda.memory_snapshot()
        if tuple(s.get("segment_pool_id", ())) == tuple(pool)
    ]
    storage_bytes = {"input": {}, "output": {}}
    for entries in manager.budget_graphs.values():
        for entry in entries.values():
            for kind, tensors in (
                ("input", entry.input_buffers.values()),
                ("output", [entry.output_buffer]),
            ):
                for tensor in tensors:
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        storage = tensor.untyped_storage()
                        storage_bytes[kind][storage.data_ptr()] = storage.nbytes()
    save(
        dict(
            event="setup",
            options=options,
            source_sha256=SOURCE_SHA256,
            host=socket.gethostname(),
            job=os.environ.get("SLURM_JOB_ID"),
            sizes=sizes,
            variants=variants,
            cases=cases,
            before=before,
            after=after,
            capture_seconds=time.perf_counter() - started,
            encoder_budget=budget.get_encoder_budget(),
            persistent_buffers={k: sum(v.values()) for k, v in storage_bytes.items()},
            pool_reserved=sum(s["total_size"] for s in segments),
            pool_active=sum(s["active_size"] for s in segments),
            gpu=torch.cuda.get_device_name(),
            model=config.model_config.model,
            note=(
                "Independent deployment grid."
                if options["memory_only"]
                else "Shared capture superset for latency, not deployment memory."
            ),
        )
    )
    if options["memory_only"]:
        save(dict(event="complete", cases=0))
        return
    patch_embed = model.visual.patch_embed
    width = (
        patch_embed.proj.in_channels
        * patch_embed.temporal_patch_size
        * patch_embed.patch_size**2
    )
    dtype = model.visual.input_norm.input_dtype or torch.bfloat16
    rng = random.Random(options["order_seed"])
    rng.shuffle(cases)

    with torch.inference_mode():
        for case in cases:
            total = sum(case["tokens"])
            if total > budget.get_encoder_budget():
                raise ValueError(
                    f"Case exceeds encoder admission budget: {case['name']}"
                )
            generator = torch.Generator(device=device).manual_seed(case["seed"])
            pixels = torch.randint(
                0,
                256,
                (total * 4, width),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            mm = (
                {"pixel_values_videos": pixels, "video_grid_thw": case["grid"]}
                if case.get("modality") == "video"
                else {"pixel_values": pixels, "image_grid_thw": case["grid"]}
            )
            specs = model.get_encoder_cudagraph_item_specs(mm)
            grouped = {}
            for variant in variants:
                plan = get_plan(manager, specs, variant)
                if plan not in grouped:
                    grouped[plan] = dict(config=variant, names=[], plan=plan)
                grouped[plan]["names"].append(variant["name"])
            groups = list(grouped.values())
            groups += [
                dict(
                    config=None,
                    names=["whole-eager"],
                    plan=(("eager", tuple(range(len(specs))), None),),
                ),
            ]
            if len(specs) <= manager.max_batch_size and total >= max_items:
                groups.append(
                    dict(
                        config=None,
                        names=["exact-single-graph"],
                        plan=(("graph", tuple(range(len(specs))), total),),
                    )
                )

            def finish(output, tokens=case["tokens"], mm=mm):
                dest = {}
                model.postprocess_encoder_output(
                    {"default": output},
                    list(range(len(tokens))),
                    tokens,
                    dest,
                    clone=True,
                    batch_mm_kwargs=mm,
                )
                return [dest[i] for i in range(len(tokens))]

            def eager(mm=mm, finish=finish):
                return finish(model.encoder_eager_forward(mm))

            def exact_graph(mm=mm, total=total, finish=finish):
                return finish(manager._run_budget_graph(mm, total))

            def select(group, mm=mm, eager=eager, exact_graph=exact_graph):
                if group["config"] is not None:
                    manager.path_token_budgets = {"default": group["config"]["sizes"]}
                    manager.oversize_policy = group["config"]["policy"]
                    return lambda: manager.execute(mm)
                return eager if group["names"] == ["whole-eager"] else exact_graph

            reference = eager()
            checks = []
            for group in groups:
                output = select(group)()
                finite = all(bool(torch.isfinite(x).all()) for x in output)
                equal = all(torch.equal(a, b) for a, b in zip(reference, output))
                close = len(output) == len(reference) and all(
                    torch.allclose(a, b, rtol=0.01, atol=0.01)
                    for a, b in zip(reference, output)
                )
                error = max(
                    float((a.float() - b.float()).abs().max())
                    for a, b in zip(reference, output)
                )
                checks.append(
                    dict(
                        names=group["names"],
                        equal=equal,
                        close=close,
                        finite=finite,
                        max_abs=error,
                    )
                )
                if not finite or not close:
                    save(dict(event="correctness_failure", case=case, checks=checks))
                    raise AssertionError(checks[-1])
                del output
            del reference
            for group in groups:
                fn = select(group)
                for _ in range(options["warmup"]):
                    fn()
            torch.accelerator.synchronize()
            samples = []
            for block in range(options["blocks"]):
                order = list(range(len(groups)))
                rng.shuffle(order)
                for index in order:
                    group = groups[index]
                    fn = select(group)
                    times = []
                    for _ in range(options["repeats"]):
                        begin, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        t0 = time.perf_counter()
                        begin.record()
                        output = fn()
                        submitted = time.perf_counter()
                        end.record()
                        end.synchronize()
                        times.append(
                            dict(
                                host_ms=(submitted - t0) * 1000,
                                wall_ms=(time.perf_counter() - t0) * 1000,
                                stream_ms=begin.elapsed_time(end),
                            )
                        )
                        del output
                    samples.append(
                        dict(
                            names=group["names"],
                            plan=group["plan"],
                            block=block,
                            times=times,
                        )
                    )
            save(dict(event="case", case=case, checks=checks, samples=samples))
            if options.get("profile_cases") and case["name"] in options[
                "profile_cases"
            ].split(","):
                for group in groups:
                    fn = select(group)
                    fn()
                    torch.accelerator.synchronize()
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                    ) as prof:
                        output = fn()
                        torch.accelerator.synchronize()
                    path = (
                        f"{options['output']}.{case['name']}."
                        f"{group['names'][0]}.trace.json"
                    )
                    prof.export_chrome_trace(path)
                    del output, prof
            print(
                f"GRID {case['name']} unique_plans={len(groups)} "
                f"equal={all(c['equal'] for c in checks)}",
                flush=True,
            )
            del fn, mm, pixels, groups, grouped, specs
            del finish, eager, exact_graph, select
    manager.clear()
    save(dict(event="complete", cases=len(cases)))


class EncoderGridWorkerExtension:
    def benchmark_encoder_grid(self, options):
        return run_grid_benchmark(self, options)


def main():
    from vllm import LLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--backend", choices=["FLASHINFER", "FLASH_ATTN"], required=True
    )
    parser.add_argument(
        "--suite",
        choices=[
            "sweep",
            "heldout",
            "irregular",
            "previous",
            "edges",
            "video",
            "items",
        ],
        default="sweep",
    )
    parser.add_argument("--configs")
    parser.add_argument("--max-items", type=int, default=64)
    parser.add_argument("--only-cases")
    parser.add_argument("--profile-cases")
    parser.add_argument("--compile-encoder", action="store_true")
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--order-seed", type=int, default=1103)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=32768,
        max_num_batched_tokens=8192,
        max_num_seqs=8,
        kv_cache_memory_bytes=8 * 1024**3,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 64, "video": 4 if args.suite == "video" else 0},
        mm_encoder_attn_backend=args.backend,
        compilation_config={
            "mode": 3 if args.compile_encoder else 0,
            "cudagraph_mode": "NONE",
            "cudagraph_mm_encoder": False,
            "compile_mm_encoder": args.compile_encoder,
        },
        worker_extension_cls="benchmarks.multimodal.benchmark_encoder_grid.EncoderGridWorkerExtension",
    )
    llm.collective_rpc("benchmark_encoder_grid", args=(vars(args),))


if __name__ == "__main__":
    main()
