# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare actual Qwen3-VL encoder weights with fixed packing/grid on one GPU.

Select FLASH_ATTN, FLASHINFER or TORCH_SDPA with --backend.
Synthetic patch tensors isolate the encoder; these timings are not serving TTFT.
"""

import argparse
import json
import math
import random
from pathlib import Path


def make_batching_cases():
    """Synthetic shape distributions, not an estimate of production frequencies."""
    cases = []

    def add(name, tokens, *, aspect="square", seed=42):
        grids = []
        for i, n in enumerate(tokens):
            h = math.isqrt(n)
            while n % h:
                h -= 1
            w = n // h
            if aspect == "mixed" and i % 3:
                h = max(1, h // 2)
                while n % h:
                    h -= 1
                w = n // h
                if i % 3 == 2:
                    h, w = w, h
            grids.append((1, 2 * h, 2 * w))
        cases.append(dict(name=name, tokens=tokens, grid=grids, seed=seed))

    add("small-256", [256])
    for total in (1024, 4096, 8192, 16384):
        for size in (256, 1024, 4096):
            if total >= size:
                add(f"uniform-{total // size}x{size}", [size] * (total // size))
    add("mixed-balanced-16K", [768] * 8 + [1280] * 8)
    add("tail-4K-many-small", [4096] + [256] * 48)
    add("tail-8K-many-small", [8192] + [256] * 32)
    add("oversize-9216", [9216])
    add("oversize-16384", [16384])
    add("cliff-1024+512", [1024, 512])
    add("boundary-4096-minus64", [1024] * 3 + [960])
    add("boundary-4096-plus64", [1024] * 4 + [64])
    add("boundary-8192-minus64", [1024] * 7 + [960])
    add("boundary-8192-plus64", [1024] * 8 + [64])
    add("boundary-16384-minus64", [1024] * 15 + [960])
    add("aspect-mixed-16x1024", [1024] * 16, aspect="mixed")

    palette = [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    for total in (4096, 8192, 16384):
        for family in ("small-heavy", "bimodal", "long-tail"):
            for seed in (17, 83):
                rng = random.Random(seed)
                remaining, tokens = total, []
                while remaining:
                    slots = 64 - len(tokens)
                    lo = max(64, remaining - (slots - 1) * 4096)
                    hi = min(remaining, 4096)
                    choices = [x for x in palette if lo <= x <= hi]
                    if not choices or slots == 1:
                        n = remaining
                    else:
                        if family == "small-heavy":
                            weights = [8 if x <= 512 else 1 for x in choices]
                        elif family == "bimodal":
                            weights = [
                                6 if x <= 256 or x >= 2048 else 1 for x in choices
                            ]
                        else:
                            weights = [4096 / x for x in choices]
                        n = rng.choices(choices, weights=weights)[0]
                    tokens.append(n)
                    remaining -= n
                rng.shuffle(tokens)
                add(f"{family}-{total}-s{seed}", tokens, aspect="mixed", seed=seed)
    return cases


def make_threshold_cases():
    cases = []
    for total in (1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 8192):
        layouts = {
            "single": [total],
            "small": [256] * (total // 256),
            "medium": [512] * (total // 512),
            "skewed": [total // 2] + [128] * (total // 256),
        }
        for name, tokens in layouts.items():
            grid = []
            for n in tokens:
                h = math.isqrt(n)
                while n % h:
                    h -= 1
                grid.append((1, h * 2, n // h * 2))
            cases.append(
                dict(
                    name=f"threshold-{total}-{name}", tokens=tokens, grid=grid, seed=91
                )
            )
    return cases


def run_batching_benchmark(worker, options):
    """Compare whole-batch eager with the existing FULL packing policy."""
    import gc
    import statistics
    import time
    from unittest.mock import patch

    import torch

    from vllm.distributed.parallel_state import graph_capture
    from vllm.multimodal.encoder_budget import MultiModalBudget
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    model = worker.model_runner.get_model()
    config = worker.vllm_config
    device = next(model.parameters()).device
    cases = (
        make_threshold_cases()
        if options["case_set"] == "threshold"
        else make_batching_cases()
    )
    if options["case_set"] == "large_capture":
        names = {
            "uniform-8x1024",
            "uniform-16x1024",
            "uniform-64x256",
            "mixed-balanced-16K",
            "long-tail-8192-s17",
            "bimodal-16384-s83",
            "oversize-9216",
        }
        cases = [case for case in cases if case["name"] in names]
    budgets = [64 * 2**i for i in range(9) if 64 * 2**i <= options["capture_cap"]]
    if not budgets or budgets[-1] != options["capture_cap"]:
        raise ValueError("Batching capture cap must be a power of two from 64 to 16384")
    mm_budget = MultiModalBudget(
        config, worker.model_runner.mm_registry, enable_cache=False
    )
    config.compilation_config.encoder_cudagraph_token_budgets = budgets
    config.compilation_config.encoder_cudagraph_oversize_policy = options[
        "oversize_policy"
    ]
    config.compilation_config.encoder_cudagraph_max_vision_items_per_batch = 64
    manager = EncoderCudaGraphManager(config, device, torch.bfloat16, model)
    pool = torch.cuda.graph_pool_handle()

    def save(row):
        with open(options["output"], "a") as f:
            f.write(json.dumps(row) + "\n")

    def memory():
        torch.accelerator.synchronize()
        free, total = torch.accelerator.get_memory_info()
        return dict(
            device_used=total - free,
            allocated=torch.accelerator.memory_allocated(),
            reserved=torch.accelerator.memory_reserved(),
            peak_allocated=torch.accelerator.max_memory_allocated(),
            peak_reserved=torch.accelerator.max_memory_reserved(),
        )

    gc.collect()
    torch.accelerator.empty_cache()
    before = memory()
    with torch.inference_mode(), graph_capture(device=device):
        manager.capture(pool)
    after = memory()
    segments = [
        s
        for s in torch.cuda.memory_snapshot()
        if tuple(s.get("segment_pool_id", ())) == tuple(pool)
    ]
    save(
        dict(
            event="setup",
            options=options,
            budgets=budgets,
            encoder_compute_budget=mm_budget.encoder_compute_budget,
            encoder_cache_size=mm_budget.encoder_cache_size,
            model=config.model_config.model,
            gpu=torch.cuda.get_device_name(),
            before=before,
            after=after,
            graph_pool_reserved=sum(s["total_size"] for s in segments),
            graph_count=len(manager.budget_graphs["default"]),
            cases=cases,
            memory_scope="Both arms retain the same captured graph pool and 8 GiB KV.",
        )
    )
    patch_embed = model.visual.patch_embed
    width = (
        patch_embed.proj.in_channels
        * patch_embed.temporal_patch_size
        * patch_embed.patch_size**2
    )
    input_dtype = model.visual.input_norm.input_dtype or torch.bfloat16
    case_order = list(range(len(cases)))
    random.Random(options["case_order_seed"]).shuffle(case_order)

    with torch.inference_mode():
        for case_index in case_order:
            case = cases[case_index]
            tokens, grid = case["tokens"], case["grid"]
            if sum(tokens) > mm_budget.get_encoder_budget():
                save(dict(event="excluded", case=case, reason="encoder budget"))
                continue
            gen = torch.Generator(device=device).manual_seed(case["seed"])
            shape = (sum(t * h * w for t, h, w in grid), width)
            pixels = (
                torch.randint(
                    0, 256, shape, device=device, dtype=input_dtype, generator=gen
                )
                if input_dtype == torch.uint8
                else torch.randn(shape, device=device, dtype=input_dtype, generator=gen)
            )
            mm = {"pixel_values": pixels, "image_grid_thw": grid}

            def eager(mm=mm, tokens=tokens):
                return model.encoder_eager_forward(mm).split(tokens)

            def replay(mm=mm):
                return manager.execute(mm)

            def threshold_replay(mm=mm):
                original_policy = manager.oversize_policy
                original_budgets = manager.path_token_budgets
                manager.oversize_policy = "eager"
                manager.path_token_budgets = {
                    "default": [b for b in budgets if b <= options["compare_threshold"]]
                }
                try:
                    return manager.execute(mm)
                finally:
                    manager.oversize_policy = original_policy
                    manager.path_token_budgets = original_budgets

            calls = []
            original_replay = manager._run_budget_graph
            original_eager = model.encoder_eager_forward

            def traced_replay(
                batch, budget, *, calls=calls, original_replay=original_replay, **kwargs
            ):
                specs = model.get_encoder_cudagraph_item_specs(batch)
                calls.append(
                    dict(
                        mode="FULL",
                        budget=budget,
                        tokens=[s.output_tokens for s in specs],
                    )
                )
                return original_replay(batch, budget, **kwargs)

            def traced_eager(
                batch, *, calls=calls, original_eager=original_eager, **kwargs
            ):
                specs = model.get_encoder_cudagraph_item_specs(batch)
                calls.append(
                    dict(
                        mode="eager",
                        budget=None,
                        tokens=[s.output_tokens for s in specs],
                    )
                )
                return original_eager(batch, **kwargs)

            expected = eager()
            with (
                patch.object(manager, "_run_budget_graph", traced_replay),
                patch.object(model, "encoder_eager_forward", traced_eager),
            ):
                actual = (
                    threshold_replay() if options.get("compare_threshold") else replay()
                )
            assert len(actual) == len(expected) == len(tokens)
            errors = [(a.float() - b.float()).abs() for a, b in zip(actual, expected)]
            check = dict(
                finite=all(bool(torch.isfinite(a).all()) for a in actual),
                equal=all(torch.equal(a, b) for a, b in zip(actual, expected)),
                close=all(
                    torch.allclose(a, b, rtol=0.01, atol=0.01)
                    for a, b in zip(actual, expected)
                ),
                max_abs=max(float(e.max()) for e in errors),
            )
            save(dict(event="correctness", case=case["name"], check=check, calls=calls))
            if not check["finite"] or not check["close"]:
                raise AssertionError(f"{case['name']}: {check}")
            del expected, actual, errors

            funcs = {"whole-eager": eager, "current-replay": replay}
            if options.get("compare_threshold"):
                funcs = {"current-replay": replay, "threshold-policy": threshold_replay}
            for fn in funcs.values():
                for _ in range(options["warmup"]):
                    fn()
            torch.accelerator.synchronize()
            for block in range(options["blocks"]):
                arms = list(funcs)
                if (block + case_index) % 2:
                    arms.reverse()
                for arm in arms:
                    gpu_ms, wall_ms = [], []
                    for _ in range(options["repeats"]):
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        ts = time.perf_counter()
                        start.record()
                        outputs = funcs[arm]()
                        end.record()
                        end.synchronize()
                        gpu_ms.append(start.elapsed_time(end))
                        wall_ms.append((time.perf_counter() - ts) * 1000)
                        del outputs
                    save(
                        dict(
                            event="timing",
                            case=case["name"],
                            arm=arm,
                            block=block,
                            gpu_ms=gpu_ms,
                            wall_ms=wall_ms,
                            gpu_median_ms=statistics.median(gpu_ms),
                            wall_median_ms=statistics.median(wall_ms),
                        )
                    )

            for arm, fn in funcs.items():
                gc.collect()
                torch.accelerator.empty_cache()
                torch.accelerator.reset_peak_memory_stats()
                mem_before = memory()
                outputs = fn()
                mem_after = memory()
                save(
                    dict(
                        event="memory",
                        case=case["name"],
                        arm=arm,
                        before=mem_before,
                        after=mem_after,
                    )
                )
                del outputs
            del pixels, mm, eager, replay, funcs, fn
            print(
                f"BATCHING case={case['name']} equal={check['equal']} "
                f"calls={len(calls)}",
                flush=True,
            )
    manager.clear()
    save(dict(event="complete", num_cases=len(case_order)))
    return {"output": options["output"]}


def run_encoder_benchmark(worker, options):
    import gc
    import statistics
    import time

    import torch

    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.compilation.encoder_cudagraph import encoder_attention_context
    from vllm.distributed.parallel_state import graph_capture
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    model = worker.model_runner.get_model()
    config = worker.vllm_config
    device = next(model.parameters()).device
    dtype = torch.bfloat16
    budgets = [64 * 2**i for i in range(9)]
    config.compilation_config.encoder_cudagraph_token_budgets = budgets
    config.compilation_config.encoder_cudagraph_max_vision_items_per_batch = 64
    patch = model.visual.patch_embed
    width = patch.proj.in_channels * patch.temporal_patch_size * patch.patch_size**2
    cases = {
        "256": [(1, 32, 32)],
        "1024": [(1, 64, 64)],
        "9216": [(1, 192, 192)],
        "16384": [(1, 256, 256)],
        "1024+512": [(1, 64, 64), (1, 32, 64)],
        "16x1024": [(1, 64, 64)] * 16,
        "512+1024": [(1, 32, 64), (1, 64, 64)],
        "256-again": [(1, 32, 32)],
    }
    rows = []

    def memory(pool=None):
        torch.accelerator.synchronize()
        free, _ = torch.accelerator.get_memory_info()
        result = dict(
            free=free,
            allocated=torch.accelerator.memory_allocated(),
            reserved=torch.accelerator.memory_reserved(),
            peak_allocated=torch.accelerator.max_memory_allocated(),
        )
        if pool is not None:
            segments = [
                s
                for s in torch.cuda.memory_snapshot()
                if tuple(s.get("segment_pool_id", ())) == tuple(pool)
            ]
            result["graph_pool_reserved"] = sum(s["total_size"] for s in segments)
            result["graph_pool_active"] = sum(s["active_size"] for s in segments)
        return result

    def save(row):
        rows.append(row)
        with open(options["output"], "a") as f:
            f.write(json.dumps(row) + "\n")

    arms = options["arms"] or (
        ["eager", "full", "breakable"]
        if options["backend"] != "TORCH_SDPA"
        else ["eager", "breakable"]
    )
    with torch.inference_mode():
        for arm in arms:
            gc.collect()
            torch.accelerator.empty_cache()
            torch.accelerator.reset_peak_memory_stats()
            before = memory()
            manager = None
            pool = None
            if arm not in ("eager", "context-eager"):
                manager = EncoderCudaGraphManager(
                    config, device, dtype, model, use_breakable=arm == "breakable"
                )
                pool = torch.cuda.graph_pool_handle()
                start = time.perf_counter()
                with graph_capture(device=device):
                    manager.capture(pool)
                    torch.accelerator.synchronize()
                entries = list(manager.budget_graphs["default"].values())
                storages = {}
                for entry in entries:
                    for tensor in [*entry.input_buffers.values(), entry.output_buffer]:
                        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                            storage = tensor.untyped_storage()
                            storages[storage.data_ptr()] = storage.nbytes()
                save(
                    dict(
                        event="capture",
                        arm=arm,
                        budgets=budgets,
                        seconds=time.perf_counter() - start,
                        before=before,
                        after=memory(pool),
                        persistent_io_bytes=sum(storages.values()),
                        graph_segments=sum(
                            e.graph.num_graphs
                            if isinstance(e.graph, BreakableCUDAGraphCapture)
                            else 1
                            for e in entries
                        ),
                        eager_breaks=sum(
                            e.graph.num_eager_breaks
                            if isinstance(e.graph, BreakableCUDAGraphCapture)
                            else 0
                            for e in entries
                        ),
                    )
                )
                del entries, entry, tensor, storage, storages
            for case, grid in cases.items():
                torch.manual_seed(42)
                input_dtype = model.visual.input_norm.input_dtype or dtype
                shape = (sum(t * h * w for t, h, w in grid), width)
                pv = (
                    torch.randint(0, 256, shape, device=device, dtype=input_dtype)
                    if input_dtype == torch.uint8
                    else torch.randn(shape, device=device, dtype=input_dtype)
                )
                mm = {"pixel_values": pv, "image_grid_thw": grid}

                def forward(manager=manager, mm=mm, arm=arm):
                    if arm == "context-eager":
                        replay = model.prepare_encoder_cudagraph_replay_buffers(
                            mm, max_batch_size=64, max_frames_per_batch=0
                        )
                        with encoder_attention_context(replay.attention_metadata):
                            return model.encoder_cudagraph_forward(dict(replay.values))
                    if manager is None:
                        return model.encoder_eager_forward(mm)
                    return torch.cat(manager.execute(mm))

                reference = model.encoder_eager_forward(mm)
                output = forward()
                error = (reference.float() - output.float()).abs()
                check = dict(
                    finite=bool(torch.isfinite(output).all()),
                    equal=bool(torch.equal(reference, output)),
                    max_abs=float(error.max()),
                    mean_abs=float(error.mean()),
                    close=bool(torch.allclose(reference, output, rtol=0.01, atol=0.01)),
                )
                del reference, output, error
                torch.accelerator.reset_peak_memory_stats()
                for _ in range(options["warmup"]):
                    forward()
                torch.accelerator.synchronize()
                gpu_ms, wall_ms = [], []
                for _ in range(options["repeats"]):
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    ts = time.perf_counter()
                    start.record()
                    forward()
                    end.record()
                    end.synchronize()
                    gpu_ms.append(start.elapsed_time(end))
                    wall_ms.append((time.perf_counter() - ts) * 1000)
                save(
                    dict(
                        event="case",
                        arm=arm,
                        case=case,
                        grid=grid,
                        correctness=check,
                        gpu_ms=gpu_ms,
                        wall_ms=wall_ms,
                        gpu_median_ms=statistics.median(gpu_ms),
                        wall_median_ms=statistics.median(wall_ms),
                        graph_hits=manager.graph_hits if manager else 0,
                        graph_misses=manager.graph_misses if manager else 0,
                        memory=memory(pool),
                    )
                )
                if not check["finite"] or not check["close"]:
                    raise AssertionError(f"{arm} {case}: {check}")
                del pv, mm, forward
            if manager is not None:
                manager.clear()
            del manager, pool
            gc.collect()
            torch.accelerator.empty_cache()
            save(dict(event="released", arm=arm, memory=memory()))
    return rows


class EncoderBenchmarkWorkerExtension:
    def benchmark_encoder(self, options):
        if options["suite"] == "batching":
            return run_batching_benchmark(self, options)
        return run_encoder_benchmark(self, options)


def main():
    from vllm import LLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--backend", choices=["FLASH_ATTN", "FLASHINFER", "TORCH_SDPA"], required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--suite", choices=["breakable", "batching"], default="breakable"
    )
    parser.add_argument("--capture-cap", type=int, default=4096)
    parser.add_argument("--compare-threshold", type=int, choices=[1024, 2048, 4096])
    parser.add_argument(
        "--oversize-policy", choices=["split", "eager"], default="split"
    )
    parser.add_argument(
        "--case-set",
        choices=["distributions", "threshold", "large_capture"],
        default="distributions",
    )
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--case-order-seed", type=int, default=0)
    parser.add_argument(
        "--arms", nargs="+", choices=["eager", "context-eager", "full", "breakable"]
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        worker_extension_cls=(
            "benchmarks.multimodal.benchmark_encoder_breakable."
            "EncoderBenchmarkWorkerExtension"
        ),
        model=args.model,
        dtype="bfloat16",
        max_model_len=32768,
        max_num_batched_tokens=8192,
        max_num_seqs=8,
        kv_cache_memory_bytes=8 * 1024**3,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={
            "image": 64 if args.suite == "batching" else 16,
            "video": 0,
        },
        mm_encoder_attn_backend=args.backend,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "NONE",
            "cudagraph_mm_encoder": False,
        },
    )
    llm.collective_rpc("benchmark_encoder", args=(vars(args),))


if __name__ == "__main__":
    main()
