# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate buffer capacity and attention bounds on two encoder regressions.

The specialized graphs are diagnostic controls for these exact inputs, not
proposed dispatch defaults. Real weights, preprocessing, and NaN guards remain
unchanged. Run with the sibling cross-model harness on PYTHONPATH.
"""

import argparse
import hashlib
import json
import os
import random
import socket
import statistics
import time
from pathlib import Path

import torch
from benchmark_encoder_crossmodel import MODELS, build_model, compare, prepare_cases


def main():
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import graph_capture
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.network_utils import get_open_port
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save(event, **values):
        row = dict(event=event, time=time.time(), **values)
        with output.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(event, values.get("case", ""), values.get("scope", ""), flush=True)

    config = EngineArgs(
        model=args.model_path,
        dtype="bfloat16",
        max_model_len=32768,
        max_num_batched_tokens=16384,
        max_num_seqs=256,
        enforce_eager=True,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 256, "video": 0},
        mm_encoder_attn_backend="FLASH_ATTN",
        mm_encoder_tp_mode="data",
        compilation_config={"mode": 0, "compile_mm_encoder": False},
    ).create_engine_config()
    torch.accelerator.set_device_index(0)
    torch.set_default_dtype(torch.bfloat16)
    device = torch.device("cuda:0")
    with set_current_vllm_config(config), torch.inference_mode():
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
        with torch.device(device):
            model, weights = build_model("deepseek", config, Path(args.model_path))
        save(
            "loaded",
            model=MODELS["deepseek"],
            weights=weights,
            gpu=torch.cuda.get_device_name(),
            host=socket.gethostname(),
            job=os.getenv("SLURM_JOB_ID"),
            torch_version=torch.__version__,
            options=vars(args),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            attention_versions=sorted(
                {
                    str(m._fa_version)
                    for m in model.modules()
                    if hasattr(m, "_fa_version")
                }
            ),
        )
        cases, _ = prepare_cases(
            "deepseek",
            model,
            Path(args.model_path),
            device,
            16384,
            False,
            "threshold-holdout",
        )
        cases = [c for c in cases if c["name"] in ("heldout-13-0", "heldout-13-1")]
        cc = config.compilation_config
        cc.encoder_cudagraph_token_budgets = [2688, 2816]
        cc.encoder_cudagraph_max_vision_items_per_batch = 128
        manager = EncoderCudaGraphManager(config, device, torch.bfloat16, model)
        pool = torch.cuda.graph_pool_handle()
        with graph_capture(device=device):
            manager.capture(pool)

        rng = random.Random(2816)
        for case in cases:
            data, tokens = case["data"], case["tokens"]
            budget = 2688 if sum(tokens) == 2636 else 2816
            entry = manager.budget_graphs["default"][budget]
            original_max = entry.input_buffers["patches"].shape[0]
            actual = model.prepare_encoder_cudagraph_replay_buffers(data, 128, 0).values

            def post(tensor, data=data, tokens=tokens):
                dest = {}
                model.postprocess_encoder_output(
                    {"default": tensor},
                    list(range(len(tokens))),
                    tokens,
                    dest,
                    clone=True,
                    batch_mm_kwargs=data,
                )
                return [dest[i] for i in range(len(tokens))]

            def eager(data=data, post=post):
                return post(model.encoder_eager_forward(data))

            def production(data=data, budget=budget, post=post):
                return post(manager._run_budget_graph(data, budget))

            reference = [t.clone() for t in eager()]
            production()
            variants = {}
            specifications = [
                ("buffer-matched", False, False, False, False),
                ("max-tight", True, False, False, False),
                ("slots-tight", False, True, False, False),
                ("attention-tight", True, True, False, False),
                ("patch-tight", True, True, True, False),
                ("all-tight", True, True, True, True),
            ]
            for (
                name,
                max_tight,
                slots_tight,
                patch_tight,
                merge_tight,
            ) in specifications:
                values = {k: v.clone() for k, v in entry.input_buffers.items()}
                values["max_seqlen"] = actual["max_seqlen"].clone()
                if not max_tight:
                    values["max_seqlen"].fill_(original_max)
                if slots_tight:
                    values["cu_seqlens"] = actual["cu_seqlens"].clone()
                if patch_tight:
                    for key in ("patches", "vit_cos", "vit_sin"):
                        values[key] = actual[key].clone()
                if merge_tight:
                    for key in ("merge_idx", "merge_mask"):
                        values[key] = actual[key].clone()
                for _ in range(3):
                    warm = model.encoder_cudagraph_forward(values)
                out = torch.empty_like(warm)
                del warm
                graph = torch.cuda.CUDAGraph()
                with graph_capture(device=device), torch.cuda.graph(graph, pool=pool):
                    out.copy_(model.encoder_cudagraph_forward(values))
                variants[name] = (values, graph, out)

            def fill(values, sources):
                for key, dst in values.items():
                    src = sources[key]
                    if key == "max_seqlen":
                        # This bound is a capture-time host constant.
                        continue
                    if dst.shape == src.shape:
                        dst.copy_(src)
                    elif key == "cu_seqlens":
                        dst[: src.shape[0]].copy_(src)
                        dst[src.shape[0] :] = src[-1]
                    else:
                        dst.zero_()
                        dst[: src.shape[0]].copy_(src)

            def complete_variant(name, variants=variants, data=data, post=post):
                values, graph, out = variants[name]
                replay = model.prepare_encoder_cudagraph_replay_buffers(data, 128, 0)
                fill(values, replay.values)
                graph.replay()
                return post(out)

            def replay_variant(name, variants=variants):
                _, graph, out = variants[name]
                graph.replay()
                return out

            complete = {"production-full": production, "packed-eager": eager}
            complete.update({n: lambda n=n: complete_variant(n) for n in variants})
            checks = {n: compare(fn(), reference) for n, fn in complete.items()}
            save(
                "correctness",
                case=case["name"],
                checks=checks,
                input={k: v for k, v in case.items() if k != "data"},
                actual_shapes={k: list(v.shape) for k, v in actual.items()},
                capture_shapes={
                    k: list(v.shape) for k, v in entry.input_buffers.items()
                },
                actual_max_seqlen=actual["max_seqlen"].item(),
                captured_max_seqlen=original_max,
            )
            if not all(c.get("finite") and c.get("close") for c in checks.values()):
                raise RuntimeError("Correctness failed; do not time invalid variants")

            for fn in complete.values():
                for _ in range(3):
                    fn()

            def production_replay(entry=entry):
                entry.graph.replay()
                return entry.output_buffer

            replay = {"production-full": production_replay}
            replay.update({n: lambda n=n: replay_variant(n) for n in variants})
            replay["packed-eager"] = lambda actual=actual: (
                model.encoder_cudagraph_forward(actual)
            )
            for scope, arms in (
                ("encoder_complete", complete),
                ("forward_prepared", replay),
            ):
                samples = {n: [] for n in arms}
                for block in range(args.blocks):
                    names = list(arms)
                    rng.shuffle(names)
                    for name in names:
                        fn = arms[name]
                        for _ in range(args.repeats):
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            torch.accelerator.synchronize()
                            begin = time.perf_counter()
                            start.record()
                            result = fn()
                            end.record()
                            end.synchronize()
                            samples[name].append(
                                dict(
                                    block=block,
                                    stream_ms=start.elapsed_time(end),
                                    wall_ms=1000 * (time.perf_counter() - begin),
                                )
                            )
                            del result
                save(
                    "timing",
                    case=case["name"],
                    scope=scope,
                    samples=samples,
                    medians={
                        n: statistics.median(s["stream_ms"] for s in ss)
                        for n, ss in samples.items()
                    },
                )
            if args.profile:
                for name in (
                    "production-full",
                    "max-tight",
                    "slots-tight",
                    "attention-tight",
                    "patch-tight",
                    "all-tight",
                    "packed-eager",
                ):
                    fn = replay[name]
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        for _ in range(3):
                            fn()
                        torch.accelerator.synchronize()
                    trace = f"{args.output}.{case['name']}.{name}.trace.json"
                    prof.export_chrome_trace(trace)
                    save("profile", case=case["name"], arm=name, path=trace)
        save("complete", cases=len(cases))


if __name__ == "__main__":
    main()
