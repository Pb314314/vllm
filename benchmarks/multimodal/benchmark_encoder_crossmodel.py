# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight encoder-only comparisons for DeepSeek V4.1 and MiniMax M3.1.

The latency run shares a capture superset. Use --memory-only with one policy
for deployment memory. No language model or KV cache is constructed.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import socket
import statistics
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

MODELS = {
    "deepseek": (
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "2cba9e42aa026125f3ed06c6d98c1db82f7ca027",
    ),
    "minimax": (
        "MiniMaxAI/MiniMax-M3.1-Flash",
        "54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e",
    ),
}


def checkpoint(model):
    name, revision = MODELS[model]
    return (
        Path("/mnt/lustre/hf-models/hub")
        / ("models--" + name.replace("/", "--"))
        / "snapshots"
        / revision
    )


def selected_weights(root, prefixes):
    from safetensors import safe_open

    index = json.loads((root / "model.safetensors.index.json").read_text())
    selected = {
        name: shard
        for name, shard in index["weight_map"].items()
        if name.startswith(prefixes)
    }
    for shard in sorted(set(selected.values())):
        with safe_open(root / shard, framework="pt", device="cpu") as reader:
            for name in sorted(k for k, v in selected.items() if v == shard):
                yield name, reader.get_tensor(name)


def build_model(kind, config, root):
    import torch
    from torch import nn

    from vllm.model_executor.models.utils import AutoWeightsLoader

    hf = config.model_config.hf_config
    mm = config.model_config.multimodal_config
    if kind == "deepseek":
        from vllm.models.deepseek_v4.common.vision import (
            DeepseekV4Aligner,
            DeepseekV4ViT,
        )
        from vllm.models.deepseek_v41.common.vl_cudagraph import (
            DeepseekV4VLEncoderCudaGraphMixin,
        )

        class Encoder(nn.Module, DeepseekV4VLEncoderCudaGraphMixin):
            def __init__(self):
                super().__init__()
                self.config = hf
                self.multimodal_config = mm
                self.vision = DeepseekV4ViT(hf)
                self.aligner = DeepseekV4Aligner(hf)
                for name in ("image_start", "image_end", "image_newline"):
                    self.register_parameter(
                        name,
                        nn.Parameter(torch.empty(hf.hidden_size, dtype=torch.float32)),
                    )

        model = Encoder()
        loaded = AutoWeightsLoader(model).load_weights(
            selected_weights(root, ("vision.", "aligner.", "image_"))
        )
    else:
        from transformers import PreTrainedConfig

        from vllm.models.minimax_m3.common.encoder_cudagraph import (
            MiniMaxM3EncoderCudaGraphMixin,
        )
        from vllm.models.minimax_m3.common.vision_tower import MiniMaxVLVisionModel

        class Encoder(nn.Module, MiniMaxM3EncoderCudaGraphMixin):
            def __init__(self):
                super().__init__()
                self.config = hf
                self.multimodal_config = mm
                self.vision_tower = MiniMaxVLVisionModel(
                    config=PreTrainedConfig.from_dict(hf.vision_config),
                    text_hidden_size=hf.text_config.hidden_size,
                    projector_hidden_size=hf.projector_hidden_size,
                    quant_config=None,
                    prefix="vision_tower",
                )

        model = Encoder()

        def mapped():
            for name, tensor in selected_weights(
                root, ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")
            ):
                if name.startswith("vision_tower."):
                    name = name.removeprefix("vision_tower.")
                name = name.replace(".mlp.fc1.", ".fc1.").replace(".mlp.fc2.", ".fc2.")
                yield name, tensor

        loaded = {
            "vision_tower." + n for n in model.vision_tower.load_weights(mapped())
        }

    expected = set(dict(model.named_parameters()))
    missing = sorted(expected - loaded)
    if missing:
        raise RuntimeError(f"Unloaded encoder parameters: {missing}")
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if method is not None:
            method.process_weights_after_loading(module)
    return model.eval(), dict(
        parameters=sum(p.numel() for p in model.parameters()),
        loaded=len(loaded),
        missing=missing,
    )


def prepare_cases(kind, model, root, device, max_tokens, smoke, scenario="default"):
    import numpy as np
    import torch
    from PIL import Image

    if kind == "deepseek":
        from vllm.models.deepseek_v41.common.mm_preprocess import (
            DeepseekV4VLImageProcessor,
            image_token_types,
        )

        processor = DeepseekV4VLImageProcessor(model.config)
    else:
        from vllm.transformers_utils.processors.minimax_m3 import (
            MiniMaxM3VLImageProcessor,
        )

        processor = MiniMaxM3VLImageProcessor.from_pretrained(
            root, local_files_only=True
        )
    cache = {}

    def item(size):
        if size not in cache:
            w, h = size
            y, x = np.indices((h, w), dtype=np.uint32)
            rgb = np.stack(((x * 3 + y) % 256, (y * 5) % 256, (x + y * 7) % 256), -1)
            image = Image.fromarray(rgb.astype(np.uint8))
            if kind == "deepseek":
                patches, vh, vw, lh, lw = processor(image)
                data = dict(
                    patches=patches.to(device),
                    vit_grid=[[vh, vw]],
                    llm_grid=[[lh, lw]],
                    types=image_token_types(lh, lw),
                )
            else:
                data = dict(processor(images=[image], return_tensors="pt"))
                data = dict(
                    pixel_values=data["pixel_values"].to(device, torch.bfloat16),
                    image_grid_thw=data["image_grid_thw"].cpu(),
                )
            cache[size] = data
        return cache[size]

    def batch(sizes):
        items = [item(size) for size in sizes]
        if kind == "deepseek":
            return dict(
                patches=torch.cat([d["patches"] for d in items]),
                vit_grid=[g for d in items for g in d["vit_grid"]],
                llm_grid=[g for d in items for g in d["llm_grid"]],
                types=torch.cat([d["types"] for d in items]),
            )
        return dict(
            pixel_values=torch.cat([d["pixel_values"] for d in items]),
            image_grid_thw=torch.cat([d["image_grid_thw"] for d in items]),
        )

    layouts = [(f"small-{n}", [(224, 224)] * n) for n in (1, 8, 32, 64, 128, 256)]
    layouts += [
        ("medium", [(672, 672)]),
        ("large", [(1344, 1344)]),
        ("mixed-gap", [(448, 896), (896, 896)]),
        ("mixed-tail", [(1344, 1344)] + [(224, 224)] * 16),
    ]
    small_tokens = model.get_encoder_cudagraph_item_specs(item((224, 224)))[
        0
    ].output_tokens
    for cap in (4096, 6144, 8192):
        n = cap // small_tokens
        for count in (n, n + 1):
            if count:
                layouts.append((f"near-{cap}-n{count}", [(224, 224)] * count))
    if smoke:
        layouts = [("small-1", [(224, 224)]), ("mixed-gap", [(448, 896), (896, 896)])]
    if scenario == "unaligned":
        layouts = [
            ("odd-small", [(252, 252)]),
            ("odd-medium", [(336, 336)]),
            ("odd-wide", [(448, 420)]),
            ("odd-mixed", [(252, 252), (336, 336), (448, 420)]),
            ("odd-many", [(336, 336)] * 17),
        ]
    elif scenario == "threshold-holdout":
        layouts = [
            (f"control-small-{n}", [(224, 224)] * n) for n in (1, 8, 32, 64, 128)
        ]
        layouts += [("control-wide-4", [(448, 896)] * 4)]
        shape_rng = random.Random(7431)
        shapes = [
            (224, 224),
            (280, 280),
            (392, 392),
            (448, 448),
            (560, 560),
            (672, 672),
            (448, 896),
            (420, 280),
            (364, 140),
        ]
        for count in (3, 5, 9, 13, 17, 25):
            for variant in range(2):
                layouts.append(
                    (
                        f"heldout-{count}-{variant}",
                        [shape_rng.choice(shapes) for _ in range(count)],
                    )
                )
    elif scenario == "threshold":
        layouts = [
            (f"small-{n}", [(224, 224)] * n) for n in (1, 4, 8, 16, 24, 32, 40, 48, 64)
        ]
        layouts += [(f"medium-{n}", [(448, 448)] * n) for n in (4, 8, 12, 16)]
    elif scenario == "padding-cliffs":
        layouts = []
        wide = (448, 896)
        wide_tokens = model.get_encoder_cudagraph_item_specs(item(wide))[
            0
        ].output_tokens
        for boundary in (256, 512, 1024, 2048, 4096):
            count = boundary // small_tokens + 1
            layouts.append((f"small-above-{boundary}", [(224, 224)] * count))
            count = max(1, (boundary - small_tokens) // wide_tokens + 1)
            layouts.append((f"mixed-above-{boundary}", [(224, 224)] + [wide] * count))
        layouts.extend(
            [
                ("mixed-square-320", [(224, 224), (448, 448)]),
                ("mixed-square-2112", [(224, 224)] + [(448, 448)] * 8),
                ("mixed-gap", [(448, 896), (896, 896)]),
                ("small-at-4K", [(224, 224)] * (4096 // small_tokens)),
                ("small-1", [(224, 224)]),
            ]
        )
    elif scenario in ("one-above", "threshold-grid"):
        targets = [b + 1 for b in (128, 256, 512, 1024, 2048, 3072, 3968, 4096)]
        if scenario == "threshold-grid":
            targets = [129, 257, 513] + [
                boundary + offset
                for boundary in (
                    1024,
                    1280,
                    1536,
                    1792,
                    2048,
                    2304,
                    2560,
                    2816,
                    3072,
                    4096,
                )
                for offset in (-1, 0, 1)
            ]
        if kind == "minimax":
            # 7x9, 8x8 and 5x13 merged patches put real batches on either
            # side of a graph boundary without fabricating processor output.
            layouts = []
            for target in targets:
                tail = 64 + (target % 64 if target % 64 < 32 else target % 64 - 64)
                tail_size = {63: (252, 196), 64: (224, 224), 65: (364, 140)}[tail]
                count = (target - tail) // 64
                layouts.append((f"many-{target}", [(224, 224)] * count + [tail_size]))
                wide_count, small_count = divmod(target - tail, 512)
                small_count //= 64
                layouts.append(
                    (
                        f"mixed-{target}",
                        [(448, 896)] * wide_count
                        + [(224, 224)] * small_count
                        + [tail_size],
                    )
                )
        else:
            from vllm.models.deepseek_v41.common.mm_preprocess import (
                num_image_tokens,
                safe_resize,
            )

            # Predict using the processor's resize helpers, then verify each
            # selected layout below with the actual processed item specs.
            catalog = {}
            cell = processor.patch_size * processor.downsample_ratio
            for h in range(1, 81):
                for w in range(1, 81):
                    width, height = w * cell, h * cell
                    resized_w, resized_h = width, height
                    if (
                        processor.max_wh_ratio is not None
                        and resized_w > resized_h * processor.max_wh_ratio
                    ):
                        resized_w = resized_h * processor.max_wh_ratio
                    if resized_w * resized_h < processor.min_pixels:
                        ratio = (processor.min_pixels / (resized_w * resized_h)) ** 0.5
                        resized_w, resized_h = (
                            int(resized_w * ratio),
                            int(resized_h * ratio),
                        )
                    patch = processor.patch_size
                    lh, lw, _, _ = safe_resize(
                        resized_h,
                        resized_w,
                        math.ceil(resized_h / patch) * patch,
                        math.ceil(resized_w / patch) * patch,
                        patch,
                        processor.downsample_ratio,
                        processor.max_n_token,
                    )
                    tokens = num_image_tokens(lh, lw)
                    preference = (abs(math.log(width / height)), width * height)
                    if tokens not in catalog or preference < catalog[tokens][0]:
                        catalog[tokens] = (preference, (width, height))
            layouts = []
            for mode in ("few", "many"):
                choices = {0: []}
                for total in range(1, max(targets) + 1025):
                    options = [
                        choices[total - n] + [n]
                        for n in catalog
                        if n <= total
                        and total - n in choices
                        and len(choices[total - n]) < 128
                    ]
                    if options:
                        choices[total] = (min if mode == "few" else max)(
                            options, key=len
                        )
                for target in targets:
                    actual = next(
                        n for n in range(target, max(choices) + 1) if n in choices
                    )
                    layouts.append(
                        (
                            f"{mode}-{target}-actual-{actual}",
                            [catalog[n][1] for n in choices[actual]],
                        )
                    )
    seen, cases, skipped = set(), [], []
    for name, sizes in layouts:
        signature = tuple(sizes)
        if signature in seen:
            continue
        seen.add(signature)
        data = batch(sizes)
        specs = model.get_encoder_cudagraph_item_specs(data)
        tokens = [s.output_tokens for s in specs]
        if scenario in ("one-above", "threshold-grid"):
            expected = int(name.rsplit("-", 1)[1])
            assert sum(tokens) == expected, (name, sizes, tokens)
        if sum(tokens) > max_tokens:
            skipped.append(
                dict(name=name, tokens=sum(tokens), reason="admission limit")
            )
            continue
        cases.append(
            dict(
                name=name,
                image_pixels=sizes,
                tokens=tokens,
                patch_rows=[s.input_size for s in specs],
                data=data,
            )
        )
    return cases, skipped


def memory():
    import torch

    torch.accelerator.synchronize()
    free, total = torch.accelerator.get_memory_info()
    return dict(
        device_used=total - free,
        allocated=torch.accelerator.memory_allocated(),
        reserved=torch.accelerator.memory_reserved(),
        peak_allocated=torch.accelerator.max_memory_allocated(),
        peak_reserved=torch.accelerator.max_memory_reserved(),
    )


def compare(actual, expected):
    import torch

    if len(actual) != len(expected) or any(
        a.shape != b.shape for a, b in zip(actual, expected)
    ):
        return dict(shapes_match=False)
    a = torch.cat(actual).float()
    b = torch.cat(expected).float()
    diff = a - b
    return dict(
        shapes_match=True,
        finite=bool(torch.isfinite(a).all()),
        equal=torch.equal(a, b),
        close=torch.allclose(a, b, atol=0.01, rtol=0.01),
        max_abs=diff.abs().max().item(),
        relative_l2=(diff.norm() / b.norm().clamp_min(1e-12)).item(),
    )


def diagnose_deepseek(model, data):
    """Separate dense/varlen attention differences from the spatial merge."""
    import torch.nn.functional as F

    from vllm.models.deepseek_v4.common.vision import get_vision_cos_sin

    data = model.select_encoder_cudagraph_items(data, [0])
    values = model.prepare_encoder_cudagraph_replay_buffers(data, 64, 0).values
    h, w = model._get_grid_list(data, "vit_grid")[0]
    patches = values["patches"]
    cos, sin = get_vision_cos_sin(h, w, model.vision.rope_dim, model.vision.rope_theta)
    cos, sin = cos.to(patches.device), sin.to(patches.device)
    dense = model.vision.patch_embed(patches)
    packed = dense.clone()
    stages = []
    for i, block in enumerate(model.vision.blocks):
        dense = block(dense, cos, sin)
        packed = block(
            packed,
            values["vit_cos"],
            values["vit_sin"],
            values["cu_seqlens"],
            values["max_seqlen"],
        )
        stages.append(dict(layer=i, **compare([packed], [dense])))
    dense = model.vision.norm(dense)
    native_merge = model.aligner(dense, h, w)
    packed_merge = model.aligner.forward_packed(
        dense, values["merge_idx"], values["merge_mask"]
    )
    r = model.aligner.downsample_ratio
    merged = dense.view(h, w, -1).permute(2, 0, 1)
    merged = F.pad(merged, (0, -w % r, 0, -h % r))
    merged = F.unfold(merged.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
    gathered = dense[values["merge_idx"]] * values["merge_mask"]
    gathered = gathered.permute(0, 2, 1).reshape(merged.shape)
    hidden, _ = model.aligner.w1(merged.contiguous())
    contiguous_out, _ = model.aligner.w2(F.gelu(hidden))
    return dict(
        grid=[h, w],
        rope_cos=compare([values["vit_cos"]], [cos]),
        rope_sin=compare([values["vit_sin"]], [sin]),
        layers=stages,
        merge_inputs=compare([gathered], [merged]),
        merge_input_strides=dict(
            native=list(merged.stride()), packed=list(gathered.stride())
        ),
        merge_with_identical_vit_input=compare([packed_merge], [native_merge]),
        packed_vs_native_contiguous_merge=compare([packed_merge], [contiguous_out]),
    )


def diagnose_minimax(model, data, device):
    """Locate shape-dependent differences before interpreting replay accuracy."""
    import torch

    n = data["pixel_values"].shape[0]
    budget = 256
    values = model.prepare_encoder_cudagraph_capture_inputs(
        budget, 64, 0, device, torch.bfloat16
    ).values
    replay = model.prepare_encoder_cudagraph_replay_buffers(data, 64, 0).values
    padding = model.get_encoder_cudagraph_config().padding_logics
    for key, src in replay.items():
        if src is None:
            continue
        dst = values[key]
        if key in padding:
            padding[key](dst, src)
        else:
            dst.zero_()
            dst[: src.shape[0]].copy_(src)

    snapshots = {}
    handles = []
    for name, module in model.named_modules():
        if not name.startswith("vision_tower.") or name.endswith(".act"):
            continue

        def hook(module, inputs, output, name=name):
            if isinstance(output, tuple):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return
            # Attention returns [1, N, H, D]; other stages have N first.
            axis = 1 if output.ndim == 4 and output.shape[0] == 1 else 0
            rows = n // 4 if "patch_merge_mlp" in name else n
            snapshots[name] = output.narrow(axis, 0, rows).clone()

        handles.append(module.register_forward_hook(hook))
        if name.endswith("layers.0.self_attn.attn"):

            def prehook(module, inputs, kwargs, name=name):
                for key in ("query", "key", "value"):
                    snapshots[name + ".input." + key] = kwargs[key][:, :n].clone()

            handles.append(module.register_forward_pre_hook(prehook, with_kwargs=True))
    try:
        model.encoder_eager_forward(data)
        native = snapshots.copy()
        model.encoder_cudagraph_forward(dict(values))
        padded = snapshots.copy()
        for _ in range(3):
            model.encoder_cudagraph_forward(dict(values))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            model.encoder_cudagraph_forward(dict(values))
        graph.replay()
        torch.accelerator.synchronize()
        captured = snapshots.copy()
        stages = {
            name: {
                "padded_eager_vs_native": compare([padded[name]], [native[name]]),
                "graph_vs_padded_eager": compare([captured[name]], [padded[name]]),
            }
            for name in native
        }
        name = "vision_tower.vision_model.encoder.layers.0.self_attn.attn"
        q, k, v = [
            native[name + ".input." + key].float().transpose(1, 2)
            for key in ("query", "key", "value")
        ]
        reference = (
            (q @ k.transpose(-2, -1) * q.shape[-1] ** -0.5).softmax(-1) @ v
        ).transpose(1, 2)
        return dict(
            stages=stages,
            first_attention_vs_fp32={
                "native": compare([native[name]], [reference]),
                "padded": compare([padded[name]], [reference]),
            },
        )
    finally:
        for handle in handles:
            handle.remove()


def benchmark_tail_guard_ablation(args, model, config, device, cases, save):
    """Measure guard cost separately from eliminating dummy attention work."""
    import torch

    from vllm.distributed.parallel_state import graph_capture
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    budgets = [128, 2048, 4096, 8192]
    config.compilation_config.encoder_cudagraph_token_budgets = budgets
    config.compilation_config.encoder_cudagraph_max_vision_items_per_batch = (
        args.max_items
    )
    managers = {}
    pool = torch.cuda.graph_pool_handle()
    for guard in ("none", "output", "qkv-output"):
        args.attention_tail_guard = guard
        manager = EncoderCudaGraphManager(config, device, torch.bfloat16, model)
        with graph_capture(device=device):
            manager.capture(pool)
        managers[guard] = manager
    save("ablation_capture", guards=list(managers), budgets=budgets)

    def run(manager, data, tokens, budget, zero_tail):
        def pad(dst, src):
            dst[: src.shape[0]].copy_(src)
            dst[src.shape[0] :] = src[-1]

        old = manager.config.padding_logics["cu_seqlens"]
        if zero_tail:
            manager.config.padding_logics["cu_seqlens"] = pad
        try:
            output = manager._run_budget_graph(
                data, budget, path="default", axis_keys=()
            )
            result = {}
            model.postprocess_encoder_output(
                {"default": output},
                list(range(len(tokens))),
                tokens,
                result,
                clone=True,
                batch_mm_kwargs=data,
            )
            return [result[i] for i in range(len(tokens))]
        finally:
            manager.config.padding_logics["cu_seqlens"] = old

    rng = random.Random(1190)
    for case in cases:
        data, tokens = case["data"], case["tokens"]
        budget = next(b for b in budgets if b >= sum(tokens))
        output = model.encoder_eager_forward(data)
        reference = {}
        model.postprocess_encoder_output(
            {"default": output},
            list(range(len(tokens))),
            tokens,
            reference,
            clone=True,
            batch_mm_kwargs=data,
        )
        reference = [reference[i] for i in range(len(tokens))]
        arms = {
            f"{mode}/{guard}": partial(
                run, manager, data, tokens, budget, mode == "zero-tail"
            )
            for guard, manager in managers.items()
            for mode in ("dummy", "zero-tail")
        }
        baseline = arms["dummy/none"]()
        checks = {}
        for name, fn in arms.items():
            actual = fn()
            checks[name] = dict(
                vs_eager=compare(actual, reference),
                vs_baseline=compare(actual, baseline),
            )
            for _ in range(args.warmup):
                fn()
        save(
            "ablation_correctness",
            name=case["name"],
            tokens=tokens,
            budget=budget,
            replays=1,
            checks=checks,
        )
        samples = {name: [] for name in arms}
        for block in range(args.blocks):
            names = list(arms)
            rng.shuffle(names)
            for name in names:
                for _ in range(args.repeats):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    torch.accelerator.synchronize()
                    begin = time.perf_counter()
                    start.record()
                    result = arms[name]()
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
            "ablation_timing",
            name=case["name"],
            tokens=tokens,
            samples=samples,
            medians={
                name: statistics.median(s["stream_ms"] for s in rows)
                for name, rows in samples.items()
            },
        )
        if case["name"] in args.profile_cases.split(","):
            for name, fn in arms.items():
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    fn()
                    torch.accelerator.synchronize()
                trace = (
                    f"{args.output}.{case['name']}.{name.replace('/', '-')}.trace.json"
                )
                prof.export_chrome_trace(trace)
                save("ablation_profile", name=case["name"], arm=name, trace=trace)
    save("complete", cases=len(cases))


def benchmark(args):
    import torch

    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import graph_capture
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.network_utils import get_open_port
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    root = (
        Path(args.model_path).resolve() if args.model_path else checkpoint(args.model)
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save(event, **values):
        with output.open("a") as f:
            f.write(json.dumps(dict(event=event, time=time.time(), **values)) + "\n")
        print(event, values.get("name", ""), flush=True)

    config = EngineArgs(
        model=str(root),
        dtype="bfloat16",
        max_model_len=32768,
        max_num_batched_tokens=args.admission,
        max_num_seqs=256,
        enforce_eager=True,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 256, "video": 0},
        mm_encoder_attn_backend=args.backend,
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
        started = time.perf_counter()
        with torch.device(device):
            model, weights = build_model(args.model, config, root)
        if (
            args.poison_attention_output
            or args.attention_tail_guard != "none"
            or args.tail_guard_ablation
        ):
            from vllm.v1.attention.backends import fa_utils

            original_attention = fa_utils.flash_attn_varlen_func
            guard_rows = {}

            def poison_attention(*a, **kw):
                q, k, v = a[:3]
                key = (q.device, q.shape[0])
                if args.attention_tail_guard != "none" and key not in guard_rows:
                    guard_rows[key] = torch.arange(q.shape[0], device=q.device)
                capturing = torch.cuda.is_current_stream_capturing()
                if (
                    capturing
                    and args.poison_attention_output
                    and kw.get("out") is not None
                ):
                    kw["out"].fill_(float("nan"))
                mask = None
                if capturing and args.attention_tail_guard != "none":
                    mask = guard_rows[key] >= kw["cu_seqlens_q"][-1]
                    if args.attention_tail_guard == "qkv-output":
                        for tensor in (q, k, v):
                            tensor.masked_fill_(mask[:, None, None], 0)
                result = original_attention(*a, **kw)
                if mask is not None:
                    result.masked_fill_(mask[:, None, None], 0)
                return result

            fa_utils.flash_attn_varlen_func = poison_attention
        save(
            "loaded",
            model=MODELS[args.model],
            weights=weights,
            seconds=time.perf_counter() - started,
            memory=memory(),
            gpu=torch.cuda.get_device_name(),
            backend=args.backend,
            options=vars(args),
            host=socket.gethostname(),
            job=os.getenv("SLURM_JOB_ID"),
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            attention_versions=sorted(
                {
                    str(m._fa_version)
                    for m in model.modules()
                    if hasattr(m, "_fa_version")
                }
            ),
        )
        if args.memory_only:
            cases, skipped = [], []
        else:
            cases, skipped = prepare_cases(
                args.model,
                model,
                root,
                device,
                args.admission,
                args.smoke,
                args.scenario,
            )
            if args.only_cases:
                selected = set(args.only_cases.split(","))
                cases = [c for c in cases if c["name"] in selected]
        save(
            "inputs",
            cases=[{k: v for k, v in c.items() if k != "data"} for c in cases],
            skipped=skipped,
        )
        if args.diagnose_deepseek and args.model == "deepseek":
            for case in cases:
                save(
                    "numerical_diagnostic",
                    name=case["name"],
                    **diagnose_deepseek(model, case["data"]),
                )
        if args.diagnose_minimax and args.model == "minimax":
            for case in cases:
                save(
                    "numerical_diagnostic",
                    name=case["name"],
                    layers=diagnose_minimax(model, case["data"], device),
                )
            save("complete", cases=len(cases))
            return
        if args.tail_guard_ablation:
            benchmark_tail_guard_ablation(args, model, config, device, cases, save)
            return
        powers = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        dense = [
            64,
            128,
            256,
            512,
            640,
            768,
            896,
            1024,
            1280,
            1536,
            1792,
            2048,
            2560,
            3072,
            3584,
            4096,
            4608,
            5120,
            5632,
            6144,
            6656,
            7168,
            7680,
            8192,
        ]
        # The production manager requires at least one output row per item.
        powers = [x for x in powers if x >= args.max_items]
        dense = [x for x in dense if x >= args.max_items]
        policies = {
            "split-8K": (powers, "split"),
            "eager-6K": ([x for x in dense if x <= 6144], "eager"),
            "eager-8K": (dense, "eager"),
        }
        if args.dispatch_ablation:
            dense = sorted(set(dense + [1152, 1408, 1664, 1920]))
            policies = {
                "split-8K": (powers, "split"),
                "single-coarse-8K": (powers, "eager"),
                "single-fine-8K": (dense, "eager"),
                "single-fine-4K": ([x for x in dense if x <= 4096], "eager"),
            }
            if args.scenario in (
                "padding-cliffs",
                "one-above",
                "threshold-grid",
                "threshold-holdout",
            ):
                linear = list(range(128, 4097, 128))
                policies["single-linear-4K"] = (linear, "eager")
                if args.guard_min_tokens is not None:
                    policies["single-padding-aware"] = (linear, "eager")
                dense = sorted(set(dense + linear))
                for cap in args.candidate_caps:
                    if cap <= 0 or cap % 128 or cap > 4096:
                        raise ValueError(
                            "Candidate cap must be a multiple of 128 up to 4096"
                        )
                    policies[f"single-cap-{cap}"] = (
                        [x for x in linear if x <= cap],
                        "eager",
                    )
        if args.smoke:
            powers = [64, 128, 256, 512, 1024, 2048]
            dense = [64, 128, 256, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048]
            policies = {"split-2K": (powers, "split"), "eager-2K": (dense, "eager")}
        totals = {sum(c["tokens"]) for c in cases if len(c["tokens"]) <= args.max_items}
        sizes = sorted(set(powers + dense) | {max(args.max_items, t) for t in totals})
        if args.memory_only:
            sizes = policies[args.policy][0]
        cc = config.compilation_config
        cc.encoder_cudagraph_token_budgets = sizes
        cc.encoder_cudagraph_max_vision_items_per_batch = args.max_items
        manager = EncoderCudaGraphManager(config, device, torch.bfloat16, model)
        pool = torch.cuda.graph_pool_handle()
        gc.collect()
        torch.accelerator.empty_cache()
        before = memory()
        started = time.perf_counter()
        with graph_capture(device=device):
            manager.capture(pool)
        segments = [
            s
            for s in torch.cuda.memory_snapshot()
            if tuple(s.get("segment_pool_id", ())) == tuple(pool)
        ]
        buffers = {"input": {}, "output": {}}
        shapes = {}
        captured_host = {}
        for budget, entry in manager.budget_graphs["default"].items():
            captured_host[budget] = {
                k: v.clone()
                for k, v in entry.input_buffers.items()
                if isinstance(v, torch.Tensor) and not v.is_cuda
            }
            shapes[str(budget)] = {
                k: list(v.shape)
                for k, v in entry.input_buffers.items()
                if isinstance(v, torch.Tensor)
            }
            for kind, ts in (
                ("input", entry.input_buffers.values()),
                ("output", [entry.output_buffer]),
            ):
                for t in ts:
                    if isinstance(t, torch.Tensor) and t.is_cuda:
                        s = t.untyped_storage()
                        buffers[kind][s.data_ptr()] = s.nbytes()
        save(
            "capture",
            before=before,
            after=memory(),
            seconds=time.perf_counter() - started,
            sizes=sizes,
            input_shapes=shapes,
            pool_reserved=sum(s["total_size"] for s in segments),
            persistent_bytes={k: sum(v.values()) for k, v in buffers.items()},
            scope="deployment"
            if args.memory_only
            else "shared superset, not deployment memory",
        )
        if args.memory_only:
            save("complete", cases=0)
            return

        def post(out, data, tokens):
            dest = {}
            model.postprocess_encoder_output(
                {"default": out},
                list(range(len(tokens))),
                tokens,
                dest,
                clone=True,
                batch_mm_kwargs=data,
            )
            return [dest[i] for i in range(len(tokens))]

        def eager(data, tokens):
            return post(model.encoder_eager_forward(data), data, tokens)

        def sequential(data, tokens):
            patches = data["patches"].to(model.aligner.w1.weight.dtype)
            outputs = []
            offset = 0
            for h, w in model._get_grid_list(data, "vit_grid"):
                outputs.append(
                    model._encode_image(patches[offset : offset + h * w], h, w)
                )
                offset += h * w
            return post(torch.cat(outputs), data, tokens)

        def packed(data, tokens):
            if args.model == "deepseek":
                values = model.prepare_encoder_cudagraph_replay_buffers(
                    data, args.max_items, 0
                ).values
            else:
                metadata = model.vision_tower.vision_model.prepare_encoder_metadata(
                    model._get_grid_thw_list(data), device=device
                )
                values = {"pixel_values": data["pixel_values"], **metadata}
            return post(model.encoder_cudagraph_forward(dict(values)), data, tokens)

        def dispatch(data, grid, policy, guard_min_tokens=None):
            old = (
                manager.path_token_budgets["default"],
                manager.oversize_policy,
                manager.eager_min_tokens,
                manager.max_padding_ratio,
            )
            manager.path_token_budgets["default"] = grid
            manager.oversize_policy = policy
            manager.eager_min_tokens = guard_min_tokens
            manager.max_padding_ratio = args.guard_max_padding
            try:
                return manager._execute_local(data)
            finally:
                (
                    manager.path_token_budgets["default"],
                    manager.oversize_policy,
                    manager.eager_min_tokens,
                    manager.max_padding_ratio,
                ) = old

        def zero_tail(data, grid, policy):
            def pad(dst, src):
                dst[: src.shape[0]].copy_(src)
                dst[src.shape[0] :] = src[-1]

            old = manager.config.padding_logics["cu_seqlens"]
            manager.config.padding_logics["cu_seqlens"] = pad
            try:
                return dispatch(data, grid, policy)
            finally:
                manager.config.padding_logics["cu_seqlens"] = old

        def single(data, tokens, budget):
            out = manager._run_budget_graph(data, budget, path="default", axis_keys=())
            return post(out, data, tokens)

        def padded_eager(data, tokens, budget):
            entry = manager.budget_graphs["default"][budget]
            graph = entry.graph

            def calculate():
                values = dict(entry.input_buffers)
                # CPU launch bounds were constants during capture, even if the
                # manager subsequently refreshed their host tensor values.
                values.update(captured_host[budget])
                entry.output_buffer.copy_(model.encoder_cudagraph_forward(values))

            entry.graph = SimpleNamespace(replay=calculate)
            try:
                return single(data, tokens, budget)
            finally:
                entry.graph = graph

        rng = random.Random(args.seed)
        rng.shuffle(cases)
        for case in cases:
            data, tokens = case["data"], case["tokens"]
            arms = {
                "model-eager": partial(eager, data, tokens),
                "packed-eager": partial(packed, data, tokens),
            }
            if args.sequential_reference and args.model == "deepseek":
                arms["sequential-eager"] = partial(sequential, data, tokens)
            for name, (grid, policy) in policies.items():
                minimum = (
                    args.guard_min_tokens if name == "single-padding-aware" else None
                )
                arms[name] = partial(dispatch, data, grid, policy, minimum)
            if args.zero_tail_ablation:
                for name, (grid, policy) in policies.items():
                    arms[name + "-zero-tail"] = partial(zero_tail, data, grid, policy)
            if len(tokens) <= args.max_items:
                total = sum(tokens)
                exact = max(args.max_items, total)
                arms["token-matched-full"] = partial(single, data, tokens, exact)
                rounded = next((b for b in powers if b >= total), None)
                if rounded is not None:
                    arms["rounded-full"] = partial(single, data, tokens, rounded)
                    if case["name"] in ("small-1", "mixed-gap", "mixed-tail"):
                        arms["rounded-padded-eager"] = partial(
                            padded_eager, data, tokens, rounded
                        )
            if args.timing_arms:
                selected_arms = args.timing_arms.split(",")
                arms = {name: arms[name] for name in selected_arms}
            reference = eager(data, tokens)
            packed_reference = packed(data, tokens)
            split_reference = dispatch(data, powers, "split")
            if args.model == "deepseek" and case["name"] == "mixed-gap":
                indices = list(reversed(range(len(tokens))))
                reordered = model.select_encoder_cudagraph_items(data, indices)
                reordered_output = eager(reordered, [tokens[i] for i in indices])
                permutation = compare(reordered_output, [reference[i] for i in indices])
                empty = model.encoder_eager_forward(
                    model.select_encoder_cudagraph_items(data, [])
                )
                assert permutation["finite"] and permutation["close"]
                assert tuple(empty.shape) == (0, model.config.hidden_size)
                save(
                    "fallback_contract",
                    name=case["name"],
                    permutation=permutation,
                    empty_shape=list(empty.shape),
                    empty_dtype=str(empty.dtype),
                )
            checks = {}
            plans = {}
            for name, fn in arms.items():
                calls = []
                orig = manager._run_budget_graph

                def record(*a, _calls=calls, _orig=orig, **kw):
                    _calls.append(dict(budget=a[1], path=kw.get("path")))
                    return _orig(*a, **kw)

                manager._run_budget_graph = record
                try:
                    out = fn()
                    checks[name] = dict(
                        vs_native=compare(out, reference),
                        vs_packed=compare(out, packed_reference),
                        vs_split=compare(out, split_reference),
                    )
                    plans[name] = calls
                    del out
                finally:
                    manager._run_budget_graph = orig
                for _ in range(args.warmup):
                    fn()
            save(
                "correctness",
                name=case["name"],
                tokens=tokens,
                checks=checks,
                replay_plans=plans,
            )
            samples = {name: [] for name in arms}
            peaks = {name: [] for name in arms}
            for block in range(args.blocks):
                names = list(arms)
                rng.shuffle(names)
                for name in names:
                    fn = arms[name]
                    torch.accelerator.synchronize()
                    torch.accelerator.reset_peak_memory_stats()
                    for _ in range(args.repeats):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        torch.accelerator.synchronize()
                        wall = time.perf_counter()
                        start.record()
                        out = fn()
                        host = time.perf_counter()
                        end.record()
                        end.synchronize()
                        samples[name].append(
                            dict(
                                block=block,
                                stream_ms=start.elapsed_time(end),
                                host_ms=1000 * (host - wall),
                                wall_ms=1000 * (time.perf_counter() - wall),
                            )
                        )
                        del out
                    peaks[name].append(memory())
            save(
                "timing",
                name=case["name"],
                tokens=tokens,
                samples=samples,
                medians={
                    k: statistics.median(s["stream_ms"] for s in v)
                    for k, v in samples.items()
                },
                memory=memory(),
                memory_by_arm=peaks,
            )
            if case["name"] in args.profile_cases.split(","):
                for name in args.profile_arms.split(","):
                    if name not in arms:
                        continue
                    torch.accelerator.synchronize()
                    with (
                        torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                        ) as prof,
                        torch.profiler.record_function(name),
                    ):
                        result = arms[name]()
                        torch.accelerator.synchronize()
                    trace = str(output) + f".{case['name']}.{name}.trace.json"
                    prof.export_chrome_trace(trace)
                    save("profile", name=case["name"], arm=name, trace=trace)
                    del result, prof
            del reference, packed_reference, split_reference
        save("complete", cases=len(cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument(
        "--model-path", help="Local snapshot of the pinned model revision"
    )
    parser.add_argument("--backend", default="FLASH_ATTN")
    parser.add_argument("--output", required=True)
    parser.add_argument("--admission", type=int, default=16384)
    parser.add_argument("--max-items", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9031)
    parser.add_argument("--timing-arms", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only-cases")
    parser.add_argument("--memory-only", action="store_true")
    parser.add_argument("--dispatch-ablation", action="store_true")
    parser.add_argument("--candidate-caps", type=int, nargs="*", default=[])
    parser.add_argument("--guard-min-tokens", type=int)
    parser.add_argument("--guard-max-padding", type=float, default=0.02)
    parser.add_argument("--profile-cases", default="")
    parser.add_argument("--diagnose-deepseek", action="store_true")
    parser.add_argument("--sequential-reference", action="store_true")
    parser.add_argument("--zero-tail-ablation", action="store_true")
    parser.add_argument("--poison-attention-output", action="store_true")
    parser.add_argument(
        "--attention-tail-guard",
        choices=["none", "output", "qkv-output"],
        default="none",
    )
    parser.add_argument("--diagnose-minimax", action="store_true")
    parser.add_argument("--tail-guard-ablation", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=[
            "default",
            "unaligned",
            "threshold",
            "padding-cliffs",
            "one-above",
            "threshold-grid",
            "threshold-holdout",
        ],
        default="default",
    )
    parser.add_argument(
        "--profile-arms",
        default="packed-eager,split-8K,token-matched-full,rounded-full",
    )
    parser.add_argument(
        "--policy",
        choices=[
            "split-8K",
            "eager-6K",
            "eager-8K",
            "single-coarse-8K",
            "single-fine-8K",
            "single-fine-4K",
            "single-linear-4K",
            "single-padding-aware",
            *[f"single-cap-{cap}" for cap in range(128, 4097, 128)],
        ],
        default="eager-6K",
    )
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
