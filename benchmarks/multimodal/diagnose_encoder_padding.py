# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate Qwen2.5-VL metadata/padded-forward/replay output differences."""

import argparse
import json
from pathlib import Path


class PaddingDiagnosticWorker:
    def diagnose_encoder_padding(self, options):
        import gc
        from unittest.mock import patch

        import torch

        from benchmarks.multimodal.benchmark_encoder_grid import make_cases
        from vllm.distributed.parallel_state import graph_capture
        from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

        model = self.model_runner.get_model()
        config = self.vllm_config
        config.compilation_config.encoder_cudagraph_token_budgets = [512, 8192]
        config.compilation_config.encoder_cudagraph_max_vision_items_per_batch = 64
        device = next(model.parameters()).device
        cases = [
            c
            for c in make_cases("sweep")
            if c["name"] in {"single-256", "single-2560", "small-2560", "skewed-7168"}
        ]
        embed = model.visual.patch_embed
        width = embed.proj.in_channels * embed.temporal_patch_size * embed.patch_size**2
        prepare = model.visual.prepare_encoder_metadata
        if options["photo"]:
            from PIL import Image
            from transformers import AutoProcessor

            photo = Image.open(options["photo"]).convert("RGB")
            processor = AutoProcessor.from_pretrained(
                config.model_config.model, local_files_only=True
            ).image_processor

        def compare(a, b):
            af, bf = a.float(), b.float()
            return dict(
                equal=torch.equal(a, b),
                close=torch.allclose(a, b, rtol=0.01, atol=0.01),
                finite=bool(torch.isfinite(b).all()),
                max_abs=float((a.float() - b.float()).abs().max()),
                rms=float((a.float() - b.float()).square().mean().sqrt()),
                reference_rms=float(af.square().mean().sqrt()),
                relative_l2=float((af - bf).norm() / af.norm()),
                cosine=float(
                    torch.nn.functional.cosine_similarity(
                        af.flatten(), bf.flatten(), dim=0
                    )
                ),
            )

        with torch.inference_mode():
            for override in (None, 1024):

                def overridden(*args, override=override, **kwargs):
                    if override is not None and "max_seqlen_window_override" in kwargs:
                        kwargs["max_seqlen_window_override"] = override
                    return prepare(*args, **kwargs)

                with patch.object(model.visual, "prepare_encoder_metadata", overridden):
                    manager = EncoderCudaGraphManager(
                        config, device, torch.bfloat16, model
                    )
                    pool = torch.cuda.graph_pool_handle()
                    with graph_capture(device=device):
                        manager.capture(pool)
                    for case in cases:
                        total = sum(case["tokens"])
                        pixels = torch.randint(
                            0,
                            256,
                            (total * 4, width),
                            device=device,
                            dtype=model.visual.input_norm.input_dtype or torch.bfloat16,
                            generator=torch.Generator(device=device).manual_seed(103),
                        )
                        if options["photo"]:
                            images = [
                                photo.resize(
                                    (w * embed.patch_size, h * embed.patch_size)
                                )
                                for _, h, w in case["grid"]
                            ]
                            processed = processor(
                                images=images,
                                do_resize=False,
                                do_normalize=False,
                                do_rescale=False,
                                return_tensors="pt",
                            )
                            assert processed["image_grid_thw"].tolist() == case["grid"]
                            pixels = processed["pixel_values"].to(
                                device=device, dtype=pixels.dtype
                            )
                        mm = dict(pixel_values=pixels, image_grid_thw=case["grid"])
                        reference = model.encoder_eager_forward(mm).clone()
                        split_reference = torch.cat(
                            [
                                model.encoder_eager_forward(
                                    model.select_encoder_cudagraph_items(mm, [i])
                                )
                                for i in range(len(case["tokens"]))
                            ]
                        )
                        replay = model.prepare_encoder_cudagraph_replay_buffers(
                            mm, 64, 0
                        )
                        exact_metadata = model.encoder_cudagraph_forward(
                            dict(replay.values)
                        ).clone()
                        graph = manager._run_budget_graph(mm, 8192)[:total].clone()
                        entry = manager.budget_graphs["default"][8192]
                        bounded = dict(entry.input_buffers)
                        padded = model.encoder_cudagraph_forward(dict(bounded))[
                            :total
                        ].clone()
                        bounded["max_seqlen_full"] = torch.tensor(
                            8192 * 4, dtype=torch.int32
                        )
                        bounded["max_seqlen_window"] = torch.tensor(
                            override or 64, dtype=torch.int32
                        )
                        padded_capture_bound = model.encoder_cudagraph_forward(bounded)[
                            :total
                        ].clone()
                        packed = torch.cat(manager.execute(mm))
                        row = dict(
                            backend=options["backend"],
                            override=override,
                            case=case,
                            input_dtype=str(pixels.dtype),
                            input_norm=type(model.visual.input_norm).__name__,
                            split_eager=compare(reference, split_reference),
                            exact_metadata=compare(reference, exact_metadata),
                            padded=compare(reference, padded),
                            padded_capture_bound=compare(
                                reference, padded_capture_bound
                            ),
                            graph=compare(reference, graph),
                            packed=compare(reference, packed),
                        )
                        with open(options["output"], "a") as f:
                            f.write(json.dumps(row) + "\n")
                        print(json.dumps(row), flush=True)
                        del pixels, mm, reference, replay, exact_metadata, graph
                        del entry, bounded, padded, padded_capture_bound, packed
                        del split_reference
                    manager.clear()
                    del manager, pool
                    gc.collect()
                    torch.accelerator.empty_cache()


def main():
    from vllm import LLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--photo")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        dtype="bfloat16",
        max_model_len=32768,
        max_num_batched_tokens=8192,
        max_num_seqs=8,
        kv_cache_memory_bytes=8 * 1024**3,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 64, "video": 0},
        mm_encoder_attn_backend=args.backend,
        compilation_config=dict(
            mode=0, cudagraph_mode="NONE", cudagraph_mm_encoder=False
        ),
        worker_extension_cls="benchmarks.multimodal.diagnose_encoder_padding.PaddingDiagnosticWorker",
    )
    llm.collective_rpc("diagnose_encoder_padding", args=(vars(args),))


if __name__ == "__main__":
    main()
