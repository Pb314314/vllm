# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Save finite DeepGEMM score samples from synthetic paged Q/K inputs."""

import argparse
import hashlib
import socket
import subprocess
from pathlib import Path

import torch
from pipeline_bench import MODELS, init_workspace_manager, logits_for_rows, make_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--valid-width", type=int, default=32768)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    init_workspace_manager(torch.device("cuda"))
    for model, (heads, width, compression, _) in MODELS.items():
        for dtype in ("mxfp4", "fp8"):
            x = make_inputs(
                args.rows,
                heads,
                width,
                compression,
                args.valid_width * compression,
                dtype,
                8,
                512,
            )
            logits = logits_for_rows(x, 0, args.rows, x["schedule"])
            valid = logits[:, : x["max_valid"]].clone()
            mask = torch.arange(valid.shape[1], device="cuda")[None] < x["ctx"]
            assert valid[mask].isfinite().all()
            valid.masked_fill_(~mask, -float("inf"))
            destination = args.out_dir / f"synthetic_deepgemm_{model}_{dtype}.pt"
            torch.save(
                dict(
                    logits=valid.cpu(),
                    seq_lens=x["ctx"].view(-1).cpu(),
                    metadata=dict(
                        provenance=(
                            "synthetic paged Q/K -> actual DeepGEMM; "
                            "not checkpoint activations"
                        ),
                        model_shape=model,
                        model=model,
                        source="synthetic_qk_deepgemm",
                        heads=heads,
                        head_dim=128,
                        capacity=width,
                        original_logits_shape=list(logits.shape),
                        original_logits_stride=list(logits.stride()),
                        compression=compression,
                        compress_ratio=compression,
                        quantization=dtype,
                        quant=dtype,
                        raw_context_tokens=args.valid_width * compression,
                        rows=args.rows,
                        q_len=8,
                        cache_layout="page values then page scales",
                        quant_scales="unit packed UE8M0",
                        seed=1741,
                        hostname=socket.gethostname(),
                        gpu=torch.cuda.get_device_name(),
                        source_sha=subprocess.check_output(
                            ["git", "rev-parse", "HEAD"],
                            text=True,
                        ).strip(),
                        benchmark_sha256=hashlib.sha256(
                            Path(__file__).with_name("pipeline_bench.py").read_bytes()
                        ).hexdigest(),
                    ),
                ),
                destination,
            )
            print(destination, flush=True)
            del x, logits, valid, mask
            torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
