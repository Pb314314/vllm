# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate FA4 padding work by updating lengths on the same captured graph."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.vllm_flash_attn import flash_attn_varlen_func


def main():
    from flashinfer.testing import bench_gpu_time_with_cupti

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nan-sweep", action="store_true")
    parser.add_argument("--sanitize-tail", action="store_true")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(119)
    cases = [("two-images", [2048, 2304], 8192), ("65-small", [256] * 65, 32768)]
    if args.nan_sweep:
        cases = [
            ("reported-128", [128], 256),
            ("reported-1196", [1196], 2048),
            ("image-576", [576], 1024),
            ("aligned-256", [256], 512),
        ]
    with torch.inference_mode():
        for name, lengths, capacity in cases:
            n = sum(lengths)
            slots = len(lengths) + 2
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + length)
            q, k, v = [
                torch.randn(capacity, 16, 80, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            for t in (q, k, v):
                t[n:].zero_()
            cu = torch.tensor(offsets + [capacity], device="cuda", dtype=torch.int32)
            assert cu.numel() == slots
            out = torch.zeros_like(q)
            row_indices = torch.arange(capacity, device="cuda")

            def forward(
                q=q,
                k=k,
                v=v,
                cu=cu,
                capacity=capacity,
                out=out,
                row_indices=row_indices,
            ):
                if args.sanitize_tail:
                    mask = (row_indices >= cu[-1])[:, None, None]
                    for tensor in (q, k, v):
                        tensor.masked_fill_(mask, 0)
                # Match the capture-time output initialization in vLLM's wrapper.
                out.zero_()
                return flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu,
                    cu_seqlens_k=cu,
                    max_seqlen_q=capacity,
                    max_seqlen_k=capacity,
                    causal=False,
                    fa_version=4,
                    out=out,
                )

            for _ in range(3):
                forward()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            real_cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
            reference = flash_attn_varlen_func(
                q[:n],
                k[:n],
                v[:n],
                cu_seqlens_q=real_cu,
                cu_seqlens_k=real_cu,
                max_seqlen_q=max(lengths),
                max_seqlen_k=max(lengths),
                causal=False,
                fa_version=4,
            )
            for mode, end in (("dummy-sequence", capacity), ("zero-tail", n)):
                cu[-1] = end
                graph.replay()
                torch.accelerator.synchronize()
                valid = out[:n].clone()
                torch.testing.assert_close(valid, reference, rtol=0.01, atol=0.01)
                samples = bench_gpu_time_with_cupti(
                    graph.replay,
                    dry_run_iters=5,
                    repeat_iters=30,
                    cold_l2_cache=True,
                    use_cuda_graph=False,
                )
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                ) as prof:
                    graph.replay()
                    torch.accelerator.synchronize()
                trace = f"{path}.{name}.{mode}.trace.json"
                prof.export_chrome_trace(trace)
                row = dict(
                    case=name,
                    mode=mode,
                    capacity=capacity,
                    real_rows=n,
                    heads=16,
                    head_dim=80,
                    dtype=str(q.dtype),
                    fa_version=4,
                    gpu=torch.cuda.get_device_name(),
                    cuda_graph=True,
                    sanitize_tail=args.sanitize_tail,
                    cold_l2=True,
                    median_ms=statistics.median(samples),
                    samples_ms=samples,
                    valid_equal=torch.equal(valid, reference),
                    valid_finite=bool(torch.isfinite(valid).all()),
                    max_abs=(valid.float() - reference.float()).abs().max().item(),
                    attention_pairs=sum(x * x for x in lengths) + (end - n) ** 2,
                    trace=trace,
                )
                with path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(
                    json.dumps({k: v for k, v in row.items() if k != "samples_ms"}),
                    flush=True,
                )
            for mode, end in (("dummy-sequence", capacity), ("zero-tail", n)):
                cu[-1] = end
                for poisoned in ("q", "k", "v", "qkv"):
                    for key, tensor in zip("qkv", (q, k, v)):
                        tensor[n:].fill_(float("nan") if key in poisoned else 0)
                    graph.replay()
                    torch.accelerator.synchronize()
                    result = dict(
                        case=name,
                        mode=mode,
                        poison=poisoned,
                        sanitize_tail=args.sanitize_tail,
                        valid_finite=bool(torch.isfinite(out[:n]).all()),
                        valid_nan_count=int(torch.isnan(out[:n]).sum()),
                        valid_equal=torch.equal(out[:n], reference),
                    )
                    with path.open("a") as f:
                        f.write(json.dumps(result) + "\n")
                    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
