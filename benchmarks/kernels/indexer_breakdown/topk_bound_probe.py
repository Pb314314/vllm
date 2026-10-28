# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expose one correctness-checked DeepSelect launch to Nsight Compute.

Run ncu with --profile-from-start off. No CUPTI timer is enabled here, so the
profiler owns the activity/counter collection. This is not a latency benchmark.
"""

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

import torch

import vllm
from vllm.model_executor.layers.indexer_topk import deep_select_topk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--width", type=int, default=1048576)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.topk <= args.length <= args.width:
        parser.error("Require topk <= length <= width")
    padding = 1024 if args.width == 524288 else 512
    logits = torch.empty_strided(
        (args.rows, args.width),
        (args.width + padding, 1),
        device="cuda",
        dtype=torch.bfloat16,
    )
    generator = torch.Generator(device="cuda").manual_seed(71)
    logits.normal_(generator=generator)
    ends = torch.full((args.rows,), args.length, device="cuda", dtype=torch.int32)
    output = torch.empty((args.rows, args.topk), device="cuda", dtype=torch.int32)

    def run():
        deep_select_topk(logits, args.topk, end=ends, output_idx=output)

    for _ in range(5):
        run()
    reference_rows = min(args.rows, 8)
    reference = logits[:reference_rows, : args.length].float().topk(args.topk).values
    actual = logits[:reference_rows].gather(1, output[:reference_rows].long()).float()
    torch.testing.assert_close(
        actual.sort(descending=True).values, reference, rtol=0, atol=0
    )
    assert bool(((output >= 0) & (output < ends[:, None])).all())
    ordered = output.sort(1).values
    assert not bool((ordered[:, 1:] == ordered[:, :-1]).any())
    torch.accelerator.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    run()
    torch.accelerator.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    native = Path(vllm.__file__).parent / "_deepselect_C.abi3.so"
    record = dict(
        status="ok",
        record_kind="counter_probe",
        unix_time=time.time(),
        hostname=socket.gethostname(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        vllm_root=str(Path(vllm.__file__).parent),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=args.rows,
        width=args.width,
        stride=logits.stride(0),
        length=args.length,
        topk=args.topk,
        correctness_reference_rows=reference_rows,
        bounds_and_uniqueness_rows=args.rows,
        minimum_score_read_bytes=2 * args.rows * args.length,
        note="Counter probe only; profiler timing is not a CUPTI benchmark result.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


if __name__ == "__main__":
    main()
