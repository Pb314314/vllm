# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the synthetic cache layout against a CPU indexer-score formula."""

import argparse
import json
import socket
from pathlib import Path

import torch
from pipeline_bench import MODELS, init_workspace_manager, logits_for_rows, make_inputs


def unpack_fp4(packed):
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    return values[(nibbles & 7).long()] * torch.where(nibbles < 8, 1, -1)


def native_head_reduce(dots, weights):
    activated = torch.relu(dots).to(torch.bfloat16).float()
    lanes = torch.zeros((4, dots.shape[1]), dtype=torch.float32)
    for head in range(dots.shape[0]):
        lane = head % 4
        lanes[lane] = (
            (activated[head] * weights[head] + lanes[lane]).to(torch.bfloat16).float()
        )
    low = (lanes[0] + lanes[2]).to(torch.bfloat16).float()
    high = (lanes[1] + lanes[3]).to(torch.bfloat16).float()
    return (low + high).to(torch.bfloat16).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    init_workspace_manager(torch.device("cuda"))
    failures = 0
    for model, (heads, width, compression, _) in MODELS.items():
        for dtype in ("mxfp4", "fp8"):
            for q_len in (1, 3, 8):
                x = make_inputs(64, heads, width, compression, 8192, dtype, q_len, 512)
                full = logits_for_rows(x, 0, 64, x["schedule"])
                page_bytes = x["kv"].stride(0)
                raw = x["kv"].as_strided(
                    (x["kv"].shape[0], page_bytes), (page_bytes, 1)
                )
                references, actuals, ideals = [], [], []
                for row in (0, 31, 63):
                    length = int(x["ctx"][row, 0])
                    columns = torch.tensor(
                        [0, 1, 2, 63, 64, 65, length - 2, length - 1], device="cuda"
                    )
                    pages = x["bt"][row, columns // 64].long()
                    offsets = columns % 64
                    if dtype == "mxfp4":
                        keys = raw[pages, :4096].view(-1, 64, 64)
                        keys = keys[torch.arange(len(columns), device="cuda"), offsets]
                        key_values = unpack_fp4(keys.cpu())
                        query = unpack_fp4(x["q"][row, 0].view(torch.uint8).cpu())
                    else:
                        keys = raw[pages, :8192].view(-1, 64, 128)
                        key_values = (
                            keys[torch.arange(len(columns), device="cuda"), offsets]
                            .view(torch.float8_e4m3fn)
                            .float()
                            .cpu()
                        )
                        query = x["q"][row, 0].float().cpu()
                    weights = x["weights"][row].float().cpu()
                    dots = query @ key_values.T
                    reference = native_head_reduce(dots, weights)
                    ideals.append((torch.relu(dots) * weights[:, None]).sum(0))
                    references.append(reference)
                    actuals.append(full[row, columns].float().cpu())
                reference, actual = torch.cat(references), torch.cat(actuals)
                ideal = torch.cat(ideals)
                close = torch.isclose(actual, reference, rtol=0.005, atol=0.001)
                record = dict(
                    model=model,
                    precision=dtype,
                    q_len=q_len,
                    rows=64,
                    raw_context=8192,
                    capacity=width,
                    heads=heads,
                    head_dim=128,
                    sampled_rows=[0, 31, 63],
                    sample_count=actual.numel(),
                    rtol=0.005,
                    atol=0.001,
                    mismatch_count=int((~close).sum()),
                    max_absolute_error=float((actual - reference).abs().max()),
                    bitwise_equal_to_native_reference=bool((actual == reference).all()),
                    ideal_fp32_max_absolute_error=float((actual - ideal).abs().max()),
                    ideal_fp32_relative_l2_error=float(
                        (actual - ideal).norm() / ideal.norm()
                    ),
                    formula="sum_h weights[q,h] * relu(dot(Q[q,h,:], K[k,:]))",
                    native_reduction=(
                        "BF16 ReLU conversion, four FMA lanes, BF16 pair adds"
                    ),
                    cache_layout="all page values followed by all page UE8M0 scales",
                    scales="unit UE8M0",
                    hostname=socket.gethostname(),
                    status="ok" if bool(close.all()) else "error",
                    actual=actual.tolist(),
                    reference=reference.tolist(),
                    ideal_fp32=ideal.tolist(),
                )
                failures += record["status"] != "ok"
                with args.out.open("a") as output:
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                print(json.dumps(record), flush=True)
                del x, full, raw
                torch.accelerator.empty_cache()
    if failures:
        raise AssertionError(f"CPU math reference failed in {failures} cases")


if __name__ == "__main__":
    main()
