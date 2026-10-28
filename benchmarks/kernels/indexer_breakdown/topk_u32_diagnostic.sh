#!/usr/bin/env bash
set -euo pipefail
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export CUDA_VISIBLE_DEVICES="${TOPK_GPU:-3}"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4
.venv/bin/python - <<'PY'
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

import torch

import vllm._custom_ops  # noqa: F401
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.v1.worker.workspace import init_workspace_manager

init_workspace_manager(torch.device("cuda", 0))
rows, width, stride, length, k = 4096, 1048576, 1049088, 4096, 512
storage = torch.full((rows, stride), -100.0, device="cuda", dtype=torch.float32)
scores = storage[:, :width]
columns = torch.arange(length, device="cuda", dtype=torch.float32)
row_ids = torch.arange(rows, device="cuda", dtype=torch.float32)
scores[:, :length] = (columns[None] + row_ids[:, None] * 997).remainder(length)
lengths = torch.full((rows, 1), length, device="cuda", dtype=torch.int32)
out = torch.empty((rows, k), device="cuda", dtype=torch.int32)
metadata = {
    "status": "diagnostic",
    "record_type": "uint32_row_address_check",
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "script_sha256": hashlib.sha256(
        Path("benchmarks/kernels/indexer_breakdown/topk_u32_diagnostic.sh").read_bytes()
    ).hexdigest(),
    "hostname": socket.gethostname(),
    "gpu": torch.cuda.get_device_name(),
    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    "rows": rows,
    "capacity": width,
    "stride": stride,
    "topk": k,
    "dtype": "float32",
    "effective_length": length,
    "input_formula": "valid(r,c)=(c+997*r)%4096; padding and invalid=-100",
}
destination = Path(
    "benchmarks/kernels/indexer_breakdown/results/topk_u32_diagnostic.jsonl"
)
for bound in (length, width):
    get_indexer_topk("persistent")(scores, lengths, 1, out, k, bound)
    torch.cuda.synchronize()
    for row in (4093, 4094, 4095):
        offset = row * stride
        wrapped = offset % (1 << 32)
        observed = out[row].long().sort().values
        own = scores[row, :length].topk(k).indices.sort().values
        wrapped_ref = (
            storage.flatten()[wrapped : wrapped + length].topk(k).indices.sort().values
        )
        record = {
            **metadata,
            "max_seq_len": bound,
            "row": row,
            "intended_offset_elements": offset,
            "wrapped_offset_elements": wrapped,
            "matches_own_row": bool(torch.equal(observed, own)),
            "matches_wrapped_address": bool(torch.equal(observed, wrapped_ref)),
            "own_reference_index_range": [int(own.min()), int(own.max())],
            "wrapped_reference_index_range": [
                int(wrapped_ref.min()),
                int(wrapped_ref.max()),
            ],
            "observed_index_range": [int(observed.min()), int(observed.max())],
            "own_reference_overlap_count": int(torch.isin(observed, own).sum()),
            "wrapped_reference_overlap_count": int(
                torch.isin(observed, wrapped_ref).sum()
            ),
        }
        with destination.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(record), flush=True)
PY
