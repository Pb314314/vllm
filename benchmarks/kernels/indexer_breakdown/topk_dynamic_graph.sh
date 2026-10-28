#!/usr/bin/env bash
set -euo pipefail
taskset -pc "${TOPK_CPUSET:-104-119}" $$
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export CUDA_VISIBLE_DEVICES="${TOPK_GPU:-3}"
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/triton
export CUDA_CACHE_PATH=/home/bopang/cg-project/design/idx-breakdown-cache/topk/cuda
export OMP_NUM_THREADS=4
.venv/bin/python - <<'PY'
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import torch

runner_path = Path("benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py")
spec = importlib.util.spec_from_file_location("topk_benchmark", runner_path)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
bench.init_workspace_manager(torch.device("cuda", 0))
python_calls = {"upcast": 0, "topk_provider": 0}
original_upcast = bench.as_fp32_logits
original_topk_provider = bench.get_indexer_topk


def counted_upcast(*args, **kwargs):
    python_calls["upcast"] += 1
    return original_upcast(*args, **kwargs)


def counted_topk_provider(*args, **kwargs):
    python_calls["topk_provider"] += 1
    return original_topk_provider(*args, **kwargs)


bench.as_fp32_logits = counted_upcast
bench.get_indexer_topk = counted_topk_provider
rows, width, k = 8, 1048576, 512
logits = bench.make_input(rows, width, 512, "normal", 123, None)
lengths = torch.full((rows,), 4096, device="cuda", dtype=torch.int32)
out = torch.empty((rows, k), device="cuda", dtype=torch.int32)
metadata = {
    "status": "diagnostic",
    "record_type": "dynamic_graph_inputs",
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "script_sha256": hashlib.sha256(
        Path("benchmarks/kernels/indexer_breakdown/topk_dynamic_graph.sh").read_bytes()
    ).hexdigest(),
    "benchmark_sha256": hashlib.sha256(runner_path.read_bytes()).hexdigest(),
    "hostname": socket.gethostname(),
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "rows": rows,
    "capacity": width,
    "stride": logits.stride(0),
    "topk": k,
    "captured_host_max_seq_len": width,
    "capture_initial_effective_length": 4096,
    "logits_data_ptr": logits.data_ptr(),
    "lengths_data_ptr": lengths.data_ptr(),
}
destination = Path(
    "benchmarks/kernels/indexer_breakdown/results/topk_dynamic_graph.jsonl"
)
for scheme in ("DS-bf16", "UP+COOP", "UP+PERS"):
    counts_before = dict(python_calls)
    lengths.fill_(4096)
    run, _ = bench.make_scheme(scheme, logits, lengths, out, k, width)
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    counts_at_capture = {
        name: python_calls[name] - counts_before[name] for name in python_calls
    }
    for length, negate in [
        (0, False),
        (128, False),
        (8192, False),
        (32768, False),
        (131072, False),
        (width, False),
        (4096, False),
        (4096, True),
    ]:
        lengths.fill_(length)
        if negate:
            previous = out.clone()
            logits.neg_()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        result = bench.correctness(logits, lengths, out, k, rows)
        trace = bench.cupti_kernel_details(graph.replay, 1, None)
        record = {
            **metadata,
            "unix_time": time.time(),
            "scheme": scheme,
            "effective_length": length,
            "input_negated": negate,
            "correctness": result,
            "kernel_names": sorted(trace["kernels"]),
            "launch_configs": {
                name: data["launch_config"] for name, data in trace["kernels"].items()
            },
            "python_calls_warmup_and_capture": counts_at_capture,
            "python_calls_after_replays": {
                name: python_calls[name] - counts_before[name] for name in python_calls
            },
        }
        if negate:
            record["changed_index_set_rows"] = int(
                (previous.sort(1).values != out.sort(1).values).any(1).sum()
            )
            logits.neg_()
        with destination.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(record), flush=True)
    del graph, run
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
PY
