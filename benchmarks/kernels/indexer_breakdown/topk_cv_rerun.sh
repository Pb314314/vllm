#!/usr/bin/env bash
set -euo pipefail
topk_gpu="${TOPK_GPU:-0}"
case "$topk_gpu" in
    0) topk_cores=0-15 ;;
    1) topk_cores=16-31 ;;
    2) topk_cores=70-85 ;;
    3) topk_cores=104-119 ;;
esac
taskset -pc "${TOPK_CPUSET:-$topk_cores}" $$
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export CUDA_VISIBLE_DEVICES="$topk_gpu"
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/triton
export TORCHINDUCTOR_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/inductor
export XDG_CACHE_HOME=/home/bopang/cg-project/design/idx-breakdown-cache/topk/xdg
export CUDA_CACHE_PATH=/home/bopang/cg-project/design/idx-breakdown-cache/topk/cuda
export OMP_NUM_THREADS=4
.venv/bin/python - <<'PY'
import collections
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path

fields = (
    "rows",
    "capacity",
    "stride",
    "topk",
    "length_profile",
    "distribution",
    "scheme",
    "execution",
    "cache",
    "bound_kind",
    "max_seq_len",
    "seed",
    "repeat",
    "query_len",
    "compress_ratio",
    "logits_source_sha256",
    "file_random_control",
)
gpu = os.environ["CUDA_VISIBLE_DEVICES"]
target_host = os.environ.get("TOPK_CV_HOSTNAME", socket.gethostname())
runner = str(
    Path("benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py").resolve()
)
for path in sorted(
    Path("benchmarks/kernels/indexer_breakdown/results").glob("topk*.jsonl")
):
    if any(
        word in path.stem
        for word in (
            "probe",
            "smoke",
            "summary",
            "archive",
            "sources",
            "diagnostic",
            "thresholds",
        )
    ):
        continue
    records = []
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if (
            entry.get("status") == "ok"
            and entry.get("key")
            and entry.get("repeat") == 100
            and entry.get("hostname") == target_host
            and entry.get("cuda_visible_devices") == gpu
            and not (
                entry.get("cache") == "cold"
                and not entry.get("public_cold_flush_bytes")
            )
            and not any(
                (entry.get(check) or {}).get(error, 0)
                for check in (
                    "correctness",
                    "replay_correctness",
                    "changed_lengths_replay_correctness",
                )
                for error in (
                    "wrong_count_rows",
                    "wrong_sentinel_rows",
                    "out_of_bounds_rows",
                    "duplicate_rows",
                    "value_multiset_mismatch_rows",
                )
            )
        ):
            records.append(entry)
    if not records:
        continue
    if any(entry.get("rerun_unstable") for entry in records):
        print(json.dumps({"file": str(path), "action": "already_retested"}), flush=True)
        continue
    groups = collections.defaultdict(list)
    for entry in records:
        groups[tuple(entry.get(field) for field in fields)].append(
            entry["gpu_span_us"]["median"]
        )
    unstable = sum(
        len(values) >= 3 and statistics.pstdev(values) / statistics.mean(values) > 0.03
        for values in groups.values()
    )
    if not unstable:
        continue
    first = records[0]
    source_root = Path(first.get("vllm_python_root", str(Path.cwd() / "vllm"))).parent
    execution_env = os.environ.copy()
    execution_env["PYTHONPATH"] = str(source_root)
    interpreter = source_root / ".venv/bin/python"
    if not interpreter.exists():
        if source_root.name == "wt-b4a0fix":
            interpreter = source_root.parent / "venv-b4a0fix/bin/python"
        else:
            raise RuntimeError(
                f"No verified interpreter for native source {source_root}"
            )
    deadline = float(os.environ.get("TOPK_CV_DEADLINE", "inf"))
    remaining = min(600, deadline - time.time() - 5)
    if remaining < 30:
        print(json.dumps({"file": str(path), "action": "deadline_reached"}), flush=True)
        break
    command = [
        str(interpreter),
        runner,
        "--preset",
        "manual",
        "--resume",
        "--rerun-unstable",
        "--output",
        str(path.resolve()),
        "--repeat",
        "100",
        "--detail-repeat",
        "100",
        "--trials",
        "3",
        "--host-repeat",
        "10",
        "--check-rows",
        "128",
        "--sentinel-every",
        "200",
        "--max-seconds",
        str(remaining),
        "--seed",
        str(first["seed"]),
        "--query-len",
        str(first["query_len"]),
        "--compress-ratio",
        str(first["compress_ratio"]),
    ]
    for option, field in (
        ("--rows", "rows"),
        ("--capacities", "capacity"),
        ("--top-k", "topk"),
        ("--profiles", "length_profile"),
        ("--distributions", "distribution"),
        ("--schemes", "scheme"),
        ("--modes", "execution"),
        ("--caches", "cache"),
        ("--bounds", "bound_kind"),
    ):
        command += [option, *map(str, sorted({entry[field] for entry in records}))]
    capture = [
        entry["max_seq_len"] for entry in records if entry["bound_kind"] == "capture"
    ]
    command += [
        "--capture-max-seq-len",
        str(max(capture or [entry["max_seq_len"] for entry in records])),
    ]
    for option, field in (
        ("--stride-padding", "stride_padding_override"),
        ("--raw-max-seq-len", "raw_max_seq_len_override"),
    ):
        if first.get(field) is not None:
            command += [option, str(first[field])]
    if first["input_source"] != "synthetic":
        command += ["--logits-file", first["input_source"]]
    if first.get("file_random_control"):
        command += ["--file-random-control"]
    if first.get("replay_check_enabled"):
        command += ["--replay-check"]
    print(
        json.dumps(
            {
                "file": str(path),
                "unstable_groups": unstable,
                "action": "rerun",
                "command": command,
            }
        ),
        flush=True,
    )
    if os.environ.get("TOPK_CV_DRY_RUN") != "1":
        subprocess.run(command, check=True, cwd=source_root, env=execution_env)
PY
