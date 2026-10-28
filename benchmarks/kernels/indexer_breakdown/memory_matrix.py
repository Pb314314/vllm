# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Complete mixed graph-mode and default-budget memory diagnostics on one node."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--only-deepselect", action="store_true")
    args = parser.parse_args()
    while args.wait_pid:
        try:
            os.kill(args.wait_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(5)
    tree = args.tree
    common = [
        str(tree / ".venv/bin/python"),
        str(Path(__file__).with_name("memory_serve.py")),
        "--tree",
        str(tree),
        "--venv",
        str(tree / ".venv"),
        "--tp",
        "4",
        "--gpus",
        "0,1,2,3",
        "--out-dir",
        str(args.root / "models"),
        "--snapshot-dir",
        str(args.root / "model_snapshots"),
        "--cache-dir",
        "/home/bopang/cg-project/design/idx-breakdown-cache/memory",
        "--results",
        str(
            tree
            / "benchmarks/kernels/indexer_breakdown/results/memory_model_extra.jsonl"
        ),
    ]
    cases = [
        [
            "--tag",
            "dsv4_mixed_tp4",
            "--modes",
            "FULL_AND_PIECEWISE",
            "--max-lens",
            "262144,1048576",
            "--bursts",
            "8,128,1024",
        ],
        [
            "--tag",
            "dsv4_stress90_tp4",
            "--modes",
            "PIECEWISE",
            "--max-lens",
            "1048576",
            "--bursts",
            "1024",
            "--max-tokens",
            "256",
            "--gpu-memory-utilization",
            "0.90",
            "--runtime-peaks",
        ],
    ]
    if args.only_deepselect:
        cases = [
            [
                "--tag",
                "dsv4_deepselect_tp4",
                "--modes",
                "PIECEWISE",
                "--max-lens",
                "1048576",
                "--bursts",
                "8,128,1024",
                "--extra-args",
                json.dumps(
                    [
                        "--kernel-config",
                        json.dumps(
                            dict(
                                sparse_indexer_topk_backend="deep_select",
                                moe_backend="deep_gemm_mega_moe",
                            )
                        ),
                    ]
                ),
            ]
        ]
    for flags in cases:
        command = common + flags
        print("START", command, flush=True)
        result = subprocess.run(command, check=False)
        print("END", flags[1], result.returncode, flush=True)


if __name__ == "__main__":
    main()
