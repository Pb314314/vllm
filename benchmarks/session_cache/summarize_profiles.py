# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import collections
import gzip
import json
import pstats
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root
report = {}
for mode in ["apc", "session"]:
    folder = root / mode
    profiles = {
        name: pstats.Stats(str(folder / (name + ".pstats")))
        for name in ["engine-input", "engine-loop", "frontend"]
    }
    functions = {}
    wanted = {
        "request_block_hasher",
        "hash_block_tokens",
        "find_longest_cache_hit",
        "touch",
        "free_blocks",
        "_bind_session",
        "retain",
        "resume",
        "add_request",
        "apply_staged_writes",
        "poll",
        "encode",
        "decode",
        "extend_from_slice",
    }
    for label, profile in profiles.items():
        functions[label] = [
            {
                "file": k[0].split("/vllm/")[-1],
                "function": k[2],
                "calls": v[1],
                "self_ms": v[2] * 1000,
                "inclusive_ms": v[3] * 1000,
            }
            for k, v in profile.stats.items()
            if k[2] in wanted
        ]
    trace_path = next((folder / "torch").glob("*.gz"))
    events = json.loads(gzip.decompress(trace_path.read_bytes()))["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    gpu_work = [
        e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in gpu_work)
    total, start, end = 0, *intervals[0]
    for a, b in intervals[1:]:
        if a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    total += end - start
    byname = collections.defaultdict(lambda: [0, 0])
    for e in kernels:
        byname[e["name"]][0] += 1
        byname[e["name"]][1] += e["dur"] / 1000
    steps = []
    for e in events:
        if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("execute_"):
            contained = [k for k in kernels if e["ts"] <= k["ts"] < e["ts"] + e["dur"]]
            steps.append(
                {
                    "name": e["name"],
                    "range_ms": e["dur"] / 1000,
                    "kernel_sum_ms": sum(k["dur"] for k in contained) / 1000,
                    "kernels": len(contained),
                }
            )
    host = json.loads((folder / "torch-host-spans.json").read_text())
    report[mode] = {
        "functions": functions,
        "host_spans": {
            k: {"calls": len(v), "inclusive_ms": sum(v)}
            for k, v in host["host_inclusive_ms"].items()
        },
        "first_schedule": host["schedule_shapes"][0],
        "gpu": {
            "kernels": len(kernels),
            "kernel_sum_ms": sum(e["dur"] for e in kernels) / 1000,
            "busy_union_ms": total / 1000,
            "first_to_last_ms": (max(b for a, b in intervals) - intervals[0][0]) / 1000,
            "steps": steps,
            "top_kernels": sorted(
                [{"name": k, "calls": v[0], "ms": v[1]} for k, v in byname.items()],
                key=lambda x: -x["ms"],
            )[:10],
        },
    }
(root / "profile-summary.json").write_text(json.dumps(report, indent=2))
for mode, r in report.items():
    print(
        mode,
        "gpu",
        {k: v for k, v in r["gpu"].items() if k not in ["steps", "top_kernels"]},
    )
    print("prefill", [s for s in r["gpu"]["steps"] if "context_1" in s["name"]])
    print("host", r["host_spans"])
    print("input", r["functions"]["engine-input"])
    print("first schedule", r["first_schedule"])
