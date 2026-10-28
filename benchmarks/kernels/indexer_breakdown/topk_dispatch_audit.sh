#!/usr/bin/env bash
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
.venv/bin/python - <<'PY'
import collections
import hashlib
import json
import pathlib
import statistics

root = pathlib.Path("benchmarks/kernels/indexer_breakdown/results")
a = [json.loads(s) for s in (root / "topk_summary_all.jsonl").open()]
models = {
    "c1": (1048576, 512, 1, "285171eb"),
    "c2": (524288, 512, 2, "285171eb"),
    "v4": (262144, 512, 4, "60303add"),
    "v32": (163840, 2048, 1, "505833f3"),
    "v4pro": (262144, 1024, 4, "a0ada22c"),
}
allrows = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    14,
    16,
    20,
    24,
    28,
    32,
    40,
    48,
    56,
    60,
    62,
    63,
    64,
    65,
    66,
    72,
    80,
    96,
    112,
    128,
    160,
    192,
    256,
    320,
    384,
    512,
    640,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
]
profiles = ["short", "mid", "long", "full", "ragged", "spec"]
results = []
for name, (width, k, comp, uuid) in models.items():
    selected = [
        x
        for x in a
        if x["capacity"] == width
        and x["topk"] == k
        and x["compress_ratio"] == comp
        and (
            str(x["gpu_uuid"]).startswith(uuid)
            or (
                x["gpu_uuid"] is None
                and name in ("c1", "c2")
                and pathlib.Path(x["source"]["path"]).name
                == (
                    "topk_v41c1_core.jsonl" if name == "c1" else "topk_v41c2_core.jsonl"
                )
            )
        )
        and x["execution"] == "graph"
        and x["cache"] == "warm"
        and x["distribution"] == "normal"
        and x.get("input_source") == "synthetic"
        and "newnative" not in x["source"]["path"]
        and x["rows"] in allrows
    ]
    groups = collections.defaultdict(list)
    for x in selected:
        groups[(x["rows"], x["length_profile"])].append(x)
    points = []
    missing = []
    bound = 163840 if name == "v32" else 1048576
    for r in allrows:
        for profile in profiles:
            group = groups.get((r, profile), [])
            ds = [
                x
                for x in group
                if x["scheme"] == "DS-bf16"
                and x["complete_three_trials"]
                and x["valid_selection"]
            ]
            baseline = [
                x
                for x in group
                if x["scheme"] == ("UP+COOP" if r <= 64 else "UP+PERS")
                and x["complete_three_trials"]
                and x["valid_selection"]
                and (r <= 64 or x["max_seq_len"] == bound)
            ]
            if not ds or not baseline:
                missing.append(
                    {
                        "rows": r,
                        "profile": profile,
                        "missing_ds": not ds,
                        "missing_baseline": not baseline,
                    }
                )
                continue
            d = ds[0]
            b = sorted(baseline, key=lambda x: x["bound_kind"] != "actual")[0]
            bv = b["gpu_span_us_median"]
            dv = d["gpu_span_us_median"]
            cv = dv if r >= 16 and k in (512, 1024) else bv
            opt = min(bv, dv)
            points.append(
                {
                    "rows": r,
                    "profile": profile,
                    "auto_us": bv,
                    "ds_us": dv,
                    "candidate_us": cv,
                    "saving_us": bv - cv,
                    "saving_percent": 100 * (1 - cv / bv),
                    "regret_us": cv - opt,
                    "regret_percent": 100 * (cv / opt - 1),
                    "noisy": max(b["trial_cv_population"], d["trial_cv_population"])
                    > 0.03,
                    "auto_keys": b["selected_keys"],
                    "ds_keys": d["selected_keys"],
                    "auto_source": b["source"],
                    "ds_source": d["source"],
                }
            )
    budget_points = [x for x in points if x["rows"] <= (512 * 2**20) // (6 * width)]
    report = {
        "status": "dispatch_audit",
        "model": name,
        "width": width,
        "k": k,
        "compression": comp,
        "expected_row_profile_pairs": 270,
        "paired_points": len(points),
        "missing": missing,
        "median_saved_us": statistics.median(x["saving_us"] for x in points),
        "median_saved_percent": statistics.median(x["saving_percent"] for x in points),
        "worst_regret": max(points, key=lambda x: x["regret_percent"]),
        "worst_slowdown_vs_auto": min(points, key=lambda x: x["saving_percent"]),
        "r_ge16_slowdown_points": sum(
            x["saving_us"] < 0 for x in points if x["rows"] >= 16
        ),
        "budget_rows_max": (512 * 2**20) // (6 * width),
        "budget_points": len(budget_points),
        "budget_median_saved_us": statistics.median(
            x["saving_us"] for x in budget_points
        ),
        "budget_median_saved_percent": statistics.median(
            x["saving_percent"] for x in budget_points
        ),
        "budget_worst_regret": max(budget_points, key=lambda x: x["regret_percent"]),
        "points": points,
    }
    results.append(report)
    print(
        name,
        "pairs",
        len(points),
        "missing",
        len(missing),
        "median saving",
        round(report["median_saved_us"], 3),
        round(report["median_saved_percent"], 2),
        "regret",
        {
            f: report["worst_regret"][f]
            for f in ("rows", "profile", "regret_us", "regret_percent")
        },
        "bad R>=16",
        report["r_ge16_slowdown_points"],
    )
(root / "topk_summary_dispatch.jsonl").write_text(
    "".join(json.dumps(x) + "\n" for x in results)
)
counts = collections.Counter()
legacy_uuid = collections.Counter()
duplicates = []
bad = []
timed = 0
required = [
    "rows",
    "capacity",
    "stride",
    "topk",
    "scheme",
    "hostname",
    "git_sha",
    "script_sha256",
    "gpu_span_us",
    "kernel_breakdown",
    "correctness",
    "memory",
    "telemetry_before",
    "telemetry_after",
]
for path in sorted(root.glob("topk*.jsonl")):
    if any(
        w in path.stem
        for w in (
            "summary",
            "archive",
            "sources",
            "manifest",
            "diagnostic",
            "threshold",
            "unsafe",
            "failures",
            "attempts",
            "dynamic",
        )
    ):
        continue
    keys = set()
    for n, line in enumerate(path.open(), 1):
        try:
            x = json.loads(line)
        except ValueError:
            bad.append({"file": str(path), "line": n, "issue": "invalid JSON"})
            continue
        counts[x.get("status")] += 1
        if x.get("key") and x.get("status") == "ok":
            timed += 1
            if x["key"] in keys:
                duplicates.append({"file": str(path), "key": x["key"]})
            keys.add(x["key"])
            if not x.get("gpu_uuid"):
                legacy_uuid[path.name] += 1
            missing = [f for f in required if f not in x]
            if missing:
                bad.append({"file": str(path), "line": n, "missing": missing})
report = {
    "status": "schema_audit",
    "timed_rows": timed,
    "status_counts": dict(counts),
    "errors": bad,
    "duplicate_timed_keys": duplicates,
    "legacy_missing_uuid_by_file": dict(legacy_uuid),
    "summary_sha256": hashlib.sha256(
        (root / "topk_summary_all.jsonl").read_bytes()
    ).hexdigest(),
}
recovered = {}
for filename in (
    "topk_newnative_v32.jsonl",
    "topk_newnative_v32_attempts.jsonl",
    "topk_newnative_v4pro.jsonl",
    "topk_newnative_v4pro_attempts.jsonl",
):
    path = root / filename
    if not path.exists():
        continue
    rows = [json.loads(line) for line in path.open()]
    keys = [x["key"] for x in rows if x.get("key")]
    groups = collections.defaultdict(set)
    for x in rows:
        if x.get("status") == "ok" and x.get("key") and "trial" in x:
            group = tuple(
                x.get(f)
                for f in (
                    "rows",
                    "capacity",
                    "topk",
                    "length_profile",
                    "scheme",
                    "bound_kind",
                    "max_seq_len",
                )
            )
            groups[group].add(x["trial"])
    recovered[filename] = {
        "lines": len(rows),
        "duplicate_keys": len(keys) - len(set(keys)),
        "successful_trial_counts": dict(
            collections.Counter(len(v) for v in groups.values())
        ),
    }
report["recovered_anchor_audit"] = recovered
all_json_errors = []
for path in root.glob("topk*.jsonl"):
    for n, line in enumerate(path.open(), 1):
        try:
            json.loads(line)
        except ValueError:
            all_json_errors.append({"file": str(path), "line": n})
report["all_topk_jsonl_parse_errors"] = all_json_errors
(root / "topk_summary_schema_audit.jsonl").write_text(json.dumps(report) + "\n")
print(
    "SCHEMA",
    timed,
    "timed rows;",
    len(bad),
    "missing/invalid JSON; duplicate keys",
    len(duplicates),
    "legacy UUID",
    dict(legacy_uuid),
    dict(counts),
)
PY
