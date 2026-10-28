#!/usr/bin/env bash
set -euo pipefail
repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$repo_root"
.venv/bin/python - <<'PY'
import collections
import hashlib
import json
import pathlib
import statistics

root = pathlib.Path("benchmarks/kernels/indexer_breakdown/results")
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
    "gpu_uuid",
    "hostname",
)
errors = (
    "wrong_count_rows",
    "wrong_sentinel_rows",
    "out_of_bounds_rows",
    "duplicate_rows",
    "value_multiset_mismatch_rows",
)
aggregate = []
inventory = []
for path in sorted(root.glob("topk*.jsonl")):
    if any(
        word in path.stem
        for word in (
            "summary",
            "archive",
            "sources",
            "attempts",
            "diagnostic",
            "threshold",
            "failures",
            "unsafe",
            "dynamic",
        )
    ):
        continue
    raw = path.read_bytes()
    rows = []
    for line in raw.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    entries = [r for r in rows if r.get("status") == "ok" and r.get("key")]
    if not entries:
        continue
    groups = collections.defaultdict(list)
    sentinel = []
    invalid = 0
    for r in entries:
        groups[tuple(r.get(f) for f in fields)].append(r)
        invalid += any(
            (r.get(check) or {}).get(e, 0)
            for e in errors
            for check in (
                "correctness",
                "replay_correctness",
                "changed_lengths_replay_correctness",
            )
        )
    for r in rows:
        if r.get("record_type") == "sentinel" and r.get("gpu_span_us"):
            sentinel.append(r["gpu_span_us"]["median"])
    source = {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "lines": len(rows),
    }
    unstable = 0
    for key, group in groups.items():
        rounds = collections.defaultdict(list)
        for r in group:
            rounds[bool(r.get("rerun_unstable"))].append(r)
        chosen = rounds.get(True) or rounds[False]
        selected = sorted(chosen, key=lambda r: r.get("trial", 0))[-3:]
        medians = [r["gpu_span_us"]["median"] for r in selected]
        cv = (
            statistics.pstdev(medians) / statistics.mean(medians)
            if len(medians) > 1
            else 0
        )
        unstable += cv > 0.03
        checks = {
            check: {
                e: sum((r.get(check) or {}).get(e, 0) for r in selected)
                for e in errors
                + ("index_set_mismatch_rows", "ties_only_different_index_rows")
            }
            for check in (
                "correctness",
                "replay_correctness",
                "changed_lengths_replay_correctness",
            )
        }
        valid = not any(checks[c][e] for c in checks for e in errors)
        out = {
            "status": "summary",
            "source": source,
            **dict(zip(fields, key)),
            "trials_available": len(group),
            "selected_trial_numbers": [r.get("trial") for r in selected],
            "selected_keys": [r["key"] for r in selected],
            "selected_rerun": bool(selected[0].get("rerun_unstable")),
            "complete_three_trials": len(selected) == 3,
            "trial_cv_population": cv,
            "valid_selection": valid,
            "cold_flush_meets_explicit_2xl2": selected[0]["cache"] != "cold"
            or bool(selected[0].get("public_cold_flush_bytes")),
            "correctness_totals": checks,
            "gpu_span_us_median": statistics.median(medians),
            "recorded_worktree_git": selected[0].get("git_sha"),
            "vllm_python_root": selected[0].get("vllm_python_root"),
            "native_artifacts": selected[0].get("native_artifacts"),
            "initial_trial_medians_us": [
                r["gpu_span_us"]["median"] for r in rounds[False]
            ],
            "input_values": selected[0].get("input_values"),
            "actual_max_row_length": selected[0].get("actual_max_row_length"),
            "valid_elements": selected[0].get("valid_elements"),
            "upcast_useful_traffic_bytes": selected[0].get(
                "upcast_useful_traffic_bytes"
            ),
            "input_source": selected[0].get("input_source"),
        }
        for field, target in (
            ("host_enqueue_us", "host_enqueue_us_median"),
            ("wall_us", "wall_us_median"),
        ):
            vals = [r[field]["median"] for r in selected if field in r]
            out[target] = statistics.median(vals) if vals else None
        vals = [
            r["kernel_breakdown"]["gpu_kernel_sum_us"]["median"]
            for r in selected
            if r.get("kernel_breakdown", {}).get("gpu_kernel_sum_us")
        ]
        out["gpu_kernel_sum_us_median"] = statistics.median(vals) if vals else None
        out["additional_peak_allocated_bytes_median"] = statistics.median(
            r["memory"]["additional_peak_allocated_bytes"] for r in selected
        )
        out["reserved_delta_bytes_median"] = statistics.median(
            r["memory"]["reserved_delta_bytes"] for r in selected
        )
        aggregate.append(out)
    inventory.append(
        {
            "status": "inventory",
            "source": source,
            "statuses": dict(collections.Counter(r.get("status") for r in rows)),
            "successful_timed_records": len(entries),
            "groups": len(groups),
            "rows": sorted({r["rows"] for r in entries}),
            "capacities": sorted({r["capacity"] for r in entries}),
            "topk": sorted({r["topk"] for r in entries}),
            "profiles": sorted({r["length_profile"] for r in entries}),
            "schemes": sorted({r["scheme"] for r in entries}),
            "executions": sorted({r["execution"] for r in entries}),
            "caches": sorted({r["cache"] for r in entries}),
            "invalid_timed_records": invalid,
            "rejected_before_timing": sum(
                r.get("status") == "invalid_selection" for r in rows
            ),
            "selected_groups_cv_gt_3_percent": unstable,
            "sentinel_medians_us": sentinel,
        }
    )
pair_fields = tuple(
    f for f in fields if f not in ("scheme", "bound_kind", "max_seq_len")
)
paired = collections.defaultdict(list)
for entry in aggregate:
    paired[(entry["source"]["path"], *[entry.get(f) for f in pair_fields])].append(
        entry
    )
comparisons = []
for key, group in paired.items():
    expected = "UP+COOP" if group[0]["rows"] <= 64 else "UP+PERS"
    baselines = [e for e in group if e["scheme"] == expected]
    alternatives = [
        e
        for e in group
        if e["scheme"] in ("DS-bf16", "UP+DS-fp32", "UP+FI", "UP+PERROW")
    ]
    for baseline in baselines:
        for alternative in alternatives:
            comparisons.append(
                {
                    "status": "comparison",
                    "source": baseline["source"],
                    **{f: baseline.get(f) for f in pair_fields},
                    "baseline": expected,
                    "baseline_bound_kind": baseline["bound_kind"],
                    "baseline_max_seq_len": baseline["max_seq_len"],
                    "alternative": alternative["scheme"],
                    "baseline_gpu_span_us": baseline["gpu_span_us_median"],
                    "alternative_gpu_span_us": alternative["gpu_span_us_median"],
                    "latency_delta_percent": 100
                    * (
                        alternative["gpu_span_us_median"]
                        / baseline["gpu_span_us_median"]
                        - 1
                    ),
                    "additional_peak_saved_mib": (
                        baseline["additional_peak_allocated_bytes_median"]
                        - alternative["additional_peak_allocated_bytes_median"]
                    )
                    / 2**20,
                    "reserved_delta_saved_mib": (
                        baseline["reserved_delta_bytes_median"]
                        - alternative["reserved_delta_bytes_median"]
                    )
                    / 2**20,
                    "both_selection_valid": baseline["valid_selection"]
                    and alternative["valid_selection"],
                    "both_flush_protocol_valid": baseline[
                        "cold_flush_meets_explicit_2xl2"
                    ]
                    and alternative["cold_flush_meets_explicit_2xl2"],
                    "both_three_trials": baseline["complete_three_trials"]
                    and alternative["complete_three_trials"],
                    "either_cv_gt_3_percent": max(
                        baseline["trial_cv_population"],
                        alternative["trial_cv_population"],
                    )
                    > 0.03,
                    "baseline_keys": baseline["selected_keys"],
                    "alternative_keys": alternative["selected_keys"],
                }
            )
for name, data in [
    ("topk_summary_all.jsonl", aggregate),
    ("topk_summary_inventory.jsonl", inventory),
    ("topk_summary_comparisons.jsonl", comparisons),
]:
    (root / name).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
    )
print(
    json.dumps(
        {
            "files": len(inventory),
            "groups": len(aggregate),
            "valid_groups": sum(r["valid_selection"] for r in aggregate),
            "complete_three_trial_groups": sum(
                r["complete_three_trials"] for r in aggregate
            ),
            "noisy_groups": sum(r["trial_cv_population"] > 0.03 for r in aggregate),
        },
        indent=2,
    )
)
for row in inventory:
    print(
        pathlib.Path(row["source"]["path"]).name,
        row["successful_timed_records"],
        row["groups"],
        row["invalid_timed_records"],
        row["selected_groups_cv_gt_3_percent"],
    )
PY
