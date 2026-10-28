# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Map PR budget policies to equivalent, already measured pipeline paths."""

import json
from collections import Counter, defaultdict
from pathlib import Path


def expected_path(rows, width, budget_mib, switch_budget_mib=None):
    budget = budget_mib * 1024**2
    switch_budget = (switch_budget_mib or budget_mib) * 1024**2
    if rows * width * 6 <= switch_budget:
        return "auto", [[0, rows]]
    height = max(16, budget // (width * 2) // 16 * 16)
    return "deep_select", [[s, min(s + height, rows)] for s in range(0, rows, height)]


def main():
    directory = Path(__file__).parent / "results"
    cases = defaultdict(dict)
    for path in sorted(directory.glob("pipeline_core_*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["status"] != "ok":
                continue
            case = {k: v for k, v in record["case"].items() if k != "requests"}
            cases[json.dumps(case, sort_keys=True)][record["variant"]] = (path, record)
    statistics = defaultdict(Counter)
    out = directory / "pipeline_policy_frontier.jsonl"
    with out.open("w") as output:
        for case_key, arms in cases.items():
            case = json.loads(case_key)
            original = arms["final-512"][1]
            baseline_us = original["timings"]["warm"]["median_us"]
            policies = [(b, b, "coupled_pr") for b in (256, 512, 1024, 2048)]
            policies.extend((b, 512, "decoupled_counterfactual") for b in (1024, 2048))
            for budget_mib, switch_mib, policy_kind in policies:
                backend, chunks = expected_path(
                    case["rows"], case["width"], budget_mib, switch_mib
                )
                candidates = (
                    "final-512",
                    "single-auto",
                    "single-ds",
                    f"ds-{budget_mib}",
                )
                match = next(
                    (
                        arms[v]
                        for v in candidates
                        if v in arms
                        and arms[v][1]["backend"] == backend
                        and arms[v][1]["chunks"] == chunks
                    ),
                    None,
                )
                assert match is not None, (case, budget_mib, backend, chunks)
                source, measured = match
                latency = measured["timings"]["warm"]["median_us"]
                change = latency / baseline_us - 1
                statistics[f"{policy_kind}/{budget_mib}"][
                    "faster_gt5pct"
                    if change < -0.05
                    else "slower_gt5pct"
                    if change > 0.05
                    else "within5pct"
                ] += 1
                record = dict(
                    record_kind="derived_policy",
                    case=case,
                    budget_mib=budget_mib,
                    switch_budget_mib=switch_mib,
                    policy_kind=policy_kind,
                    backend=backend,
                    chunks=chunks,
                    source_file=source.name,
                    source_key=measured["key"],
                    source_variant=measured["variant"],
                    source_benchmark_sha256=measured["source_sha256"],
                    warm_median_us=latency,
                    cold_median_us=measured["timings"]["cold"]["median_us"],
                    capture_bound_eager_wall_us=measured["timings"]["eager"]["wall"][
                        "median_us"
                    ],
                    peak_extra_bytes=measured["memory"]["peak_extra_bytes"],
                    warm_change_vs_final512=change,
                    provenance=(
                        "Derived by selecting an equivalent measured "
                        "operator sequence; no new GPU timing"
                    ),
                    planner_source=(
                        "PR64138 commit4e5b4f8ce9 coupled policy, "
                        "or explicitly labeled unimplemented counterfactual"
                    ),
                )
                output.write(json.dumps(record) + "\n")
    print("Derived policies:", len(cases) * 6)
    print("Relative to measured final512:", dict(statistics))


if __name__ == "__main__":
    main()
