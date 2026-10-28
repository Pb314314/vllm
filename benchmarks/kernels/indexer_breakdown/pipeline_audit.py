# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit accepted pipeline records without changing raw measurements."""

import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    directory = Path(__file__).parent / "results"
    failures = []
    reports = []
    checks = (
        "chunked_logit_mismatch_rows",
        "nonfinite_valid_logits",
        "invalid_index_rows",
        "duplicate_index_rows",
        "exact_topk_value_mismatch_rows",
    )
    for path in sorted(directory.glob("pipeline*.jsonl")):
        if any(word in path.name for word in ("telemetry", "power_case", "audit")):
            continue
        raw = path.read_bytes()
        records = [json.loads(line) for line in raw.splitlines()]
        accepted = [r for r in records if r.get("status") == "ok"]
        for index, record in enumerate(records):
            if record.get("status") == "error" and record.get("timings"):
                failures.append([path.name, index, "rejected record has timing"])
            if record.get("status") != "ok" or "correctness" not in record:
                continue
            check = record["correctness"]
            bad = {name: check[name] for name in checks if check.get(name, 0)}
            bad.update(
                {
                    f"candidate/{name}": value
                    for name, value in check.get("candidate_check", {}).items()
                    if value
                }
            )
            if bad:
                failures.append([path.name, index, bad])
            if (
                "newnative" in path.name
                and "probe" not in path.name
                and record.get("variant", "").startswith("final-")
                and not record.get("production_helper_for_final")
            ):
                failures.append([path.name, index, "missing production helper"])
        noisy = sum(
            any(v.get("noisy", False) for v in r.get("timings", {}).values())
            for r in accepted
        )
        reports.append(
            dict(
                file=path.name,
                sha256=hashlib.sha256(raw).hexdigest(),
                records=len(records),
                status=dict(Counter(r.get("status", "unspecified") for r in records)),
                accepted_with_noisy_gpu_trial=noisy,
            )
        )
    result = dict(record_kind="audit", files=reports, failures=failures)
    (directory / "pipeline_audit.jsonl").write_text(json.dumps(result) + "\n")
    for report in reports:
        print(report["file"], report["records"], report["status"])
    print("Audit failures:", failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
