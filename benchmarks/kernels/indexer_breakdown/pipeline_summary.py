# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Print compact pipeline tables from append-only JSONL measurements."""

import json
from collections import Counter
from pathlib import Path


def main():
    directory = Path(__file__).parent / "results"
    records = []
    for path in sorted(directory.glob("pipeline_core*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    counts = Counter(record["status"] for record in records)
    print(f"Core records: {dict(counts)}")
    good = [r for r in records if r["status"] == "ok"]
    print(
        "Noisy final trials:",
        sum(any(v.get("noisy", False) for v in r["timings"].values()) for r in good),
    )
    lookup = {
        (
            r["case"]["model"],
            r["case"]["dtype"],
            r["case"]["context_tokens"],
            r["case"]["rows"],
            r["variant"],
        ): r
        for r in good
    }
    print("\nR8192, topk512, q8. Complete-pipeline microseconds; warm L2.")
    print(
        "| Shape | Precision | Context | Old auto | Single DS | PR 512 | "
        "GPU change | Eager wall old/PR | Peak MiB old/PR |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in ("v4", "v41-c2", "v41-c1"):
        for dtype in ("mxfp4", "fp8"):
            for context in (1536, 8192, 32768):
                keys = [
                    (model, dtype, context, 8192, v)
                    for v in ("single-auto", "single-ds", "final-512")
                ]
                if not all(k in lookup for k in keys):
                    continue
                old, single, new = [lookup[k] for k in keys]
                a, b, c = [
                    r["timings"]["warm"]["median_us"] for r in (old, single, new)
                ]
                eager = [r["timings"]["eager"]["wall"]["median_us"] for r in (old, new)]
                memory = [r["memory"]["peak_extra_bytes"] / 1024**2 for r in (old, new)]
                print(
                    f"| {model} | {dtype} | {context} | {a:.2f} | {b:.2f} | "
                    f"{c:.2f} | {(c / a - 1) * 100:+.1f}% | "
                    f"{eager[0]:.1f}/{eager[1]:.1f} | "
                    f"{memory[0]:.2f}/{memory[1]:.2f} |"
                )
    print("\nR8192, context8192, warm graph budget ablation (us).")
    variants = [
        "single-auto",
        "single-ds",
        "ds-256",
        "ds-512",
        "ds-1024",
        "ds-2048",
        "prebuilt-512",
    ]
    print("| Shape | Precision | " + " | ".join(variants) + " |")
    print("| --- | --- | " + " | ".join("---:" for _ in variants) + " |")
    for model in ("v4", "v41-c2", "v41-c1"):
        for dtype in ("mxfp4", "fp8"):
            row = [lookup.get((model, dtype, 8192, 8192, v)) for v in variants]
            values = [
                f"{r['timings']['warm']['median_us']:.2f}" if r else "—" for r in row
            ]
            print(f"| {model} | {dtype} | " + " | ".join(values) + " |")
    mismatches = Counter()
    for r in good:
        c = r["correctness"]
        if r["variant"] == "single-auto":
            mismatches["auto_rows"] += c["rows"]
            mismatches["auto_value_mismatch_rows"] += c[
                "exact_topk_value_mismatch_rows"
            ]
            mismatches["auto_index_set_mismatch_rows"] += c[
                "exact_topk_index_set_mismatch_rows"
            ]
    print("\nAuto versus exact-reference rows:", dict(mismatches))


if __name__ == "__main__":
    main()
