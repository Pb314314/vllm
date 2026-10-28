#!/usr/bin/env bash
set -euo pipefail
repo_root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$repo_root"
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("benchmarks/kernels/indexer_breakdown/results")
pairs = [json.loads(line) for line in (root / "topk_summary_comparisons.jsonl").open()]
groups = [json.loads(line) for line in (root / "topk_summary_all.jsonl").open()]
fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
used = []
colors = {"short": "#3178b8", "long": "#de7c22", "full": "#9b52ad"}
for column, (family, capacity) in enumerate(
    [("v4", 262144), ("v41c2", 524288), ("v41c1", 1048576)]
):
    source_name = f"topk_{family}_core.jsonl"
    ax = axes[0, column]
    for profile, color in colors.items():
        data = sorted(
            [
                p
                for p in pairs
                if Path(p["source"]["path"]).name == source_name
                and p["alternative"] == "DS-bf16"
                and p["length_profile"] == profile
                and p["baseline_bound_kind"]
                == ("actual" if p["rows"] <= 64 else "capture")
                and p["both_selection_valid"]
                and p["both_three_trials"]
            ],
            key=lambda p: p["rows"],
        )
        used.extend(data)
        ax.plot(
            [p["rows"] for p in data],
            [p["alternative_gpu_span_us"] / p["baseline_gpu_span_us"] for p in data],
            color=color,
            marker="o",
            markersize=3,
            linewidth=1,
            label={
                "short": "4K effective",
                "long": "128K effective",
                "full": "Full capacity",
            }[profile],
        )
        noisy = [p for p in data if p["either_cv_gt_3_percent"]]
        ax.scatter(
            [p["rows"] for p in noisy],
            [p["alternative_gpu_span_us"] / p["baseline_gpu_span_us"] for p in noisy],
            marker="x",
            s=35,
            color="black",
            zorder=5,
        )
    ax.axhline(1, color="#555555", linewidth=1, linestyle="--")
    ax.axvline(64, color="#999999", linewidth=0.8, linestyle=":")
    ax.set(
        xscale="log",
        ylim=(0, 2.4),
        title=f"W = {capacity:,}; k = 512",
        xlabel="Query rows R",
        ylabel="BF16 DS latency / current auto",
    )
    ax.grid(alpha=0.18)
    if column == 0:
        ax.legend(fontsize=8)
    memory_ax = axes[1, column]
    for scheme, color in [("auto", "#3178b8"), ("DS-bf16", "#29956d")]:
        data = sorted(
            [
                g
                for g in groups
                if Path(g["source"]["path"]).name == source_name
                and g["length_profile"] == "short"
                and g["scheme"]
                == (
                    ("UP+COOP" if g["rows"] <= 64 else "UP+PERS")
                    if scheme == "auto"
                    else scheme
                )
                and g["bound_kind"]
                == ("actual" if scheme != "auto" or g["rows"] <= 64 else "capture")
                and g["valid_selection"]
                and g["complete_three_trials"]
            ],
            key=lambda g: g["rows"],
        )
        memory_ax.plot(
            [g["rows"] for g in data],
            [g["additional_peak_allocated_bytes_median"] / 2**20 for g in data],
            color=color,
            marker="o",
            markersize=3,
            linewidth=1,
            label="Current auto" if scheme == "auto" else "Direct BF16 DS",
        )
    memory_ax.set(
        xscale="log",
        yscale="symlog",
        ylim=(-0.2, None),
        xlabel="Query rows R",
        ylabel="Additional allocated peak (MiB)",
    )
    memory_ax.grid(alpha=0.18)
    if column == 0:
        memory_ax.legend(fontsize=8)
fig.suptitle(
    "Top-k stage only: GPU graph replay, GB200, historical cff native\n"
    "100 invocations x 3 trials; repeated inputs; unlocked clocks",
    fontsize=13,
)
fig.text(
    0.5,
    -0.025,
    "Latency < 1 favors DS. Black x: trial CV > 3%. Lines only guide the eye.\n"
    "Memory excludes BF16 input, final output and warmed shared radix "
    "workspace; not model graph-pool reserved.",
    ha="center",
    fontsize=9,
)
artifacts = []
for extension in ("png", "svg"):
    path = root / f"topk_core_tradeoffs.{extension}"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    artifacts.append(
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
manifest = {
    "status": "plot_manifest",
    "artifacts": artifacts,
    "source_summary_sha256": hashlib.sha256(
        (root / "topk_summary_comparisons.jsonl").read_bytes()
    ).hexdigest(),
    "source_pairs": used,
}
(root / "topk_plot_manifest.jsonl").write_text(json.dumps(manifest) + "\n")
print(json.dumps(artifacts, indent=2))
PY
