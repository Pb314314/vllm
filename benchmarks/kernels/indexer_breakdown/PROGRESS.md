# Indexer/top-k breakdown progress

## Status

Execution started 2026-10-28 UTC. Baseline and measurement corrections are
recorded in PLAN.md. No product behavior changes are permitted in this branch.

## Phase 0: setup

- Plan commit: d84086a863d2d4c3aef82e3292627b3f178ff8cb.
- Historical product baseline: cff6224872ad24b10c224bc4186778dcd90a1b21.
- Final chunk planner reference: 4e5b4f8ce9b067d92e7373d950727d8238e9a289.
- Shared worktree: /home/bopang/cg-project/wt-indexer-topk-breakdown.
- Kernel node reservation: Slurm 40469 (8h); memory/model node: 40470 (6h).
  Both queued at startup; idle development GPUs are used only for setup/probes.
- Three independent workers own top-k, pipeline/chunking, and model-memory
  measurements. The coordinator owns integration, validation, and publishing.
- Resumable JSONL data and per-phase reports will be saved incrementally.

## Pending

- Environment import and CUPTI probe.
- Top-k boundary sweep and full-pipeline/chunk ablations.
- Actual model PW/FULL allocation snapshots and shape inventory.
- Analysis and limited model-level confirmation, subject to the timebox.
