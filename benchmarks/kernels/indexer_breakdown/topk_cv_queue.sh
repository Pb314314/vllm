#!/usr/bin/env bash
set -euo pipefail
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
wait_pid="${TOPK_WAIT_PID:?Set the preceding same-GPU queue shell PID}"
printf 'CV queue waits for pid %s on GPU %s\n' "$wait_pid" "${TOPK_GPU:?Set GPU}"
while kill -0 "$wait_pid" 2>/dev/null; do sleep 5; done
date -u
export TOPK_CV_DEADLINE="${TOPK_CV_DEADLINE:-$(date -u -d '2026-10-28 16:45:00 UTC' +%s)}"
exec bash benchmarks/kernels/indexer_breakdown/topk_cv_rerun.sh
