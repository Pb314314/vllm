#!/usr/bin/env bash
set -euo pipefail
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
for wait_pid in "${TOPK_WAIT_PID:?Set same-GPU CV queue PID}" "${TOPK_OTHER_WAIT_PID:-1030049}"; do
    while kill -0 "$wait_pid" 2>/dev/null; do sleep 5; done
done
date -u
exec bash benchmarks/kernels/indexer_breakdown/topk_newnative_models.sh
