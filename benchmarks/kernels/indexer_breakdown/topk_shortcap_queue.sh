#!/usr/bin/env bash
set -euo pipefail
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
wait_pid="${TOPK_WAIT_PID:?Set the preceding same-GPU shell PID}"
while kill -0 "$wait_pid" 2>/dev/null; do sleep 5; done
date -u
exec bash benchmarks/kernels/indexer_breakdown/topk_p0_shortcap.sh
