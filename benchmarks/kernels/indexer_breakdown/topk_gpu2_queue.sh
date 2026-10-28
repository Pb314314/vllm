#!/usr/bin/env bash
set -euo pipefail
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export TOPK_GPU=2
export TOPK_CPUSET=70-85
export TOPK_CAPACITIES=262144
wait_pid="${TOPK_WAIT_PID:-930205}"
while kill -0 "$wait_pid" 2>/dev/null; do sleep 5; done
date -u
exec bash benchmarks/kernels/indexer_breakdown/topk_p0_k512_tail.sh
