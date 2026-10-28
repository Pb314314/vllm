#!/usr/bin/env bash
set -euo pipefail
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export TOPK_GPU=0
export TOPK_CPUSET=0-15
wait_pid="${TOPK_WAIT_PID:-915369}"
while kill -0 "$wait_pid" 2>/dev/null; do sleep 5; done
date -u
bash benchmarks/kernels/indexer_breakdown/topk_newnative.sh
bash benchmarks/kernels/indexer_breakdown/topk_p0_k512_tail.sh
TOPK_MODEL=v4 bash benchmarks/kernels/indexer_breakdown/topk_p0_shortcap.sh
