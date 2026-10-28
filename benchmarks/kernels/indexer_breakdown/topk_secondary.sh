#!/usr/bin/env bash
set -euo pipefail
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export TOPK_GPU=2
export TOPK_CPUSET=70-85
/bin/bash benchmarks/kernels/indexer_breakdown/topk_real_samples.sh
/bin/bash benchmarks/kernels/indexer_breakdown/topk_samples.sh
/bin/bash benchmarks/kernels/indexer_breakdown/topk_supplement.sh
