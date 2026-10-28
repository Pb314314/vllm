#!/usr/bin/env bash
set -euo pipefail
taskset -pc "${TOPK_CPUSET:-70-85}" $$
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export CUDA_VISIBLE_DEVICES="${TOPK_GPU:-2}"
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/triton
export TORCHINDUCTOR_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/inductor
export XDG_CACHE_HOME=/home/bopang/cg-project/design/idx-breakdown-cache/topk/xdg
export CUDA_CACHE_PATH=/home/bopang/cg-project/design/idx-breakdown-cache/topk/cuda
export OMP_NUM_THREADS=4
.venv/bin/python benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py \
    --output benchmarks/kernels/indexer_breakdown/results/topk_modes_cache_1048576_ceiling.jsonl \
    --preset manual --resume --rows 1 16 64 256 1024 4096 \
    --capacities 1048576 --capture-max-seq-len 1048576 --compress-ratio 1 \
    --top-k 512 --profiles short long --bounds actual raw_actual capture \
    --modes eager graph --caches cold --repeat 100 --detail-repeat 100 \
    --trials 3 --host-repeat 10 --check-rows 128 --sentinel-every 200 \
    --prune-unused-bounds --max-seconds 1800 \
    --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32 UP-only
