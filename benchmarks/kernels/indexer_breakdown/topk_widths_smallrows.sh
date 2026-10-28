#!/usr/bin/env bash
set -euo pipefail
taskset -pc "${TOPK_CPUSET:-0-15}" $$
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export CUDA_VISIBLE_DEVICES="${TOPK_GPU:-0}"
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/triton
export TORCHINDUCTOR_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/inductor
export XDG_CACHE_HOME=/home/bopang/cg-project/design/idx-breakdown-cache/topk/xdg
export OMP_NUM_THREADS=4
runner=benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py
result=benchmarks/kernels/indexer_breakdown/results
common=(--preset manual --resume --repeat 100 --detail-repeat 100 --trials 3
        --host-repeat 10 --check-rows 128 --sentinel-every 200
        --prune-unused-bounds --bounds actual capture --modes graph --caches warm
        --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32 UP-only --max-seconds 3600)

# Resolve the BF16 cluster/normal transition and dense cooperative boundaries.
.venv/bin/python "$runner" "${common[@]}" \
    --rows 3 5 6 7 10 12 14 20 24 28 40 56 62 \
    --capacities 1048576 524288 262144 --capture-max-seq-len 1048576 \
    --top-k 512 --profiles short long full \
    --output "$result/topk_smallrows_dense.jsonl"

# DeepSeek-V3.2-Exp production width/k from its local checkpoint config.
.venv/bin/python "$runner" "${common[@]}" \
    --rows 1 8 16 32 63 64 65 128 512 4096 --capacities 163840 \
    --capture-max-seq-len 163840 --top-k 2048 --compress-ratio 1 \
    --profiles short mid long full ragged \
    --output "$result/topk_v32_width.jsonl"

# Short-cap V4 deployment: M256K / compression4, rather than the 1M cap.
.venv/bin/python "$runner" "${common[@]}" \
    --rows 1 8 16 32 63 64 65 128 512 4096 --capacities 65536 \
    --capture-max-seq-len 262144 --top-k 512 --compress-ratio 4 \
    --profiles short mid full ragged \
    --output "$result/topk_v4_short_cap.jsonl"

# Compact sparse-consumer logits; legacy arms are only counterfactuals here.
.venv/bin/python "$runner" "${common[@]}" \
    --rows 1 8 32 64 65 128 512 4096 --capacities 16384 \
    --capture-max-seq-len 16384 --top-k 512 --compress-ratio 1 \
    --profiles short full ragged \
    --output "$result/topk_v41_compact.jsonl"
