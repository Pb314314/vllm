#!/usr/bin/env bash
set -euo pipefail
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/triton
export TORCHINDUCTOR_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/inductor
export XDG_CACHE_HOME=/home/bopang/cg-project/design/idx-breakdown-cache/topk/xdg
export OMP_NUM_THREADS=4
runner=benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py
result=benchmarks/kernels/indexer_breakdown/results
common=(--preset manual --resume --repeat 100 --detail-repeat 100 --trials 3
        --host-repeat 10 --check-rows 128 --sentinel-every 200
        --prune-unused-bounds --bounds actual capture --modes graph
        --caches warm --top-k 512 --max-seconds 10800
        --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32
                  COOP-fp32-only PERS-fp32-only UP-only)
rows=(1 2 4 8 16 32 48 60 63 64 65 66 72 80 85 86 96 128 160 170 171 192 256)
if [[ "${TOPK_RERUN_UNSTABLE:-0}" == 1 ]]; then
    taskset -pc 0-15 $$
    common+=(--rerun-unstable)
fi
.venv/bin/python "$runner" "${common[@]}" --rows "${rows[@]}" \
    --capacities 1048576 --capture-max-seq-len 1048576 --compress-ratio 1 \
    --profiles short mid long full ragged --output "$result/topk_v41c1_core.jsonl"
.venv/bin/python "$runner" "${common[@]}" --rows "${rows[@]}" \
    --capacities 524288 --capture-max-seq-len 1048576 --compress-ratio 2 \
    --profiles short mid long full ragged --output "$result/topk_v41c2_core.jsonl"
.venv/bin/python "$runner" "${common[@]}" \
    --rows 1 8 16 32 48 60 63 64 65 66 80 96 128 256 512 1024 2048 4096 8192 \
    --capacities 262144 --capture-max-seq-len 1048576 --compress-ratio 4 \
    --profiles short mid long full ragged --output "$result/topk_v4_core.jsonl"
