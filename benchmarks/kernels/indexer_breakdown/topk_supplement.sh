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
        --prune-unused-bounds --bounds actual capture --max-seconds 5400
        --skip-cases "$result/topk_unsafe_cases.jsonl")

# Small-row cooperative/persistent crossover at the other supported k values.
.venv/bin/python "$runner" "${common[@]}" \
    --rows 1 8 32 63 64 65 128 512 4096 --capacities 1048576 262144 \
    --capture-max-seq-len 1048576 --top-k 1024 2048 \
    --profiles short long ragged --modes graph --caches warm \
    --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32 UP-only \
    --output "$result/topk_other_k.jsonl"

# Both CPU launch and cold-L2 effects, including large rows in eager mode.
for capacity in 1048576 262144; do
    compression=1
    if [[ "$capacity" == 262144 ]]; then compression=4; fi
    .venv/bin/python "$runner" "${common[@]}" \
    --rows 1 16 64 256 1024 4096 --capacities "$capacity" \
    --compress-ratio "$compression" --bounds actual raw_actual capture \
    --capture-max-seq-len 1048576 --top-k 512 --profiles short long \
    --modes eager graph --caches warm cold \
    --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32 UP-only \
    --output "$result/topk_modes_cache_${capacity}.jsonl"
done

# Available alternatives, matching the expensive-arm pruning in the plan.
.venv/bin/python "$runner" "${common[@]}" \
    --rows 1 8 64 512 4096 --capacities 1048576 262144 \
    --capture-max-seq-len 1048576 --top-k 512 --profiles short long \
    --modes graph --caches warm \
    --schemes DS-bf16 UP+COOP UP+PERS UP+FI UP+PERROW \
    --output "$result/topk_alternatives.jsonl"

# Real neighboring graph rows, plus V4.1 large-row controls omitted above.
for capacity in 1048576 524288; do
    compression=1
    if [[ "$capacity" == 524288 ]]; then compression=2; fi
    .venv/bin/python "$runner" "${common[@]}" \
    --rows 84 88 512 1024 2048 4096 8192 \
    --capacities "$capacity" --compress-ratio "$compression" \
    --capture-max-seq-len 1048576 \
    --top-k 512 --profiles short long ragged --modes graph --caches warm \
    --schemes DS-bf16 UP+PERS UP+DS-fp32 UP-only \
    --output "$result/topk_graph_neighbors_large_${capacity}.jsonl"
done

# Bounds/sentinel behavior with empty/short rows and extensive ties.
.venv/bin/python "$runner" "${common[@]}" \
    --rows 1 8 64 65 512 --capacities 262144 --capture-max-seq-len 1048576 \
    --top-k 512 1024 2048 --profiles padding spec \
    --distributions normal ties --compress-ratio 4 --query-len 8 \
    --replay-check \
    --modes graph --caches warm \
    --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32 UP+FI UP+PERROW \
    --output "$result/topk_correctness_stress.jsonl"
