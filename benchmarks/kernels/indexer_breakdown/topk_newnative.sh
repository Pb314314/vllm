#!/usr/bin/env bash
set -euo pipefail
taskset -pc "${TOPK_CPUSET:-0-15}" $$
cd /home/bopang/cg-project/wt-b4a0fix
export CUDA_VISIBLE_DEVICES="${TOPK_GPU:-0}"
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/newnative/triton
export TORCHINDUCTOR_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/newnative/inductor
export XDG_CACHE_HOME=/home/bopang/cg-project/design/idx-breakdown-cache/topk/newnative/xdg
export CUDA_CACHE_PATH=/home/bopang/cg-project/design/idx-breakdown-cache/topk/newnative/cuda
export OMP_NUM_THREADS=4
runner=/home/bopang/cg-project/wt-indexer-topk-breakdown/benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py
result=/home/bopang/cg-project/wt-indexer-topk-breakdown/benchmarks/kernels/indexer_breakdown/results
for capacity in 1048576 524288 262144; do
    compression=1
    if [[ "$capacity" == 524288 ]]; then compression=2; fi
    if [[ "$capacity" == 262144 ]]; then compression=4; fi
    /home/bopang/cg-project/venv-b4a0fix/bin/python "$runner" \
        --output "$result/topk_newnative_${capacity}.jsonl" \
        --preset manual --resume --rows 1 2 4 6 7 8 16 64 65 85 86 128 256 \
        --capacities "$capacity" --capture-max-seq-len 1048576 \
        --compress-ratio "$compression" --top-k 512 \
        --profiles short long full --bounds actual capture \
        --modes graph --caches warm --repeat 100 --detail-repeat 100 \
        --trials 3 --host-repeat 10 --check-rows 128 --replay-check \
        --sentinel-every 200 --prune-unused-bounds --max-seconds 1800 \
        --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32 UP-only
done
