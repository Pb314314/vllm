#!/usr/bin/env bash
set -euo pipefail
taskset -pc "${TOPK_CPUSET:-16-31}" $$
cd /home/bopang/cg-project/wt-b4a0fix
export CUDA_VISIBLE_DEVICES="${TOPK_GPU:-1}"
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/newnative/triton
export TORCHINDUCTOR_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/newnative/inductor
export XDG_CACHE_HOME=/home/bopang/cg-project/design/idx-breakdown-cache/topk/newnative/xdg
export CUDA_CACHE_PATH=/home/bopang/cg-project/design/idx-breakdown-cache/topk/newnative/cuda
export OMP_NUM_THREADS=4
case "${TOPK_MODEL:-v32}" in
    v32) capacity=163840; model_max=163840; compression=1; selected_k=2048 ;;
    v4pro) capacity=262144; model_max=1048576; compression=4; selected_k=1024 ;;
    *) echo 'Unsupported PR-native model anchor' >&2; exit 2 ;;
esac
deadline="${TOPK_NATIVE_DEADLINE:-$(date -u -d '2026-10-28 16:40:00 UTC' +%s)}"
remaining=$((deadline - $(date +%s)))
if (( remaining < 180 )); then echo 'Optional native-anchor deadline reached'; exit 0; fi
runner=/home/bopang/cg-project/wt-indexer-topk-breakdown/benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py
result=/home/bopang/cg-project/wt-indexer-topk-breakdown/benchmarks/kernels/indexer_breakdown/results
common=(--output "$result/topk_newnative_${TOPK_MODEL:-v32}.jsonl"
        --preset manual --resume --rows 1 8 16 64 65 256
        --capacities "$capacity" --capture-max-seq-len "$model_max"
        --compress-ratio "$compression" --top-k "$selected_k"
        --profiles short long full --bounds actual capture
        --modes graph --caches warm --repeat 100 --detail-repeat 100
        --trials 3 --host-repeat 10 --check-rows 128 --replay-check
        --sentinel-every 200 --prune-unused-bounds --max-seconds "$remaining"
        --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32 UP-only)
/home/bopang/cg-project/venv-b4a0fix/bin/python "$runner" "${common[@]}"
/home/bopang/cg-project/venv-b4a0fix/bin/python "$runner" "${common[@]}" --rerun-unstable
