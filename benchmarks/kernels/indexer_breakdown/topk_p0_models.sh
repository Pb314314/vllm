#!/usr/bin/env bash
set -euo pipefail
taskset -pc "${TOPK_CPUSET:-16-31}" $$
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export CUDA_VISIBLE_DEVICES="${TOPK_GPU:-1}"
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/triton
export TORCHINDUCTOR_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/inductor
export XDG_CACHE_HOME=/home/bopang/cg-project/design/idx-breakdown-cache/topk/xdg
export CUDA_CACHE_PATH=/home/bopang/cg-project/design/idx-breakdown-cache/topk/cuda
export OMP_NUM_THREADS=4
model="${TOPK_MODEL:-v32}"
case "$model" in
    v32) capacity=163840; model_max=163840; compression=1; selected_k=2048 ;;
    v4pro) capacity=262144; model_max=1048576; compression=4; selected_k=1024 ;;
    *) echo "Unsupported model shape: $model" >&2; exit 2 ;;
esac
runner=benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py
result=benchmarks/kernels/indexer_breakdown/results
rows=(1 2 3 4 5 6 7 8 10 12 14 16 20 24 28 32 40 48 56 60 62 63 64
      65 66 72 80 96 112 128 160 192 256 320 384 512 640 768 1024 1536
      2048 3072 4096 6144 8192)
common=(--preset manual --resume --rows "${rows[@]}" --capacities "$capacity"
        --capture-max-seq-len "$model_max" --compress-ratio "$compression"
        --top-k "$selected_k" --modes graph --caches warm --repeat 100
        --detail-repeat 100 --trials 3 --host-repeat 10 --check-rows 128
        --sentinel-every 200 --prune-unused-bounds --max-seconds 7200
        --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32
                  COOP-fp32-only PERS-fp32-only UP-only)
.venv/bin/python "$runner" "${common[@]}" \
    --profiles short mid long full ragged --bounds actual raw_actual capture \
    --output "$result/topk_p0_${model}_regular.jsonl"
# Spec profiles preserve query-group structure. The compressed actual bound is
# a named counterfactual; FULL uses the uncompressed configured host bound.
.venv/bin/python "$runner" "${common[@]}" \
    --profiles spec --bounds actual capture \
    --output "$result/topk_p0_${model}_spec.jsonl"
