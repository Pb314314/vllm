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
case "${TOPK_MODEL:-v32}" in
    v32) compression=1; selected_k=2048 ;;
    v4pro) compression=4; selected_k=1024 ;;
    v4) compression=4; selected_k=512 ;;
    *) echo 'Unsupported short-cap model' >&2; exit 2 ;;
esac
runner=benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py
result=benchmarks/kernels/indexer_breakdown/results
capacity="${TOPK_CAPACITY:-65536}"
model_max=$((capacity * compression))
rows=(1 2 3 4 5 6 7 8 10 12 14 16 20 24 28 32 40 48 56 60 62 63 64
      65 66 72 80 96 112 128 160 192 256 320 384 512 640 768 1024 1536
      2048 3072 4096 6144 8192)
reuse=()
if [[ "${TOPK_MODEL:-v32}" == v4 && "$capacity" == 65536 ]]; then
    reuse=(--resume-from "$result/topk_v4_short_cap.jsonl")
fi
deadline="${TOPK_PHASE_DEADLINE:-$(date -u -d '2026-10-28 16:15:00 UTC' +%s)}"
remaining=$((deadline - $(date +%s)))
if (( remaining < 60 )); then echo 'Short-cap deadline reached'; exit 0; fi
.venv/bin/python "$runner" \
    --output "$result/topk_p0_short_${TOPK_MODEL:-v32}_${capacity}.jsonl" \
    --preset manual --resume "${reuse[@]}" --rows "${rows[@]}" \
    --capacities "$capacity" --capture-max-seq-len "$model_max" \
    --compress-ratio "$compression" --top-k "$selected_k" \
    --profiles short mid long full ragged spec --bounds actual capture \
    --modes graph --caches warm --repeat 100 --detail-repeat 100 --trials 3 \
    --host-repeat 10 --check-rows 128 --sentinel-every 200 \
    --prune-unused-bounds --max-seconds "$remaining" \
    --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32 UP-only
