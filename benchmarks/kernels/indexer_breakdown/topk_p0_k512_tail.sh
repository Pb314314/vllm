#!/usr/bin/env bash
set -euo pipefail
taskset -pc "${TOPK_CPUSET:-0-15}" $$
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export CUDA_VISIBLE_DEVICES="${TOPK_GPU:-0}"
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/triton
export TORCHINDUCTOR_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/inductor
export XDG_CACHE_HOME=/home/bopang/cg-project/design/idx-breakdown-cache/topk/xdg
export CUDA_CACHE_PATH=/home/bopang/cg-project/design/idx-breakdown-cache/topk/cuda
export OMP_NUM_THREADS=4
if [[ "${TOPK_GPU:-0}" == 2 ]]; then
    bash benchmarks/kernels/indexer_breakdown/topk_cold_rounding_fix.sh
fi
runner=benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py
result=benchmarks/kernels/indexer_breakdown/results
rows=(1 2 3 4 5 6 7 8 10 12 14 16 20 24 28 32 40 48 56 60 62 63 64
      65 66 72 80 96 112 128 160 192 256 320 384 512 640 768 1024 1536
      2048 3072 4096 6144 8192)
reuse=("$result/topk_v41c1_core.jsonl" "$result/topk_v41c2_core.jsonl"
       "$result/topk_v4_core.jsonl" "$result/topk_smallrows_dense.jsonl")
for source in "$result/topk_modes_cache_262144.jsonl" "$result/topk_alternatives.jsonl"; do
    if [[ -f "$source" ]]; then reuse+=("$source"); fi
done
deadline="${TOPK_PHASE_DEADLINE:-$(date -u -d '2026-10-28 16:15:00 UTC' +%s)}"
read -r -a capacities <<< "${TOPK_CAPACITIES:-1048576 524288}"
for capacity in "${capacities[@]}"; do
    compression=1
    if [[ "$capacity" == 524288 ]]; then compression=2; fi
    if [[ "$capacity" == 262144 ]]; then compression=4; fi
    remaining=$((deadline - $(date +%s)))
    if (( remaining < 60 )); then echo "P0 tail deadline reached"; break; fi
    .venv/bin/python "$runner" \
        --output "$result/topk_p0_k512_${capacity}_tail.jsonl" \
        --preset manual --resume --resume-from "${reuse[@]}" \
        --rows "${rows[@]}" --capacities "$capacity" \
        --capture-max-seq-len 1048576 --compress-ratio "$compression" \
        --top-k 512 --profiles short mid long full ragged spec \
        --bounds actual capture --modes graph --caches warm \
        --repeat 100 --detail-repeat 100 --trials 3 --host-repeat 10 \
        --check-rows 128 --sentinel-every 200 --prune-unused-bounds \
        --max-seconds "$remaining" \
        --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32 \
                  COOP-fp32-only PERS-fp32-only UP-only
done
