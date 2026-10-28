#!/usr/bin/env bash
set -euo pipefail
taskset -pc "${TOPK_CPUSET:-104-119}" $$
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export CUDA_VISIBLE_DEVICES="${TOPK_GPU:-3}"
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/triton
export TORCHINDUCTOR_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/inductor
export XDG_CACHE_HOME=/home/bopang/cg-project/design/idx-breakdown-cache/topk/xdg
export CUDA_CACHE_PATH=/home/bopang/cg-project/design/idx-breakdown-cache/topk/cuda
export OMP_NUM_THREADS=4
runner=benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py
result=benchmarks/kernels/indexer_breakdown/results
sample=/home/bopang/cg-project/results/indexer_breakdown_memory_20261028/models/real_logits/logits_644513_2.pt
# Controlled capacity layouts using identical V4 learned scores. These are
# capacity counterfactuals, not actual V4.1 checkpoint activations.
for capacity in 65536 262144 524288 1048576; do
  capture_bound=1048576
  if [[ "$capacity" == 65536 ]]; then capture_bound=262144; fi
  for sample_kind in checkpoint randn_control; do
    control=()
    if [[ "$sample_kind" == randn_control ]]; then
        control=(--file-random-control)
    fi
    .venv/bin/python "$runner" --preset manual --resume \
        --rows 1 8 64 65 512 4096 --capacities "$capacity" \
        --capture-max-seq-len "$capture_bound" --raw-max-seq-len 10261 \
        --compress-ratio 4 --top-k 512 --profiles file --logits-file "$sample" \
        "${control[@]}" --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32 UP-only \
        --prune-unused-bounds --bounds actual raw_actual capture \
        --modes graph --caches warm --repeat 100 --detail-repeat 100 \
        --trials 3 --host-repeat 10 --check-rows 128 --replay-check \
        --sentinel-every 200 --max-seconds 1800 \
        --output "$result/topk_real_capacity_${capacity}_${sample_kind}.jsonl"
  done
done
