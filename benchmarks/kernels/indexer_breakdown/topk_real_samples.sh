#!/usr/bin/env bash
set -euo pipefail
taskset -pc "${TOPK_CPUSET:-70-85}" $$
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export CUDA_VISIBLE_DEVICES="${TOPK_GPU:-2}"
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/triton
export TORCHINDUCTOR_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/inductor
export XDG_CACHE_HOME=/home/bopang/cg-project/design/idx-breakdown-cache/topk/xdg
export OMP_NUM_THREADS=4
runner=benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py
result=benchmarks/kernels/indexer_breakdown/results
samples=/home/bopang/cg-project/results/indexer_breakdown_memory_20261028/models/real_logits
.venv/bin/python "$runner" --preset manual --resume \
    --rows 1 8 64 65 --capacities 1048576 --capture-max-seq-len 1048576 \
    --compress-ratio 1 --profiles short long --top-k 512 \
    --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32 UP-only \
    --prune-unused-bounds --bounds actual capture --modes graph --caches warm \
    --repeat 100 --detail-repeat 100 --trials 3 --host-repeat 10 \
    --sentinel-every 200 --max-seconds 600 \
    --output "$result/topk_anchor_gpu2.jsonl"
for layer_sample in 1 2 3; do
  for sample_kind in checkpoint randn_control; do
    control=()
    if [[ "$sample_kind" == randn_control ]]; then
        control=(--file-random-control)
    fi
    .venv/bin/python "$runner" --preset manual --resume \
        --rows 1 8 32 64 65 128 512 4096 --capacities 65536 \
        --capture-max-seq-len 262144 --raw-max-seq-len 10261 \
        --compress-ratio 4 --stride-padding 512 --top-k 512 --profiles file \
        --logits-file "$samples/logits_644513_${layer_sample}.pt" \
        "${control[@]}" \
        --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32 UP+FI UP+PERROW \
                  COOP-fp32-only PERS-fp32-only UP-only \
        --prune-unused-bounds --bounds actual raw_actual capture \
        --modes graph eager --caches warm --repeat 100 --detail-repeat 100 \
        --trials 3 --host-repeat 10 --check-rows 128 --replay-check \
        --sentinel-every 200 --max-seconds 2400 \
        --output "$result/topk_real_layer${layer_sample}_${sample_kind}.jsonl"
  done
done
