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
samples=/home/bopang/cg-project/design/idx-breakdown-cache/pipeline/score_samples
for model in v4 v41-c2 v41-c1; do
    case "$model" in
        v4) capacity=262144; compression=4 ;;
        v41-c2) capacity=524288; compression=2 ;;
        v41-c1) capacity=1048576; compression=1 ;;
    esac
    for quant in mxfp4 fp8; do
      for sample_kind in saved randn_control; do
        if [[ "$quant" == fp8 && "$sample_kind" == randn_control ]]; then
            continue
        fi
        control=()
        if [[ "$sample_kind" == randn_control ]]; then
            control=(--file-random-control)
        fi
        .venv/bin/python "$runner" --preset manual --resume \
            --rows 1 8 32 64 65 128 512 4096 --capacities "$capacity" \
            --capture-max-seq-len 1048576 --compress-ratio "$compression" \
            --top-k 512 --profiles file \
            --logits-file "$samples/synthetic_deepgemm_${model}_${quant}.pt" \
            "${control[@]}" \
            --schemes DS-bf16 UP+COOP UP+PERS UP+DS-fp32 UP+FI UP+PERROW UP-only \
            --prune-unused-bounds --bounds actual raw_actual capture \
            --modes graph --caches warm \
            --repeat 100 --detail-repeat 100 --trials 3 --host-repeat 10 \
            --check-rows 128 --sentinel-every 200 --max-seconds 1800 \
            --replay-check \
            --skip-cases "$result/topk_unsafe_cases.jsonl" \
            --output "$result/topk_samples_${model}_${quant}_${sample_kind}.jsonl"
      done
    done
done
