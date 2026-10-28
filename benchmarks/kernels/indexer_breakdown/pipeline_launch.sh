#!/usr/bin/env bash
set -euo pipefail
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/pipeline/triton
export TORCHINDUCTOR_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/pipeline/inductor
for model in v41-c1 v41-c2 v4; do
    for dtype in mxfp4 fp8; do
        .venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_bench.py \
            --out "benchmarks/kernels/indexer_breakdown/results/pipeline_core_${model}_${dtype}.jsonl" \
            --models "$model" --dtypes "$dtype" \
            --rows 8,64,65,256,512,1024,2048,4096,8192 \
            --contexts 1536,8192,32768 --repeat 100 --trials 3 \
            --cache warm,cold --eager --eager-reps 10 --stages --resume
    done
done
