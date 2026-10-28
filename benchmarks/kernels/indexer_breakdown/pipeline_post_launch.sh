#!/usr/bin/env bash
set -euo pipefail
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/pipeline/triton
export TORCHINDUCTOR_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/pipeline/inductor

.venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_bench.py \
    --out benchmarks/kernels/indexer_breakdown/results/pipeline_actual_bound_8192.jsonl \
    --models v4,v41-c2,v41-c1 --dtypes mxfp4,fp8 --rows 8192 \
    --contexts 1536,8192,32768 --max-seq-mode actual \
    --variants single-auto,single-ds,final-512 \
    --repeat 100 --trials 3 --cache warm,cold --eager --eager-reps 30 --eager-warmup 10 \
    --stages --resume

.venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_bench.py \
    --out benchmarks/kernels/indexer_breakdown/results/pipeline_eager_quiet_repeat.jsonl \
    --models v41-c2 --dtypes mxfp4,fp8 --rows 2048,4096,8192 \
    --contexts 1536,32768 \
    --variants single-auto,single-ds,final-512,ds-512 \
    --repeat 100 --trials 3 --cache warm,cold --eager --eager-reps 30 --eager-warmup 10 \
    --no-profile --resume

.venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_bench.py \
    --out benchmarks/kernels/indexer_breakdown/results/pipeline_noisy_repeat.jsonl \
    --models v41-c1 --dtypes fp8 --rows 4096 --contexts 32768 \
    --variants single-auto,single-ds,final-512 \
    --repeat 300 --trials 5 --cache warm,cold --eager --eager-reps 30 --eager-warmup 10 \
    --no-profile --resume

.venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_interleaved.py \
    --out benchmarks/kernels/indexer_breakdown/results/pipeline_interleaved.jsonl
