#!/usr/bin/env bash
set -euo pipefail
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/pipeline/triton
export TORCHINDUCTOR_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/pipeline/inductor

.venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_math_check.py \
    --out benchmarks/kernels/indexer_breakdown/results/pipeline_math_reference.jsonl

.venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_bench.py \
    --out benchmarks/kernels/indexer_breakdown/results/pipeline_eager_recheck.jsonl \
    --models v4,v41-c2,v41-c1 --dtypes mxfp4,fp8 --rows 2048,4096,8192 \
    --contexts 1536,8192,32768 \
    --variants single-auto,single-ds,final-512,ds-512,prebuilt-512,ds-2048 \
    --repeat 100 --trials 3 --cache warm,cold --eager --eager-reps 30 \
    --no-profile --resume

.venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_bench.py \
    --out benchmarks/kernels/indexer_breakdown/results/pipeline_recheck.jsonl \
    --models v4,v41-c2,v41-c1 --dtypes mxfp4,fp8 --rows 64,512 \
    --contexts 8192,32768 \
    --variants single-auto,single-ds,prototype-1024,ds-1024,final-512 \
    --repeat 100 --trials 3 --cache warm,cold --eager --eager-reps 10 \
    --stages --resume

for candidate in select mask; do
    .venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_bench.py \
        --out "benchmarks/kernels/indexer_breakdown/results/pipeline_candidate_${candidate}.jsonl" \
        --models v41-c1 --dtypes mxfp4,fp8 --rows 64,256,1024,4096 \
        --contexts 8192,32768 --candidate "$candidate" \
        --variants single-auto,single-ds,final-512,ds-1024 \
        --repeat 100 --trials 3 --cache warm,cold \
        --eager --eager-reps 10 --stages --resume
done

for q_len in 1 2 3 4 6; do
    .venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_bench.py \
        --out "benchmarks/kernels/indexer_breakdown/results/pipeline_q${q_len}.jsonl" \
        --models v4,v41-c1 --dtypes mxfp4,fp8 --q-len "$q_len" \
        --rows 64,256,1024,4096 --contexts 8192,32768 \
        --variants single-auto,single-ds,final-512 \
        --repeat 100 --trials 3 --cache warm,cold --eager --eager-reps 10 \
        --stages --resume
done

.venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_bench.py \
    --out benchmarks/kernels/indexer_breakdown/results/pipeline_actual_bound.jsonl \
    --models v4,v41-c2,v41-c1 --dtypes mxfp4,fp8 \
    --rows 8,64,65,256,1024,4096 --contexts 1536,8192,32768 \
    --variants single-auto,single-ds,final-512 --max-seq-mode actual \
    --repeat 100 --trials 3 --cache warm,cold --eager --eager-reps 10 \
    --stages --resume

.venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_bench.py \
    --out benchmarks/kernels/indexer_breakdown/results/pipeline_fp32_budgets.jsonl \
    --models v4,v41-c2,v41-c1 --dtypes mxfp4,fp8 --rows 1024,4096,8192 \
    --contexts 1536,8192,32768 \
    --variants auto-256,auto-1024,auto-2048,persistent-256,persistent-512,persistent-1024,persistent-2048 \
    --repeat 100 --trials 3 --cache warm,cold --eager --eager-reps 10 \
    --stages --resume

.venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_bench.py \
    --out benchmarks/kernels/indexer_breakdown/results/pipeline_long.jsonl \
    --models v4,v41-c2,v41-c1 --dtypes mxfp4,fp8 --rows 64,256,1024,4096 \
    --contexts 131072 \
    --variants single-auto,single-ds,final-512,ds-1024,ds-2048,prebuilt-512 \
    --repeat 100 --trials 3 --cache warm,cold --eager --eager-reps 10 \
    --stages --resume

.venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_bench.py \
    --out benchmarks/kernels/indexer_breakdown/results/pipeline_full_context.jsonl \
    --models v4,v41-c2,v41-c1 --dtypes mxfp4,fp8 --rows 64,256,1024 \
    --contexts 1048576 \
    --variants single-auto,single-ds,final-512,ds-1024,ds-2048,prebuilt-512 \
    --repeat 100 --trials 3 --cache warm,cold --eager --eager-reps 10 \
    --stages --resume
