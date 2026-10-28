#!/usr/bin/env bash
set -euo pipefail
cd /home/bopang/cg-project/wt-b4a0fix
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/pipeline/newnative/triton
export TORCHINDUCTOR_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/pipeline/newnative/inductor
pipeline_python=/home/bopang/cg-project/venv-b4a0fix/bin/python
pipeline_script=/home/bopang/cg-project/wt-indexer-topk-breakdown/benchmarks/kernels/indexer_breakdown/pipeline_bench.py
pipeline_results=/home/bopang/cg-project/wt-indexer-topk-breakdown/benchmarks/kernels/indexer_breakdown/results

"$pipeline_python" "$pipeline_script" \
    --out "$pipeline_results/pipeline_newnative_core.jsonl" \
    --models v4,v41-c2,v41-c1 --dtypes mxfp4,fp8 \
    --rows 64,256,1024,4096,8192 --contexts 1536,8192,32768 \
    --variants single-auto,single-ds,final-512 --production-helper \
    --repeat 100 --trials 3 --cache warm,cold \
    --eager --eager-reps 10 --eager-warmup 10 --resume

"$pipeline_python" "$pipeline_script" \
    --out "$pipeline_results/pipeline_newnative_candidate.jsonl" \
    --models v41-c1 --dtypes mxfp4,fp8 --rows 1024,4096 \
    --contexts 8192,32768 --candidate select \
    --variants single-auto,single-ds,final-512 --production-helper \
    --repeat 100 --trials 3 --cache warm,cold \
    --eager --eager-reps 10 --eager-warmup 10 --stages --resume

"$pipeline_python" "$pipeline_script" \
    --out "$pipeline_results/pipeline_newnative_actual.jsonl" \
    --models v41-c1 --dtypes mxfp4,fp8 --rows 4096 \
    --contexts 1536,8192,32768 --max-seq-mode actual \
    --variants single-auto,single-ds,final-512 --production-helper \
    --repeat 100 --trials 3 --cache warm,cold \
    --eager --eager-reps 10 --eager-warmup 10 --resume
