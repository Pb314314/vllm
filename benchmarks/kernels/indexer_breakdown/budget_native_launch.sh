#!/usr/bin/env bash
set -euo pipefail
study_group=${1:?core, long, candidate or mask}
cd /home/bopang/cg-project/wt-b4a0fix
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/budget-v2/triton
study_python=/home/bopang/cg-project/venv-b4a0fix/bin/python
study_script=/home/bopang/cg-project/wt-indexer-topk-breakdown/benchmarks/kernels/indexer_breakdown/pipeline_bench.py
study_results=/home/bopang/cg-project/wt-indexer-topk-breakdown/benchmarks/kernels/indexer_breakdown/results
study_common=(--dtypes "mxfp4,fp8" --production-helper --strict-cold-l2
              --variants "single-auto,single-ds,ds-512,ds-1024,ds-2048,final-512,final-1024,final-2048"
              --repeat 100 --trials 3 --cache "warm,cold"
              --eager --eager-reps 10 --eager-warmup 10 --resume)
case "$study_group" in
    core)
        "$study_python" "$study_script" "${study_common[@]}" \
            --out "$study_results/pipeline_budget_v2_core.jsonl" \
            --models v4,v41-c2,v41-c1 \
            --rows 64,128,256,512,1024,2048,4096,8192 \
            --contexts 1536,8192,32768
        ;;
    long)
        "$study_python" "$study_script" "${study_common[@]}" \
            --out "$study_results/pipeline_budget_v2_long.jsonl" \
            --models v4,v41-c2,v41-c1 --rows 256,1024,4096 --contexts 131072
        "$study_python" "$study_script" "${study_common[@]}" \
            --out "$study_results/pipeline_budget_v2_full.jsonl" \
            --models v4,v41-c2,v41-c1 --rows 64,256,1024 --contexts 1048576
        ;;
    candidate)
        "$study_python" "$study_script" "${study_common[@]}" \
            --out "$study_results/pipeline_budget_v2_candidate.jsonl" \
            --models v41-c1 --rows 128,256,512,1024,4096,8192 \
            --contexts 8192,32768 --candidate select
        ;;
    mask)
        "$study_python" "$study_script" "${study_common[@]}" \
            --out "$study_results/pipeline_budget_v2_mask.jsonl" \
            --models v41-c1 --rows 256,1024,4096 \
            --contexts 8192,32768 --candidate mask
        ;;
    *) exit 2 ;;
esac
