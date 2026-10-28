#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
taskset -pc "${TOPK_CPUSET:-70-85}" $$
cd /home/bopang/cg-project/wt-indexer-topk-breakdown
export CUDA_VISIBLE_DEVICES="${TOPK_GPU:-2}"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR=/home/bopang/cg-project/design/idx-breakdown-cache/topk/triton
export XDG_CACHE_HOME=/home/bopang/cg-project/design/idx-breakdown-cache/topk/xdg
export CUDA_CACHE_PATH=/home/bopang/cg-project/design/idx-breakdown-cache/topk/cuda
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_GENERATION_FLAGS=skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory
export CUDA_COREDUMP_FILE=/home/bopang/cg-project/design/idx-breakdown-cache/topk/debug/fi_coredump_%h.%p.%t
export CUDA_LAUNCH_BLOCKING="${TOPK_LAUNCH_BLOCKING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${TOPK_ALLOC_CONF:-}"
mkdir -p /home/bopang/cg-project/design/idx-breakdown-cache/topk/debug
.venv/bin/python benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py \
    --output "benchmarks/kernels/indexer_breakdown/results/${TOPK_RESULT_STEM:-topk_fi_ima_probe}.jsonl" \
    --preset manual --rows 4096 --capacities 524288 --profiles file \
    --logits-file /home/bopang/cg-project/design/idx-breakdown-cache/pipeline/score_samples/synthetic_deepgemm_v41-c2_mxfp4.pt \
    --schemes UP+FI --bounds actual --capture-max-seq-len 1048576 \
    --compress-ratio 2 --modes eager --caches warm --trials 1 \
    --repeat 1 --detail-repeat 1 --host-repeat 1 --sentinel-every 0 \
    --max-seconds 120 --resume
