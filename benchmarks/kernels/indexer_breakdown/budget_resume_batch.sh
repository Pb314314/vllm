#!/usr/bin/env bash
set -euo pipefail
study_root=/home/bopang/cg-project/wt-indexer-topk-breakdown
native_root=/home/bopang/cg-project/wt-b4a0fix
study_python=/home/bopang/cg-project/venv-b4a0fix/bin/python
study_results="$study_root/benchmarks/kernels/indexer_breakdown/results"
study_cache=/home/bopang/cg-project/design/idx-breakdown-cache
cd "$native_root"
export PYTHONPATH="$native_root"
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR="$study_cache/budget-v2/triton"
if [[ $(git rev-parse HEAD) != 4e5b4f8ce9b067d92e7373d950727d8238e9a289 ]]; then
    echo 'Native source revision changed; refusing to relabel results.' >&2
    exit 2
fi
monitor_pid=''
cleanup() {
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
nvidia-smi --query-gpu=timestamp,uuid,clocks.sm,temperature.gpu,power.draw,power.limit,pstate,clocks_event_reasons.active,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_thermal_slowdown --format=csv,noheader,nounits --loop=1 \
    > "$study_cache/budget-resume-${SLURM_JOB_ID}-telemetry.log" 2>&1 &
monitor_pid=$!
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 bash \
    "$study_root/benchmarks/kernels/indexer_breakdown/budget_native_launch.sh" core \
    > "$study_cache/budget-resume-${SLURM_JOB_ID}-core.log" 2>&1 &
core_pid=$!
CUDA_VISIBLE_DEVICES=1 taskset -c 16-31 bash \
    "$study_root/benchmarks/kernels/indexer_breakdown/budget_native_launch.sh" long \
    > "$study_cache/budget-resume-${SLURM_JOB_ID}-long.log" 2>&1 &
long_pid=$!
CUDA_VISIBLE_DEVICES=2 taskset -c 70-85 "$study_python" \
    "$study_root/benchmarks/kernels/indexer_breakdown/budget_graph_replay.py" \
    --out "$study_results/budget_graph_replay_v2.jsonl" --resume \
    > "$study_cache/budget-resume-${SLURM_JOB_ID}-replay.log" 2>&1 &
replay_pid=$!
(
    export CUDA_VISIBLE_DEVICES=3
    # Repeat the one partly measured case entirely on this physical GPU.
    taskset -c 104-119 "$study_python" \
        "$study_root/benchmarks/kernels/indexer_breakdown/pipeline_bench.py" \
        --out "$study_results/pipeline_budget_v2_core_recheck.jsonl" \
        --models v4 --dtypes fp8 --rows 2048 --contexts 32768 \
        --variants single-auto,single-ds,ds-512,ds-1024,ds-2048,final-512,final-1024,final-2048 \
        --production-helper --strict-cold-l2 --repeat 100 --trials 3 \
        --cache warm,cold --resume
    for study_rows in 1024 128; do
        taskset -c 104-119 "$study_python" \
            "$study_root/benchmarks/kernels/indexer_breakdown/pipeline_interleaved.py" \
            --out "$study_results/pipeline_budget_v2_interleaved_${study_rows}.jsonl" \
            --rows "$study_rows" --variants single-auto,single-ds,ds-512,ds-1024,ds-2048,final-512,final-1024,final-2048 \
            --production-helper --strict-cold-l2
    done
) > "$study_cache/budget-resume-${SLURM_JOB_ID}-controls.log" 2>&1 &
controls_pid=$!
wait "$core_pid"
wait "$long_pid"
wait "$replay_pid"
wait "$controls_pid"
printf 'Budget GPU continuation finished; raw data awaits final cross-file audit.\n'
