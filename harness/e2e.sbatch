#!/bin/bash
#SBATCH --job-name=v-d3c25120-shmfix
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH -p normal
#SBATCH -x gb200-rack1-[01-03,06,09,12-17]
#SBATCH -o /home/agent-v/v/d3c25120/results/e2e-%j/out.txt
# FIX11 e2e: deterministic RemoveIPC simulation on single-node TP4 DSv4.1-Flash.
# Arm A (baseline, unpatched image): purge must kill the serve (AttributeError).
# Arm B (patched overlay): serve must reach READY and answer the smoke prompt.
set -uo pipefail
T=/home/agent-v/v/d3c25120
RUN=$T/results/e2e-$SLURM_JOB_ID
mkdir -p "$RUN"
SQSH=/home/agent-v/v/2d0f499c/images/vllm-nightly-cu134-arm64-latest.sqsh
SNAP=/mnt/model-cache/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
CN=fx11-$SLURM_JOB_ID
J=/mnt/data/agent-v/d3c25120-e2e-$SLURM_JOB_ID
JIT=/mnt/data/agent-v/jit/fix11
export ENROOT_RUNTIME_PATH=$J/runtime ENROOT_CACHE_PATH=$J/cache ENROOT_DATA_PATH=$J/data
export ENROOT_SYSCONF_PATH=/home/agent-v/v/2d0f499c/enroot-sysconf
mkdir -p "$ENROOT_RUNTIME_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$JIT" "$J/local"
cleanup() { enroot remove -f "$CN" 2>/dev/null || true; rm -rf "$J"; }
trap cleanup EXIT

echo "=== node $(hostname) job $SLURM_JOB_ID; image $SQSH"
rm -f /dev/shm/psm_* 2>/dev/null || true   # clear stale segments from earlier crashed runs
enroot create -n "$CN" "$SQSH" || { echo "enroot create failed"; exit 1; }

PKG=/usr/local/lib/python3.12/dist-packages/vllm
UNIT_MOUNTS="-m $T/unit_shm_test.py:/unit_shm_test.py"
PATCH_MOUNTS="-m $T/patched/shm_broadcast.py:$PKG/distributed/device_communicators/shm_broadcast.py -m $T/patched/multiproc_executor.py:$PKG/v1/executor/multiproc_executor.py"

echo "=== image vllm version:"
enroot start --rc $T/rc_exec.sh "$CN" vllm --version 2>/dev/null | tail -1

echo "=== unit test (patched overlay)"
enroot start $UNIT_MOUNTS $PATCH_MOUNTS --rc $T/rc_exec.sh "$CN" \
  python3 /unit_shm_test.py 2>&1 | tail -12
echo "unit exit: $?"

# purger: once the MQ segment exists, wait 90s (patched workers have attached by
# then; unpatched attach is ~5 min away at post-load), then delete all psm_*
purger() {
  local log=$1
  for i in $(seq 1 240); do
    grep -q "DP group leader" "$log" 2>/dev/null && break
    sleep 2
  done
  sleep 90
  echo "PURGE at $(date +%H:%M:%S)"
  # only this user's segments (sticky /dev/shm); mirrors logind RemoveIPC
  find /dev/shm -maxdepth 1 -name 'psm_*' -user "$(id -un)" -delete 2>/dev/null || true
}

serve_arm() {  # $1=label $2=extra-mounts
  local label=$1 extra=$2
  local log=$RUN/server-$label.log
  echo "=== arm $label starting $(date +%H:%M:%S)"
  purger "$log" &
  PURGE_PID=$!
  enroot start $extra \
    -m /mnt/model-cache:/mnt/model-cache -m /mnt/data/agent-v:/mnt/data/agent-v \
    -m /home/agent-v:/home/agent-v -m "$J/local:$J/local" -m "$JIT:$JIT" \
    -e HF_HUB_OFFLINE=1 -e NCCL_MNNVL_ENABLE=1 -e NCCL_CUMEM_ENABLE=1 \
    -e VLLM_ENGINE_READY_TIMEOUT_S=1800 -e PYTHONUNBUFFERED=1 -e TMPDIR=$J/local \
    -e TRITON_CACHE_DIR=$JIT/triton -e TORCH_EXTENSIONS_DIR=$JIT/torchext \
    -e FLASHINFER_WORKSPACE_BASE=$JIT/flashinfer \
    --rc $T/rc_exec.sh "$CN" vllm serve "$SNAP" \
    --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8 --no-enable-prefix-caching \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","enable_adaptive_verification":false}' \
    --engram-config '{"cpu_offload":false,"embedding_across_dp":true,"use_thp":false}' \
    --max-model-len 32768 --max-num-seqs 32 --max-num-batched-tokens 1260 \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":128}' \
    --disable-uvicorn-access-log --port 8000 > "$log" 2>&1 &
  VPID=$!
  local status=timeout
  for i in $(seq 1 200); do
    if ! kill -0 $VPID 2>/dev/null; then status=died; break; fi
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then status=ready; break; fi
    sleep 5
  done
  kill $PURGE_PID 2>/dev/null || true
  echo "=== arm $label status: $status"
  grep -m1 -E "AttributeError|could not attach" "$log" | cut -c1-200 || true
  if [ "$status" = ready ]; then
    curl -s --max-time 60 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
      -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","prompt":"18 + 25 = 43\n120 + 7 = 127\n64 + 33 =","max_tokens":16,"temperature":0}' \
      | tee "$RUN/smoke-$label.json"; echo
  fi
  kill $VPID 2>/dev/null || true; sleep 5; kill -9 $VPID 2>/dev/null || true
  sleep 10
}

serve_arm baseline ""
serve_arm patched "$PATCH_MOUNTS"
echo "=== done $(date +%H:%M:%S)"
