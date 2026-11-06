#!/bin/bash
#SBATCH --job-name=v-d3c25120-mainfinal
#SBATCH --nodes=1
#SBATCH --time=3:30:00
#SBATCH --mem=64G
#SBATCH -p normal
#SBATCH -x gb200-rack1-[01-03,06,09,12-17]
#SBATCH -o /home/agent-v/v/d3c25120/results/final-%j/out.txt
# FIX11 definitive run at current main 86564975c5a5: prewarm, warm B B A A
# no-purge spread, then purge pair. Full instrumentation: sitecustomize shm
# trace (create/attach/unlink with pid), full install log, extracted + loaded
# extension hashes, precise purge of traced own segments only.
set -o pipefail
T=/home/agent-v/v/d3c25120
RUN=$T/results/final-$SLURM_JOB_ID
mkdir -p "$RUN"
J=
for BASE in /tmp/d3c25120 /mnt/data/agent-v/d3c25120; do
  if mkdir -p "$BASE" 2>/dev/null && touch "$BASE/.w" 2>/dev/null; then
    AVAIL=$(df -BG --output=avail "$(dirname "$BASE")" 2>/dev/null | tail -1 | tr -dc 0-9)
    if [ "${AVAIL:-0}" -ge 180 ]; then J=$BASE/final-$SLURM_JOB_ID; break; fi
  fi
done
[ -z "$J" ] && { echo "FATAL: no disk"; exit 1; }
mkdir -p "$J/instr"
cp $T/sitecustomize.py $J/instr/
echo "workdir: $J ($(hostname))" | tee "$RUN/info.log"
CN=fx11f-$SLURM_JOB_ID
JE=/mnt/data/agent-v/d3c25120-final-enroot-$SLURM_JOB_ID
mkdir -p "$JE" 2>/dev/null || JE=$J/enroot
export ENROOT_RUNTIME_PATH=$JE/runtime ENROOT_CACHE_PATH=$JE/cache ENROOT_DATA_PATH=$JE/data
export ENROOT_SYSCONF_PATH=/home/agent-v/v/2d0f499c/enroot-sysconf
mkdir -p "$ENROOT_RUNTIME_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"
cleanup() { enroot remove -f "$CN" 2>/dev/null || true; rm -rf "$J" "$JE"; }
trap cleanup EXIT
SQSH=/home/agent-v/v/2d0f499c/images/vllm-nightly-cu134-arm64-latest.sqsh
BARE=$T/vllm-bare.git
BASEC=86564975c5a5ba96b0bea94708101480ca18a2cd
SNAP=/mnt/model-cache/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
MNT="-m /home/agent-v:/home/agent-v -m /mnt/model-cache:/mnt/model-cache -m $J:$J"
RC="--rc $T/rc_exec.sh"
FILES="vllm/distributed/device_communicators/shm_broadcast.py vllm/v1/executor/multiproc_executor.py vllm/v1/executor/ray_executor_v2.py"

enroot create -n "$CN" "$SQSH" || { echo "enroot create failed"; exit 1; }

git --git-dir="$BARE" worktree remove -f $J/wt 2>/dev/null || true
rm -rf $J/wt
git --git-dir="$BARE" worktree add -f --detach $J/wt $BASEC
# FULL install log (no filtering): shows the resolved precompiled wheel commit
enroot start -w $MNT $RC "$CN" sh -c "
  set -e
  python3 -m venv --system-site-packages $J/venv
  cd $J/wt && SETUPTOOLS_SCM_PRETEND_VERSION=0.33.0rc1.dev0+g86564975c5 VLLM_USE_PRECOMPILED=1 $J/venv/bin/pip install --no-deps -e .
  echo '--- extracted extension hashes:'
  sha256sum $J/wt/vllm/*.abi3.so
" > "$RUN/install.log" 2>&1 || { echo "FATAL: install failed"; tail -20 "$RUN/install.log"; exit 1; }
tail -15 "$RUN/install.log"
ls $J/wt/vllm/_C*.so >/dev/null || { echo "FATAL: precompiled artifacts missing"; exit 1; }

set_arm() {  # $1 = baseline|after — host-side file swap + grep assert (fatal)
  if [ "$1" = after ]; then
    cp $T/patched-main/shm_broadcast.py $J/wt/vllm/distributed/device_communicators/shm_broadcast.py || return 1
    cp $T/patched-main/multiproc_executor.py $J/wt/vllm/v1/executor/multiproc_executor.py || return 1
    cp $T/patched-main/ray_executor_v2.py $J/wt/vllm/v1/executor/ray_executor_v2.py || return 1
    grep -q "_attach_input_mq" $J/wt/vllm/v1/executor/multiproc_executor.py || return 1
  else
    git -C $J/wt checkout -q $BASEC -- $FILES || return 1
    ! grep -q "_attach_input_mq" $J/wt/vllm/v1/executor/multiproc_executor.py || return 1
  fi
  echo "ARM $1 files OK"
}

wait_gpu_free() {
  for i in $(seq 1 60); do
    local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${used:-0}" -lt 10000 ] && return 0
    sleep 5
  done
  return 1
}

hash_loaded_exts() {  # $1 = cell dir; run while serve is up
  local CELL=$1
  ( for p in $(pgrep -f "vllm serve"); do
      grep -hoE '/[^ ]*vllm/[^ ]*\.abi3\.so' /proc/$p/maps 2>/dev/null
    done | sort -u | xargs -r sha256sum ) > "$CELL/loaded_ext.sha256" 2>/dev/null || true
}


purger() {  # $1 = serve log, $2 = cell dir; purge ONLY segments this serve created (per trace)
  local log=$1 cell=$2
  for i in $(seq 1 240); do
    grep -q "DP group leader" "$log" 2>/dev/null && break
    sleep 2
  done
  sleep 90
  python3 - "$cell" > "$cell/purge.log" <<'PYEOF'
import json, sys
cell = sys.argv[1]
names = set()
try:
    for line in open(f"{cell}/shm_trace.jsonl"):
        e = json.loads(line)
        if e["event"] == "create" and e["name"].startswith("psm_"):
            names.add("/dev/shm/" + e["name"])
except FileNotFoundError:
    pass
print("PURGE targets (created by this serve, per shm trace):")
for n in sorted(names):
    print(n)
with open(f"{cell}/purge.names", "w") as f:
    f.write("\n".join(sorted(names)) + "\n")
PYEOF
  echo "PURGE at $(date +%H:%M:%S)" | tee -a "$cell/purge.log"
  xargs -r rm -f < "$cell/purge.names" 2>/dev/null || true
}

serve_cell() {  # $1 label, $2 purge(0|1), $3 arm
  local LABEL=$1 PURGE=$2 ARM=$3
  local CELL=$RUN/$LABEL; mkdir -p "$CELL"
  wait_gpu_free || echo "$LABEL WARN: gpu not free before start" | tee -a "$RUN/timings.log"
  set_arm $ARM || { echo "$LABEL INVALID: set_arm failed" | tee -a "$RUN/timings.log"; return 1; }
  enroot start -w $MNT $RC "$CN" $J/venv/bin/python -c "
import inspect, sys
sys.path.insert(0, '$J/wt')
import vllm
import vllm.v1.executor.multiproc_executor as m
ok_path = vllm.__file__.startswith('$J/wt/')
marker = '_attach_input_mq' in inspect.getsource(m.WorkerProc)
print('PROVENANCE vllm.__file__=', vllm.__file__, 'marker=', marker, 'arm=$ARM')
assert ok_path and marker == ('$ARM' == 'after')
" > "$CELL/provenance.log" 2>&1 || { echo "$LABEL INVALID: provenance failed" | tee -a "$RUN/timings.log"; cat "$CELL/provenance.log"; return 1; }
  local T0=$(date +%s)
  local PURGE_PID=""
  if [ "$PURGE" = 1 ]; then purger "$CELL/server.log" "$CELL" & PURGE_PID=$!; fi
  enroot start $MNT -m /mnt/data/agent-v:/mnt/data/agent-v \
    -e PYTHONPATH=$J/wt:$J/instr -e SHM_TRACE=$CELL/shm_trace.jsonl \
    -e HF_HUB_OFFLINE=1 -e HF_HUB_CACHE=$J/hf-cache -e PYTHONUNBUFFERED=1 -e TMPDIR=$J \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e NCCL_MNNVL_ENABLE=1 -e NCCL_CUMEM_ENABLE=1 \
    -e TRITON_CACHE_DIR=$J/triton -e TORCH_EXTENSIONS_DIR=$J/torchext \
    -e FLASHINFER_WORKSPACE_BASE=$J/flashinfer \
    $RC "$CN" $J/venv/bin/vllm serve "$SNAP" \
    --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --port 8000 \
    --disable-uvicorn-access-log \
    --tensor-parallel-size 4 --kv-cache-dtype fp8 --no-enable-prefix-caching \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","enable_adaptive_verification":false}' \
    --engram-config '{"cpu_offload":false,"embedding_across_dp":true,"use_thp":false}' \
    --max-model-len 32768 --max-num-seqs 32 --max-num-batched-tokens 1260 \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":128}' \
    > "$CELL/server.log" 2>&1 &
  local VPID=$!
  local status=timeout
  for i in $(seq 1 480); do
    if ! kill -0 $VPID 2>/dev/null; then status=died; break; fi
    if curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then status=ready; break; fi
    sleep 5
  done
  local T1=$(date +%s)
  echo "$LABEL status=$status startup_s=$((T1-T0))" | tee -a "$RUN/timings.log"
  [ -n "$PURGE_PID" ] && kill $PURGE_PID 2>/dev/null
  grep -m1 -E "AttributeError|could not attach" "$CELL/server.log" | cut -c1-200 || true
  if [ "$status" = ready ]; then
    hash_loaded_exts "$CELL" || true
    curl -s --max-time 60 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
      -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","prompt":"18 + 25 = 43\n120 + 7 = 127\n64 + 33 =","max_tokens":16,"temperature":0}' \
      > "$CELL/smoke.json"; head -c 260 "$CELL/smoke.json"; echo
  fi
  kill $VPID 2>/dev/null; sleep 8; kill -9 $VPID 2>/dev/null || true; sleep 5
}

# prewarm (cold JIT + page cache), discarded
set_arm baseline || exit 1
echo "=== prewarm cell (discarded)"
serve_cell prewarm-discard 0 baseline

# warm no-purge spread: B B A A
serve_cell warm-baseline-r1 0 baseline
serve_cell warm-baseline-r2 0 baseline
serve_cell warm-after-r1 0 after
serve_cell warm-after-r2 0 after

# purge pair
serve_cell purge-baseline 1 baseline
serve_cell purge-after 1 after

echo "=== timings"; cat "$RUN/timings.log"
touch "$RUN/DONE"
echo "FINAL-E2E DONE -> $RUN"
