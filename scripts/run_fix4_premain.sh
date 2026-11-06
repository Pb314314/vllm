#!/bin/bash
# FIX4 precompiled-main arm runner, v3 (matches the proven run_fix4_arm.sh pattern).
# Baseline = worktree at the exact base commit, VLLM_USE_PRECOMPILED wheel.
# After = same + PR's mimo_v2_omni.py copied in (marker asserted).
# env (via --export): RUN_LABEL, MODEL?, AFTER?, EXTRA_VLLM_ARGS?, SKIP_GROUNDING?, HEALTH_ITERS?
set -eo pipefail
T=$TASKDIR
SQSH=$IMAGEDIR/vllm-nightly-cu134-arm64-latest.sqsh
BASE=d60c4404e2b5c4ffb54873fb18e06a335e1f2a94
BARE=$T/vllm-bare.git
MODEL=${MODEL:-XiaomiMiMo/MiMo-V2.6-Flash-RL}
AFTER=${AFTER:-0}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-}
SKIP_GROUNDING=${SKIP_GROUNDING:-0}
HEALTH_ITERS=${HEALTH_ITERS:-240}
RUN=$T/results/$RUN_LABEL
mkdir -p "$RUN"
rm -f "$RUN"/{READY,FAILED,DONE}

# worktree + venv live in worker-local /tmp (per precompiled-main README); enroot on $DATADIR
FREE_G=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')
[ "${FREE_G:-0}" -ge 150 ] || { echo "FAIL: /tmp has ${FREE_G}G free (<150G)"; exit 1; }
J=/tmp/fix4pm-$SLURM_JOB_ID
mkdir -p "$J"
E=$DATADIR/enroot-job/$SLURM_JOB_ID-$RUN_LABEL
if ! mkdir -p "$E" 2>/dev/null; then
  echo "FATAL: $DATADIR not writable on $(hostname)"; touch "$RUN/FAILED"; exit 1
fi
CN=fix4pm-$SLURM_JOB_ID
export ENROOT_RUNTIME_PATH=$E/runtime ENROOT_CACHE_PATH=$E/cache ENROOT_DATA_PATH=$E/data
export ENROOT_SYSCONF_PATH=$ENROOT_SYSCONF
unset XDG_RUNTIME_DIR
mkdir -p "$ENROOT_RUNTIME_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"
HFC=$DATADIR/hf-cache
mkdir -p "$HFC"

df -BG /tmp $DATADIR > "$RUN/df_preflight.txt"
git --git-dir="$BARE" worktree add -f --detach "$J/wt" "$BASE"
if [ "$AFTER" = "1" ]; then
  cp "$T/after-files/mimo_v2_omni.py" "$J/wt/vllm/model_executor/models/mimo_v2_omni.py"
fi
{
  git -C "$J/wt" rev-parse HEAD
  echo "AFTER=$AFTER MODEL=$MODEL EXTRA=$EXTRA_VLLM_ARGS"
  nvidia-smi -L 2>/dev/null | head -5
} > "$RUN/provenance.txt"
{
  echo "=== overlay diff vs worktree base (empty on Baseline)"
  git -C "$J/wt" diff --stat
  git -C "$J/wt" diff
  echo "=== encoder-DP markers in the effective model file"
  grep -c "supports_encoder_tp_data" "$J/wt/vllm/model_executor/models/mimo_v2_omni.py" || true
  grep -c "0 if use_data_parallel else" "$J/wt/vllm/model_executor/models/mimo_v2_omni.py" || true
} >> "$RUN/provenance.txt" 2>&1

enroot create -n "$CN" "$SQSH"
trap 'enroot remove -f "$CN" 2>/dev/null || true; rm -rf "$J" "$E"' EXIT

EXPECTED_MARKER=$([ "$AFTER" = "1" ] && echo True || echo False)
enroot start --rc $T/rc_exec.sh \
  -m "$J:$J" -m $HOME:$HOME -m $DATADIR:$DATADIR \
  -e HF_HUB_CACHE=$HFC -e PYTHONUNBUFFERED=1 -e TMPDIR=$J \
  -e J=$J -e TASK=$T -e RESULTS=$RUN -e MODEL="$MODEL" \
  -e EXTRA_VLLM_ARGS="$EXTRA_VLLM_ARGS" -e EXTRA_VLLM_ARGS_B64="${EXTRA_VLLM_ARGS_B64:-}" -e SKIP_GROUNDING=$SKIP_GROUNDING \
  -e HEALTH_ITERS=$HEALTH_ITERS -e EXPECTED_MARKER=$EXPECTED_MARKER -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  "$CN" bash $T/pm_inside.sh
