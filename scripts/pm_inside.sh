#!/bin/bash
# Inside-container phase for FIX4 precompiled-main arms.
# env: J (workdir), RESULTS, TASK, MODEL, AFTER, EXTRA_VLLM_ARGS, SKIP_GROUNDING, HEALTH_ITERS, HF_HOME
set -eo pipefail
if [ -n "${EXTRA_VLLM_ARGS_B64:-}" ]; then
  EXTRA_ARGS=$(echo "$EXTRA_VLLM_ARGS_B64" | base64 -d)
else
  EXTRA_ARGS="${EXTRA_VLLM_ARGS:-}"
fi
python3 -m venv --system-site-packages "$J/venv"
cd "$J/wt"
export VLLM_PRECOMPILED_WHEEL_COMMIT=d60c4404e2b5c4ffb54873fb18e06a335e1f2a94
export SETUPTOOLS_SCM_PRETEND_VERSION=0.33.0rc1.dev872+gd60c4404e
VLLM_USE_PRECOMPILED=1 "$J/venv/bin/pip" install --no-deps -e . > "$RESULTS/wheel_install.log" 2>&1 || { tail -30 "$RESULTS/wheel_install.log"; exit 1; }
grep -i "precompiled\|wheel" "$RESULTS/wheel_install.log" | tail -3 >> "$RESULTS/provenance.txt" || true
{
  "$J/venv/bin/python" -c 'import vllm; print("vllm.__file__:", vllm.__file__)'
  "$J/venv/bin/python" -c 'import vllm.model_executor.models.mimo_v2_omni as m; print("module:", m.__file__)'
} >> "$RESULTS/provenance.txt" 2>&1
HAS=$("$J/venv/bin/python" -c 'import inspect, vllm.model_executor.models.mimo_v2_omni as m; print("_video_timestamps_from_metadata" in inspect.getsource(m))')
echo "marker=$HAS" >> "$RESULTS/provenance.txt"
[ "$HAS" = "$EXPECTED_MARKER" ] || { echo "FAIL: module marker mismatch (want $EXPECTED_MARKER got $HAS)"; exit 1; }
# CRITICAL: the editable install's .pth appends the worktree AFTER system
# dist-packages, so the image's vllm would shadow it (and did). Force the
# worktree first for every venv invocation, and re-assert with a neutral cwd.
export PYTHONPATH="$J/wt"
cd /
HAS2=$("$J/venv/bin/python" -c 'import inspect, vllm.model_executor.models.mimo_v2_omni as m; assert m.__file__.startswith("'"$J"'/wt/"), m.__file__; print("_video_timestamps_from_metadata" in inspect.getsource(m))')
[ "$HAS2" = "$EXPECTED_MARKER" ] || { echo "FAIL: shadowed import (marker=$HAS2)"; exit 1; }
echo "shadow-check ok (marker=$HAS2, cwd-independent)"
{
  "$J/venv/bin/python" -c 'import sys, torch, transformers; print("python", sys.version.split()[0]); print("torch", torch.__version__); print("transformers", transformers.__version__)'
  echo "=== compiled extensions in worktree:"
  ls -la "$J"/wt/vllm/*.so 2>/dev/null
  sha256sum "$J"/wt/vllm/_C.abi3.so 2>/dev/null || true
} >> "$RESULTS/provenance.txt" 2>&1


# in-situ label probe through the real frontend path (parse_chat_messages + processor)
CLIP=$TASKDIR/clips2/g60.mp4 MODEL="$MODEL" HF_HOME=$DATADIR/hf-cache \
  "$J/venv/bin/python" "$TASK/labelprobe2.py" > "$RESULTS/labelprobe.txt" 2>"$RESULTS/labelprobe.err" || true
grep "^labels:" "$RESULTS/labelprobe.txt" || tail -5 "$RESULTS/labelprobe.err"

if [ "${PROBE_ONLY:-0}" = "1" ]; then echo PROBE_DONE; exit 0; fi

echo "PHASE: weights check"
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub} "$J/venv/bin/python" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$MODEL')
print('weights at', p)
"
echo "PHASE: serve start"
"$J/venv/bin/vllm" serve "$MODEL" --served-model-name mimo --port 8000 \
  --tensor-parallel-size 4 --max-model-len 32768 --trust-remote-code \
  --limit-mm-per-prompt '{"image":2,"video":2}' --gpu-memory-utilization 0.9 \
  $EXTRA_ARGS > "$RESULTS/server.log" 2>&1 &
SPID=$!
for i in $(seq 1 "${HEALTH_ITERS:-240}"); do
  curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; tail -60 "$RESULTS/server.log"; exit 1; }
  sleep 15
done
curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 || { echo "SERVER TIMEOUT"; tail -40 "$RESULTS/server.log"; exit 1; }
{
  "$J/venv/bin/python" -c '
import hashlib, sys, vllm
mods = [(n, getattr(m, "__file__", None)) for n, m in sys.modules.items() if getattr(m, "__file__", None) and ".so" in str(getattr(m, "__file__"))]
for n, f in mods: print("loaded native:", n, f)
import glob, os
for f in sorted(glob.glob(os.path.join(os.path.dirname(vllm.__file__), "*.so"))):
    print("worktree so:", os.path.basename(f), hashlib.sha256(open(f, "rb").read()).hexdigest()[:16])
' || echo "native digest unavailable"
  "$J/venv/bin/python" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$MODEL', allow_patterns=['config.json'])
print('checkpoint revision:', p.split('/snapshots/')[-1])
" || echo "snapshot revision unavailable"
} >> "$RESULTS/provenance.txt" 2>&1
grep -E "KV cache size|Graph|cudagraph|CUDA graph" "$RESULTS/server.log" | tail -8 > "$RESULTS/kv_graph.txt" || true
echo SERVER_UP
MICROBENCH_MODEL="$MODEL" "$J/venv/bin/python" "$TASK/microbench.py" > "$RESULTS/microbench.json" 2>"$RESULTS/microbench.err" || true
SKIP_GROUNDING=$SKIP_GROUNDING python3 "$TASK/fix4_client.py" --url http://127.0.0.1:8000 --model mimo --output "$RESULTS/client.json" > "$RESULTS/client.log" 2>&1 || true
kill $SPID 2>/dev/null || true; sleep 20; kill -9 $SPID 2>/dev/null || true
echo CLIENT_DONE
