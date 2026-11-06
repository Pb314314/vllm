#!/bin/bash
#SBATCH --job-name=v-d3c25120-unittest
#SBATCH --nodes=1
#SBATCH --time=0:40:00
#SBATCH --mem=64G
#SBATCH -p normal
#SBATCH -x gb200-rack1-[01-03,06,09,12-17]
#SBATCH -o /home/agent-v/v/d3c25120/results/unit-%j/out.txt
# FIX11 round-2: run the 7 focused shm_broadcast tests (CPU-only) with the
# patched overlay (3 files) on the 11-03 nightly image.
set -o pipefail
T=/home/agent-v/v/d3c25120
RUN=$T/results/unit-$SLURM_JOB_ID
mkdir -p "$RUN"
SQSH=/home/agent-v/v/2d0f499c/images/vllm-nightly-cu134-arm64-latest.sqsh
CN=fx11ut-$SLURM_JOB_ID
J=/mnt/data/agent-v/d3c25120-unit-$SLURM_JOB_ID
export ENROOT_RUNTIME_PATH=$J/runtime ENROOT_CACHE_PATH=$J/cache ENROOT_DATA_PATH=$J/data
export ENROOT_SYSCONF_PATH=/home/agent-v/v/2d0f499c/enroot-sysconf
mkdir -p "$ENROOT_RUNTIME_PATH" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$J/local"
cleanup() { enroot remove -f "$CN" 2>/dev/null || true; rm -rf "$J"; }
trap cleanup EXIT

enroot create -n "$CN" "$SQSH" || exit 1
PKG=/usr/local/lib/python3.12/dist-packages/vllm
enroot start \
  -m $T/patched/shm_broadcast.py:$PKG/distributed/device_communicators/shm_broadcast.py \
  -m $T/patched/multiproc_executor.py:$PKG/v1/executor/multiproc_executor.py \
  -m $T/patched/ray_executor_v2.py:$PKG/v1/executor/ray_executor_v2.py \
  -m $T/unit_shm_test.py:/unit_shm_test.py \
  -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=2 -e PYTHONUNBUFFERED=1 -e TMPDIR=$J/local \
  --rc $T/rc_exec.sh "$CN" \
  python3 /unit_shm_test.py 2>&1 | tail -25
echo "unit exit: ${PIPESTATUS[0]}"
