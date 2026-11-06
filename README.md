# FIX11 evidence — ShmRingBuffer attach race (PR #11)

Raw artifacts for Pb314314/vllm#11 ("[V] ShmRingBuffer: attach before load + fail-fast on
unlinked segment"). Three runs, all on a single GB200 node (TP4) via Slurm + enroot; the
enroot image (`vllm-nightly-cu134-arm64-latest.sqsh`, vllm `0.33.0rc1.dev573+g40941f2c4.d20261103.cu134`)
supplies only torch/CUDA/dependencies in the main-source runs.

## Layout

- `final-46332/` — **definitive run (2026-11-06 13:14–14:20 UTC)** at current upstream main
  `86564975c5a5` (worktree + `VLLM_USE_PRECOMPILED=1`, `PYTHONPATH=<worktree>`; after arms =
  PR head `75d5447a1`'s exact files). Prewarm cell (cold, discarded), then warm
  baseline/baseline/after/after no-purge, then purge pair. Per cell: `provenance.log`
  (vllm.__file__ + marker assert), `server.log` (full), `shm_trace.jsonl` (every POSIX-shm
  create/attach/unlink with pid/ppid/ts, via `harness/sitecustomize.py`), purge cells also
  `purge.log`/`purge.names` (exact own-segment targets) and `smoke.json` when ready;
  `purge-after/loaded_ext.sha256` (hashes from /proc/<serve pids>/maps). Plus `timings.log`,
  `install.log` (full, unfiltered), `info.log`, `out.txt` (driver script stdout).
- `main-46103/` — earlier run at base `073049553c2` (same method; after cells used the
  nightly-backported file versions — see the PR's provenance note).
- `e2e-45915/` — first run: 2026-11-03 nightly image with the PR's files bind-mounted over
  the installed package (baseline died with the production AttributeError; patched READY).
- `unit-46077/` — in-container run of the standalone unit checks (T1–T6; T6 skips, no ray in
  the image).
- `harness/` — the exact job scripts (`e2e*.sbatch`, `unit.sbatch`), the standalone unit
  runner, and the shm-tracing sitecustomize shim.

## Headline results (final-46332)

- warm startup spread: baseline 436/435 s, after 440/411 s (no regression);
- purge-baseline: died at 295 s with `AttributeError: 'ShmRingBuffer' object has no
  attribute 'shared_memory'` (ring created T0, *no attach ever* in shm_trace, purged T0+90 s,
  crash after load_model);
- purge-after: ready at 471 s; ring attached by all 4 workers at T0+9.3–9.8 s (before load),
  purge harmless; smoke answer correct.
