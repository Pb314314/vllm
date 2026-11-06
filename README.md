# FIX4 evidence bundle — Pb314314/vllm PR #3 (MiMo-V2 video timestamps)

Raw artifacts for the measurements quoted in the PR body. All runs were executed at base
commit `d60c4404e2b5c4ffb54873fb18e06a335e1f2a94` ("current main" as of 2026-11-05 22:15
UTC; the 5-commit delta to `bc9a3ac5dc` touches no MiMo file) via `VLLM_USE_PRECOMPILED`
editable installs (wheel `vllm-0.33.0rc1.dev872+gd60c4404e.precompiled`, pinned with
`VLLM_PRECOMPILED_WHEEL_COMMIT`). Every arm asserts the loaded `mimo_v2_omni.py` module
path from a neutral cwd plus the presence (After) or absence (Baseline) of the
`_video_timestamps_from_metadata` marker, and records the full overlay diff in
`provenance.txt`.

Layout:

- `scripts/` — the exact scorer, clip generator, client, probes and arm runner. Paths are
  scrubbed (`$TASKDIR`, `$DATADIR`, `$WORKDIR` placeholders). `fix4_client.py` is the
  corrected client (full responses + finish_reason); `fix4_client_legacy.py` is the
  client that produced the retracted accuracy numbers (kept so the retracted records
  stay reproducible). `scorer_selftest.json` is a 3-entry self-test input
  demonstrating the corrected scorer: completed answers score normally (1/3 below), unfinished
  responses are excluded from hits (`1/3 (unfinished: 1)`).
- `clips/` — the three synthetic grounding clips (solid-color flashes pixel-verified onto
  loader-sampled frames).
- `manifest.json` — decoded `frames_indices`, `fps`, per-event true media times.
- `runs/<arm>/` — per-arm `client.json` (raw answers, completion tails, perf timings,
  prompt tokens), `labelprobe.txt` (frontend-path label dump), `provenance.txt` (commit,
  wheel log, module path, marker, overlay diff, GPU/torch/transformers versions),
  `microbench.json` (30-rep processor latency), `kv_graph.txt` (KV cache / cudagraph
  lines from the server log).

Arm naming: `pm2-b*` = Baseline (worktree as-is), `pm2-a*` = After (PR's `mimo_v2_omni.py`
overlaid — diff recorded in provenance.txt); `pm2-<cell>-*` for the satellite checkpoints
(V2.5, Flash-MOPD, Pro-RL, MTP spec-decode). The `pm-b*`/`pm-a*` runs are an earlier batch
whose editable-install sys.path shadowing made both arms execute the same image code —
they are kept as the **same-code noise band** referenced in the perf analysis and are
labeled as such; they are not a valid Baseline/After comparison.

Scoring: `python3 scripts/score_grounding.py runs/<arm>/client.json manifest.json`
(tolerance = `2 * frame_step` per clip, identical for both arms).
