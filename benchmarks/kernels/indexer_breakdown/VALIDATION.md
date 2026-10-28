# Validation and reproduction

The evidence has separate levels: controlled operator correctness, actual
PR-native pipeline behavior, and historical-baseline model allocation.
Microbenchmark speedups are not model throughput or accuracy validation.

Commands run from the experiment worktree using its virtual environment:

```bash
.venv/bin/python benchmarks/kernels/indexer_breakdown/pipeline_audit.py
bash benchmarks/kernels/indexer_breakdown/topk_analyze.sh
bash benchmarks/kernels/indexer_breakdown/topk_dispatch_audit.sh
bash benchmarks/kernels/indexer_breakdown/topk_plot.sh
.venv/bin/pre-commit run
```

The GPU experiments are launched by the archived topk/pipeline scripts;
paths, native-library hashes and actual commands are also recorded per case.
`ENV_memory.md` records the allocator/model runner commands. New optimization
commands and end-aware consumer restrictions are in `OPTIMIZATION.md`.

- Top-k: 67901 timed raw rows, 21754 complete valid 100x3 groups, zero JSON
  parse errors and zero duplicate timed keys. Noisy groups and invalid
  selections are retained and excluded from stable winner claims.
- Pipeline audit: no accepted score/index errors, no rejected case with timing,
  actual-final-PR helper provenance present in changed paths.
- Memory: 129 valid native operator cases, nine successful TP4 launches;
  earlier failed starts and obsolete input diagnostics are separated.
- Optimization: zero accepted correctness/audit failures; three-layer schedule
  reuse updates within replay, and valid-prefix masking checks use an explicit
  end-aware consumer. Missing edge coverage is listed before production use.
- Python syntax compilation and shell syntax checks passed. Repository hooks
  cover Ruff, formatting, Markdown, ShellCheck and import/API restrictions.

Large allocator snapshots and the GPU core stay in the external paths named
by the manifests, with hashes. Portable real-logit inputs and the two compact
cuda-gdb evidence logs are included. `results/final_manifest.json` inventories
committed raw JSONL bytes/hashes and statuses; derived rows and attempted runs
must not be counted as additional independent GPU measurements.

Final integration hooks passed without skips. The allocator snapshot readers
use a primitive-only opcode interpreter; actual operator/model summaries,
including a 70MB model snapshot, match the existing golden outputs. Object
construction/execution opcodes and trailing payload rejection were checked.
The final raw-file hash manifest and benchmark-only changed-file scope were
also verified before committing.
