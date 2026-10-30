# Reproduction and evidence map

## Exact base and patch order

Use an isolated checkout of persistent PR1 commit `293880733f87a095acac787a996e68b097eac7cc`:

```bash
git apply /path/to/review/BF16_INTEGRATION.patch
git apply /path/to/review/ADDRESS_DEPENDENCY.patch
git apply /path/to/review/PR2_ONLY.patch
```

The first patch archives the tested reconciliation with cooperative #62172; it is a prerequisite, not a new PR2 contribution. The second changes two offset multiplications to `size_t` and is a separate correctness prerequisite for large unchunked baselines. Build the native extension following `docs/contributing/incremental_build.md` with the same environment/toolchain. Follow the repository's `uv`/`.venv/bin/python` rules; do not use system Python or bare pip.

The experiments already have an immutable built runtime at:
`/home/bopang/cg-project/design/indexer-chunk-default-20261030-v2/refinement/runtime`.
Its caller SHA256 is `ad09bbb1fc973efe250b7044f95df68e0e0a32936387663953a1fff6150e26f0`; native SHA256 is `f96b1e72533bc79d154f245d0ffc62817fd99fd2987e97b8e4e63a14dbe02f92`. The environment executable is recorded in each `IDENTITY.json`. A rebuilt binary can have a different byte hash; do not claim it is the measured artifact without revalidation.

## Quick functional verification in this environment

```bash
sbatch /home/bopang/cg-project/design/indexer-chunk-default-20261030-v2/refinement/gates.sbatch
```

This runs nine focused tests and two real-DeepGEMM edge cases, including empty rows, tail slicing, candidate scores, changed weights and graph length updates. Outputs use a fresh Slurm job ID. The script checks the runtime identity/default and imports the final test file. Inspect the log and both `RESULT.json` receipts; submission success alone is not test success.

For an installed matching build:

```bash
.venv/bin/python -m pytest tests/kernels/test_top_k_per_row.py -k decode_chunk -v
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/kernels/indexer_chunk/benchmark.py --case smoke --out /tmp/indexer-chunk-smoke-unique
```

Use a fresh output directory and idle GB200 GPU. Larger cases require tens of GiB of device memory. Do not run the largest unchunked baseline without the address-width correction.

## Full experiment commands used

All paths below are under `/home/bopang/cg-project/design/indexer-chunk-default-20261030-v2`:

|Question|Script|Completed/assigned job|
|---|---|---:|
|Broad 21-shape budget sweep, eager and graph|`kernel.sbatch`|41432|
|Largest-first versus smaller-first allocator ordering|`order_ab/run.sbatch`|41439|
|Final 1.5/2/2.5/3/4 GiB refinement, 10 shapes|`refinement/run.sbatch`|41449|
|Final 2/4 GiB model forward, four workloads|`refinement/model.sbatch`|41450|
|Focused correctness/profile follow-up|`refinement/gates.sbatch`|41453|
|V4 collocated A/B/A real-text smoke|`e2e/run.sbatch`|41456|
|V4.1 collocated A/B/A real-text smoke|`e2e_v41/run.sbatch`|41468|
|Balanced versus maximum-sized chunks with tails|`tail_policy/run.sbatch`|41474|
|Two-node NIXL PD A/B/A real-text smoke|`pd/run.sbatch`|41481|
|1/1.5/2/4 GiB final whole-model boundary check|`low_budget/model.sbatch`|41484|
|Candidate fallback without DeepSelect|`fallback/run.sbatch`|41498|
|Same-prepared-batch interleaved eager comparison|`eager_pairs/model.sbatch`|41510|
|Production large-graph startup, unchunked A|`startup/run.sbatch`|41502 (failed KV headroom)|
|Same startup, 2 GiB B only|`startup/run_B.sbatch`|41511|

Submit these scripts with `sbatch`; logs and fresh job-ID output paths are retained. They assert idle GPUs or refuse unowned work. The real-text drivers stop only their own servers/steps. Native regression job41431 and its 451-pass log are in the preceding `indexer-chunk-eager-graph-20261030-v1` package; its exact launch command/build receipt is retained there.

## Reading results

- `RESULT.json`: all measured records, mode/budget, raw timing samples, live peak, private graph reservation and kernel traces. Require `status=complete`.
- `IDENTITY.json`: input case, runtime/native/script hashes, GPU identity and loaded native mappings.
- `proof_*.json`: score digests, exact top-k score multisets, no whole-logits upcasts and replay checks. These are supported by executable assertions in the corresponding harness.
- `SUMMARY.json` / `RESULTS.md`: paired forward/reverse visits. They do not replace raw measurements.
- `model_*/RESULT_rank*.json`: per-rank timings, shape/chunk plans and numerical diagnostics. `hidden_outputs.pt` is retained separately and excluded from lightweight evidence archives.
- Serving `DRIVER.json`, `RESULT.json`, client records, NIXL metrics and `native_trace/chunk.*.json`: require both normal completion and actual path witnesses. Cleanup-triggered engine shutdown messages after successful results are not inference failures.
- `PRECOMMIT.log`: full final lint/type/documentation checks, exit status zero.

Model-forward graphs use real weights with synthetic tokens and zero historical KV. Real-text smokes force small budgets to exercise the path and are not performance evidence for the 2 GiB default. Keep these claims separate.
