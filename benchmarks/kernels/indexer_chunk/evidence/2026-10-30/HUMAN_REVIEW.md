# Human review and reproduction: SM100 decode chunking

AI assistance was used. The code is pushed for review; no upstream PR2 has been opened. Publication is already authorized by the user, but human line review and relevant testing have not been confirmed.

## Read the isolated changes

- [PR2 only: all 11 changed files](https://github.com/Pb314314/vllm/compare/c86f902b07ec2adfe691e7322b38f9d688c86343...d51f40aeffe014e4ee1f77759b22501893f84d5e)
- [Separate cooperative BF16 integration prerequisite](https://github.com/Pb314314/vllm/commit/d839fe26bd21e37018b2bfb93526b9cd7d5bf44a)
- [Separate large-row address-width prerequisite](https://github.com/Pb314314/vllm/commit/c86f902b07ec2adfe691e7322b38f9d688c86343)
- [Execution flow and review order](CODE_REVIEW_GUIDE.md)
- [Full experimental report, regressions and limitations](REPORT.md)
- [Prepared English PR description](PR_DESCRIPTION.md)

PR2 adds 202 production Python lines, 46 test lines, 19 memory-guide lines and 1080 benchmark lines. The tested base integrates both persistent #64138 and cooperative #62172; it is not unmodified main. The feature is isolated from prerequisites in the linked diff. An upstream PR must clearly declare these dependencies and reconcile them before merge.

## Relevant tests for the human submitter

The following commands submit fresh, isolated GPU jobs using the already validated runtime in this environment. Read the completed logs and correctness receipts; successful submission is not a passing test.

```bash
# Existing FP32/BF16 kernel suite; the address fix is common to both arms.
sbatch /home/bopang/cg-project/design/indexer-chunk-eager-graph-20261030-v1/check.sbatch

# Final chunk planner/profile tests plus actual DeepGEMM edge/replay checks.
sbatch /home/bopang/cg-project/design/indexer-chunk-default-20261030-v2/refinement/gates.sbatch
```

For a matching locally rebuilt/installed checkout, the minimal entry points are:

```bash
.venv/bin/python -m pytest tests/kernels/test_top_k_per_row.py -k decode_chunk -v
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/kernels/indexer_chunk/benchmark.py --case smoke --out /tmp/indexer-chunk-human-review-unique
```

Use a fresh output path and an idle GPU. See [REPRODUCE.md](REPRODUCE.md) for build identity, patch order, model/PD scripts and full commands. Representative real-model validation receipts are in the report; they do not claim generated-token identity or standard GSM8K accuracy.

## Publication gate

Repository `AGENTS.md`, line 28, explicitly requires:

> The submitting human must review every changed line and run relevant tests.

Please confirm completion of both requirements after reviewing the final patch and running the relevant checks. This is the outstanding repository accountability requirement, not a request to reauthorize publishing. The final production-startup health and capture receipts are complete in the report.
