# Vision encoder CUDA graph research

This branch is a work-in-progress snapshot for collaboration, based on
`596e53067548fab65de6fc3abdcd020f752e7ae7`. It combines experimental encoder
breakable capture, whole-batch graph/eager dispatch, and benchmark tooling.
It is not a claim of production readiness or universally improved defaults.
AI assistance was used for implementation, experiments, and analysis.

## Current implementation

- `EncoderCudaGraphManager` supports whole-batch dispatch: at most one replay
  per active encoder path, or whole-batch eager execution. The explicit `split`
  policy retains greedy multi-replay behavior. This branch defaults to `eager`.
- DeepSeek-V4.1's encoder eager fallback runs a packed batch. DeepSeek and
  MiniMax avoid recopying an unchanged whole-batch selection.
- Optional token-padding thresholds are experimental tuning controls, not
  automatic calibration. Output-token padding does not bound patch padding.
- The breakable prototype supplies an encoder-specific attention boundary and
  replay metadata. Its initial model integration is Qwen3-VL; it is not enabled
  for every vision encoder/backend.
- FULL input/output storage reuse is opt-in through the model capability flag.

Tighter patch-capacity graphs and automatic latency calibration are **not**
general production features in this snapshot. The exact-shape DeepSeek graphs
below are diagnostic controls; their coverage is limited to the tested shapes.

## Results and limitations

Publication checks: all 105 tests in
`tests/v1/cudagraph/test_encoder_cudagraph.py` passed on GB200 (job 43589).
The 14 tests affected by subsequent typing-only cleanup also passed separately.
Repository pre-commit checks passed, including Ruff, mypy, and documentation
checks. This does not replace the serving validation listed below.

See the [design and experiment report](../../docs/design/vision_encoder_piecewise_proposal.md)
for earlier experiments, correctness limitations, memory tradeoffs, and the
DeepSeek regression attribution. The report is chronological; its final section
contains the latest two-case attribution.

The checked-in [dispatch validation table](results/dispatch_validation.csv)
contains 35 held-out shapes, ten targeted boundary checks, and two repeated
controls on real DeepSeek-V4.1-Flash/MiniMax-M3.1-Flash vision weights. All 47
candidate encoder-output checks were bitwise equal to packed eager. These
results use GB200, BF16, FA4, and TP1 and do not establish task-level accuracy,
serving TTFT, or a universal absence of regressions.

Two further independent runs isolated DeepSeek's 2636/2794-token regressions:
extra patch-row capacity costs about 5.7/6.2 ms, unused attention sequence slots
about 0.75 ms, and tighter `max_seqlen` alone has no material benefit. All 30
arm/case output comparisons were bitwise equal to packed eager. The
[latency table](results/deepseek_attribution_latencies.csv) and
[summary with source hashes](results/deepseek_attribution_summary.json) preserve
the measured scopes and controls. Profiler traces and full logs remain on the
experiment machine, not in Git.

Important open work:

- Reduce actual patch capacity without losing legal-shape coverage. Matching
  solely by output tokens is insufficient for efficient DeepSeek capture.
- Validate full-serving temporary-memory headroom: packed eager may use more
  runtime workspace even when persistent graph memory falls.
- Preserve MiniMax's existing NaN protection. Injected-tail stress tests are
  not evidence that an ordinary production request spontaneously produces NaN.
- Validate additional backends, models, and parallelism before upstreaming.
  The snapshot also includes opt-in buffer sharing beyond the source state
  used by the earlier dispatch experiments.

## Reproduce the latest DeepSeek attribution

Use a vLLM environment with CUDA support and the pinned checkpoint
`deepseek-ai/DeepSeek-V4.1-Flash` at revision
`2cba9e42aa026125f3ed06c6d98c1db82f7ca027`. The benchmark loads only vision,
aligner, and image-marker weights; it does not instantiate the language model
or allocate a KV cache. `MODEL_PATH` must contain the config and indexed shards
holding those tensors.

```bash
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0

.venv/bin/python benchmarks/multimodal/benchmark_deepseek_encoder_attribution.py \
  --model-path "$MODEL_PATH" \
  --output /tmp/deepseek-encoder-attribution.jsonl \
  --blocks 5 --repeats 8 --profile
```

Inputs are deterministic synthetic images processed by the model's real image
processor. All arms check output correctness before timing. `encoder_complete`
includes preparation, staging, forward, and output assembly; `forward_prepared`
excludes preparation/staging/output assembly. CUDA graph output copies and
existing correctness guards remain included. Profiled kernel sums are separate
from the unprofiled latency measurements.

For broader sweeps, use `benchmark_encoder_crossmodel.py --help`. Explicitly
select the checkpoint, workload, grid, and policy; the historical tuning recipes
are hardware-specific examples. For encoder manager regression tests:

```bash
.venv/bin/python -m pytest tests/v1/cudagraph/test_encoder_cudagraph.py -q
```
