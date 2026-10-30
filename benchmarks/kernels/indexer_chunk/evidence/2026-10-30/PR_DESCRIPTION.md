<!-- markdownlint-disable -->

## Overview

Bound SM100 MXFP4 decode indexer logits by processing independent query rows in chunks. Add a configurable 2048 MiB BF16 payload budget, preserving the existing top-k providers and capturing all chunks in one FULL graph replay.

This depends on [persistent BF16 input](https://github.com/vllm-project/vllm/pull/64138) and [cooperative BF16 input](https://github.com/vllm-project/vllm/pull/62172). It complements [SM90/SM120 uniform-decode request chunking](https://github.com/vllm-project/vllm/pull/62169), which excludes SM100 varlen; it does not duplicate that path or prefill chunking. The benchmark's separately identified row-address overflow correction must also be integrated and reviewed.

## Claims

- Reduce temporary dense/source decode logits from all query rows to a bounded row chunk; batches within budget retain the existing path. `VLLM_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB=0` disables chunking; `4096` trades more memory for fewer chunks.
- Keep the existing BF16 cooperative/persistent selection logic. Valid producer and candidate scores match bitwise in the tested cases; exact top-k values, unique valid positions and sentinel padding pass. Tied index identity/order is not guaranteed.
- On tested GB200 V4.1 TP4 model-forward workloads, 2 GiB saved 2.5–17.5 GiB of peak allocated memory per GPU with approximately 0.05–0.30% added graph-forward time. This is a memory optimization with a latency tradeoff, not a speedup claim.

## Validation

Model-forward comparison uses actual V4.1 weights, synthetic tokens and zero historical KV, fixed KV allocation and production-capacity indexer graph metadata. Two reverse visits, 150 samples per visit; max-rank latency. These are model-forward measurements, not generation throughput.

| Requests × Q/request, history | No chunk → 2 GiB graph ms | Change | Peak saved/GPU |
|---|---:|---:|---:|
|256 × 8, 8K|40.156 → 40.175|+0.05%|2.50 GiB|
|512 × 8, 8K|71.544 → 71.689|+0.20%|7.50 GiB|
|1024 × 8, 4K|118.817 → 119.070|+0.21%|17.51 GiB|
|320 × 8, mixed 4K/8K/16K|46.635 → 46.774|+0.30%|4.14 GiB|

A controlled eager-only follow-up prepares the batch once and interleaves budgets with unchunked baselines. At 1024 requests × 8 rows / 4K history, 2 GiB costs +3.34% to +3.73% versus +1.38% to +2.16% at 4 GiB. FULL graph remains faster than eager. The 512-request comparison has more baseline drift; previous mixed-protocol eager regressions, including an apparent roughly 25%, are retained rather than treated as reliable precise estimates. Small-budget operator regressions are also retained in the full report.

- 451 existing FP32/BF16 native tests passed on the common corrected build.
- Nine focused final chunk-planning/profiling tests passed.
- Real DeepGEMM sweeps include 512 MiB, 1/1.5/2/2.5/3/4/6/8 GiB, mixed histories, nonaligned tails, empty rows, changed weights and same-graph length updates. Correctness gates run before timing.
- Real-text V4 collocated A/B/A: 32/32 final answers in every arm; all four B workers witnessed chunk graph replay. Eight unique arithmetic prompts with long neutral prefixes repeated four times; this is not a standard GSM8K evaluation. B deliberately uses 8 MiB to force the path. Generated sequences differ in both A/B and A/A; no token-identity claim.
- V4.1 collocated A/B/A: 64/64 final answers per arm at forced 32 MiB; V4 two-node NIXL PD A/B/A: 32/32 per arm at forced 8 MiB. All four B workers witnessed actual chunk graph replay in both tests. NIXL transferred 7,546,079,232 bytes per PD arm. Same limited repeated-question scope and no token-identity or serving-throughput claim.
- Existing candidate-selection fallback without DeepSelect passed score/top-k and graph replay checks after repairing an observer-only omission.

Actual production startup with max model1M, max sequences1024, DSpark7 and FULL_DECODE_ONLY: the unchunked arm failed with negative KV headroom; 2 GiB completed health checks. The same profiling-capture phase used33.68 versus11.66 GiB/worker; final B capture used10.00 GiB. KV headroom includes a profiling-reservation correction and must not all be called actual memory savings. This is startup evidence, not throughput.

Core reproduction commands (matching combined BF16 build required):

```bash
.venv/bin/python -m pytest tests/kernels/test_top_k_per_row.py -k decode_chunk -v
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/kernels/indexer_chunk/benchmark.py --case smoke --out /tmp/indexer-chunk-smoke
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/kernels/indexer_chunk/benchmark.py --case c1_source --out /tmp/indexer-chunk-c1-source
```

Full protocol, executed scripts, raw numeric samples/proofs and plots are provided in [the separate evidence package](https://github.com/Pb314314/vllm/tree/cg/indexer-decode-chunk-evidence-v1/benchmarks/kernels/indexer_chunk/evidence/2026-10-30). The large-row baseline uses the same two-expression `size_t` offset correction in both arms; without it an existing uint32 overflow makes that baseline invalid.

## Details

For R query rows and compressed layer capacity W, BF16 payload is `2*R*W` bytes. Derive aligned chunk rows from the byte budget, balance the ranges, and allocate the largest first so graph capture can reuse scratch. Each query still scores its whole effective history; there is no inter-chunk top-k merge. Candidate score buffers, padding and other scratch are additional to the payload budget.

A 2 GiB default permits 1024/2048/4096 query rows at compression 1/2/4 for a 1M model capacity. Per-layer compression already changes the row limit without separate budget knobs. The selected default is a measured memory/latency compromise; 4 GiB lowers repeated scheduling/launch overhead at higher memory cost. The planner does not switch to a new selector or guarantee that DeepGEMM becomes compute-bound.

The initial scope is SM100 MXFP4, DCP=1, no PCP, dense/candidate-source decode, auto/cooperative/persistent and k=512/1024/2048. Compact sparse consumers, FP8 caches and other devices retain their current path. All chunks are inside one FULL graph; eager retains Python dispatch overhead. A future fused decode scorer/selector could avoid global logits materialization, but would be a separate kernel change with separate correctness and performance evidence.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [ ] **Testing and Validation:** Automated functional/performance receipts are complete; human testing confirmation pending.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [ ] **Pull Request Contents:** Final evidence links, dependency integration and human review pending.
</details>

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>**
