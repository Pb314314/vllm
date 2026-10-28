# Next indexer optimizations: evidence and bounded experiments

Two additional ablations ran on the actual PR64138 checkout
`4e5b4f8ce9b067d92e7373d950727d8238e9a289`, its independent native library,
and GB200 node04/GPU2 with CPU70–85. Product code was not changed. The scripts
and JSONL are research artifacts; the examples below are not production
patches or whole-model throughput measurements. Main performance and memory
coverage is already complete in `PIPELINE.md`, `TOPK.md` and `MEMORY.md`.

The near-term choices are conditional: share schedules among compatible
layers; remove full-capacity mask work where the dense-mask fallback really
runs; then address repeated DeepGEMM launches or add explicit context graph
buckets. Increasing the chunk budget can improve local GPU time but spends
more scratch and changes the current backend-switch threshold too.

## Rebuild shared chunk schedules once per step

Question: can three V4.1 c2 layers share chunk schedules without freezing
sequence lengths at capture time? The existing `prebuilt-512` measurement
excluded all schedule updates and was only an upper-bound attribution. This
new experiment retains those updates inside every graph replay.

For R4096/W524288/H32/D128/MXFP4/topk512, the planner uses eight 512-row
chunks. The baseline calls the actual PR `_varlen_decode_rows` for three
separate layers, producing 24 metadata launches. The alternative builds
eight schedules at the beginning of each step and reuses them across three
layers' distinct Q/weight/K/output buffers. DeepGEMM and DeepSelect still run
24 times each. Only the schedule lifetime and number of updates change.

Both graphs were captured once at raw context32768, then replayed after
in-place sequence-length changes to8192 and1536. Each layer's valid logits,
exact selected-score multiset, bounds and uniqueness passed against the
single-call reference at every length. This is dynamic GPU work in a fixed
graph, not an offline choice based on future contexts.

| Raw context | Baseline warm | Shared warm | Saved per three-layer step | Shared cold / baseline cold |
| --- | ---: | ---: | ---: | ---: |
| 32768 | 1175.38 us | 1127.59 us | 47.79 us (4.1%) | 1129.38 / 1178.50 us |
| 8192 | 629.36 us | 574.85 us | 54.51 us (8.7%) | 584.19 / 636.39 us |
| 1536 | 417.74 us | 364.16 us | 53.58 us (12.8%) | 378.90 / 434.74 us |

Each point uses three randomized A/B × warm/cold trials of100 CUPTI replays;
maximum trial-median CV is below0.30%. At the 32K profile point, metadata
changes from24 calls/98.7 us to8 calls/32.5 us; MQA/top-k counts stay24.
The measured whole-span saving is smaller than the isolated kernel-time
difference, so the component sums are not substituted for the replay span.

Production design: own the schedule bank in attention/forward metadata,
keyed by compression, row partition, KV page size, SM/scheduler geometry and
query-to-request indices. Populate it once per step on the GPU before the
first compatible layer. Keep addresses stable through capture/replay and
retain buffers until the last compatible layer finishes. Rebuild after
request/length changes; do not reuse previous-step contents without updates.
The three c2 layers can share; the c1 candidate layer needs a different bank.
Layers with different compressed lengths, DCP metadata or page geometry must
not silently join the same group.

For the tested V4.1 layout, approximately48–55 us is the observed saving
over the three c2 indexers; the c1/candidate layer and all other model work
remain. It is not a 13% model-speed claim. If all21 V4 indexers share the
same compatible c4 schedule, historical per-layer metadata near30 us gives
an optimistic removable component around20×30 us=0.6 ms per step. That
cross-layer/model upper bound has not been measured and is not a promise.

## Avoid full-capacity mask writes on the end-aware path

The dense candidate-mask fallback currently clears flags for capacity-sized
block arrays and launches its mask over `(rows, ceil(capacity/1024))`.
For c1/R4096/W1M that is about4.2 million mask CTAs per layer even at8K
actual context. Historical kernel profiling attributes about4.97 ms to the
mask and0.218 ms to flags in the chunked BF16 path. Generated-code evidence
also shows scalar BF16 stores; reducing element bytes alone did not make
that path faster than the FP32 fallback.

The new ablation keeps the original flag allocation, initialization and
membership kernel. A separate benchmark-only Triton kernel visits each
row's **GPU-loaded valid end**, masks excluded candidates inside that prefix,
and leaves the unused tail untouched. Its fixed graph grid and GPU loop
remain valid when lengths change. The consumer is explicitly BF16
DeepSelect with an end tensor; no consumer is allowed to read the untouched
tail. This removes work rather than assuming that CUDA graphs erase GPU
launches or stores.

At R1024/W1M/H32/MXFP4/topk512, four256-row chunks, a deterministic pattern
keeps every fourth eight-token candidate block. Both causal bounds and
candidate membership matter. The graphs capture at32K and replay unchanged
after lengths shrink to8K. Exact selected values against a separately
masked single-call score matrix, index bounds and uniqueness all pass.

| Raw context | Original full-capacity mask pipeline | GPU-prefix mask pipeline, 32 programs/row | Reduction |
| --- | ---: | ---: | ---: |
| 32768 | 1528.15 us | 278.10 us | 81.8% |
| 8192 | 1423.71 us | 173.20 us | 87.8% |

These are whole-indexer warm GPU spans, not just mask kernels. Cold-cache
numbers are1529.62→279.25 us and1423.57→173.86 us. Three trials ×100 replays
were used; the largest CV is2.74% for the32K warm candidate. The input is
synthetic finite Q/K and periodic candidates, not checkpoint candidates.
A small programs-per-row sweep is recorded separately rather than claiming
32 is a tuned optimum.

| Fixed programs/row | Prefix pipeline,32K | Prefix pipeline,8K |
| --- | ---: | ---: |
| 8 | 279.26 us | 165.82 us |
| 32 | 278.10 us | 173.20 us |
| 128 | 316.45 us | 225.22 us |

Each graph keeps its program count fixed across both lengths. Eight and32
are broadly similar at32K, while eight improves the short case;128 adds
unnecessary work here. This is a coarse sweep, not tuning across all model
shapes. Selecting the best row independently for each context is an offline
comparison, not a new runtime dispatcher. The fixed eight-program graph
itself passes both contexts and avoids that oracle assumption.

Correctness scope is deliberately narrow: decode has `starts=None`, logits
are BF16, and the selected backend consumes only positions below row end.
The existing mask writes invalid tails to-inf for general consumers; simply
deleting those writes is unsafe for full-width `torch.topk` or another
consumer that observes the tail. Prefill packed starts, edge-clamped
out-of-width candidates, zero/<k lengths, NaNs, arbitrary candidate patterns,
all backend fallbacks and graph growth as well as shrink need dedicated
coverage before production integration. The unchanged flag initialization
still scales with capacity and still allocates its original buffer.
The full logits allocation also remains unchanged, so this ablation does
not claim an additional peak-memory reduction beyond the original chunking.

If four later layers actually use the dense-mask fallback, the tested local
component improvement is roughly4×1.25 ms=5 ms per step at R1024. This is
an optimistic indexer-component estimate, not measured model latency. With
sparse-logits enabled, those layers already score compact candidate entries
and do not run this fallback; the ablation then offers no saving there.
Prefer the established compact sparse path where its capability/semantics
apply. Restrict a new prefix-mask implementation to an explicit end-aware
consumer contract rather than broadening it through assumptions.

## Repeated persistent DeepGEMM calls are still real GPU work

The historical c1/R8192/8K launch trace keeps MQA's grid at152 CTAs with512
threads,128 registers/thread and227328 dynamic shared bytes. Row chunking
repeats that same persistent launch32 times; it does not shrink the grid to
one32nd. The isolated profile reports MQA about183 us for one call versus
431 us across32 calls. DeepSelect also gains repeated launches and smaller
per-call grids. Removing CPU dispatch with FULL capture cannot remove this
GPU overhead.

On final-native c1/R8192/8K, single DeepSelect is321 us versus788 us for
the PR path, leaving approximately467 us of local headroom. The single call
requires about16 GiB of BF16 scores, so that is a memory-spending lower-cost
reference, not a solution within512 MiB. Sharing schedules alone cannot
recover that gap. The budget sweep supports larger row chunks when memory
permits; separate the backend-switch threshold from the chunk-size budget
if exploring that policy, and validate global KV reservation/model impact.

A deeper kernel direction is a bounded-scratch producer/selector pipeline
that avoids restarting all persistent scoring state for every Q chunk.
Fusion or a work-queue design must wait until each score's complete head
reduction has finished before top-k, preserve the current BF16 arithmetic,
and provide correct barriers between buffer reuse phases. Streaming over K
tiles additionally requires exact merging of tile top-k results; selecting
partial head sums is invalid. Double buffering spends additional scratch.
No such kernel was implemented or timed here; the467 us difference is not
an expected fusion gain.

## Context graph buckets need an explicit dispatcher

The current FULL graph fixes its backend, row chunks, capacity/stride and
buffer addresses at capture. GPU `seq_lens` values may change at replay;
Python cannot silently choose a new backend or a narrower logits capacity
inside that existing graph. The new schedule and mask experiments use
GPU data-dependent work within fixed graphs and do not require an oracle.

A different design could key graphs by `(query bucket, context upper bucket,
compression, precision, backend/chunk policy)`. The scheduler already knows
the current host context maximum, chooses a safe upper bucket externally,
and stages the same input tensors before replay. A narrower score capacity
must still cover every valid compressed K index. Growth crosses buckets;
shrinking may reuse the larger safe graph. This needs a graph-key/runtime
metadata change and tests, not just a benchmark environment variable.

Capacity changes affect DeepGEMM output padding, top-k specialization,
candidate-column clamp semantics, capture count and retained graph-pool
memory. Candidate IDs and full-model cache capacity remain distinct from
the transient scoring width. Invalid padding and packed-start callers need
careful treatment. More context buckets multiply capture time and graph
storage; model-level memory accounting must precede a default policy.
Offline `actual-context` or reduced-capacity results are counterfactual
evidence, not a working FULL dispatcher. The top-k address-overflow defect
must also be fixed independently before selecting the affected old backend.

## Reproduction and boundaries

`optimization_schedule.py` and `optimization_mask.py` use the final-native
environment in `ENV_pipeline.md`. Run from `wt-b4a0fix` with its Python,
`PYTHONPATH` pointing there, GPU2 selected, and `OPTIMIZATION_RESULTS` pointing
to a new JSONL. Mask tuning uses `MASK_PROGRAMS`; schedule updates remain
inside replay in every arm. Both scripts preserve source hashes, graph
checks and per-trial data. A first schedule setup attempted unsupported
nested graph launch in the profiling helper; it was rejected before timing,
recorded in `optimization_setup_failure.jsonl`, and corrected by capturing
the function directly. It is a benchmark setup error, not a product failure.

The execution environment was
`/home/bopang/cg-project/venv-b4a0fix/bin/python`, with
`PYTHONPATH=/home/bopang/cg-project/wt-b4a0fix`; scripts reside in this report
directory. `OPTIMIZATION_RESULTS` must name a fresh output JSONL because
these small ablations append rather than resume. Schedule and mask profiles,
36 schedule trials,72 mask trials and their summaries are preserved in
`results/optimization*.jsonl`. The audit records file hashes and zero failed
correctness/noise checks. An accepted trial is not an independent model run.

All added GPU processes exited by16:31:23 UTC. Coarse before/after clock
observations are retained where available; clocks were not locked. CUDA
graphs and three randomized trials control the comparison, but these short
experiments do not replace a longer stock-power/full-model evaluation.

No production source, PR state or publication was changed. These ablations
establish kernel-level opportunities. A follow-up should implement one
bounded change, add the missing behavioral coverage, and run complete
model decode/accuracy/memory checks before claiming a serving improvement.
