# Indexer allocation and graph ownership

Status: 96-case operator grid, V4.1 workspace probes, six real-model A/B/C
capture cases, default-budget stress and a DeepSelect configuration ablation
are complete.
These results use the historical `cff6224872` product baseline,
not latest upstream main or final PR #64138. The latter is assessed separately.

## Measured operator reproduction

`results/memory_operator_valid.jsonl` and
`results/memory_operator_valid_midrows.jsonl` have 96 cases: R=512/1024/2048/4096,
W=65536/262144, four backend/chunk variants, and eager/FULL/breakable capture.
Inputs use the physical production cache layout: each page stores quantized
values followed by unit UE8M0 scales. Q/K are MXFP4, H=64, D=128, k=512;
actual compressed history is 2048 columns. Host max_seq_len is W*4.

All 96 cases had zero nonfinite valid scores, valid nonduplicate indices,
and identical sorted selected score values between warmup and replay.
Bit-identical indices are not required for BF16 ties. This checks replay
preservation, not model accuracy or every possible selector shape.

The earlier `memory_operator_smoke.jsonl`, `memory_operator_v2.jsonl`, and
`memory_check.json` used uncontrolled random scale bytes rather than the
deliberate planar unit-scale construction. They remain diagnostic
allocation records only; their selection comparisons must not be used as
correctness evidence. The valid planar-input rerun supersedes them.
Those obsolete diagnostics are archived outside the repository under the
external results directory's `obsolete/` subdirectory.
This audit does not diagnose the legacy selector on that extreme scale
distribution. Passing representative inputs is not proof over all FP32 values.

R=4096, W=262144 (MiB, measured from CUDA allocator):

| Backend / chunking | Chunks | Live peak increment | FULL private-pool reserved | Breakable reserved increment after replay |
| --- | ---: | ---: | ---: | ---: |
| Legacy auto, unchunked | 1 | 6156.00 | 6156 | 12312 |
| BF16 DeepSelect, unchunked | 1 | 2052.00 | 2052 | 4104 |
| Legacy auto, 336-row chunks | 13 | 504.98 | 510 | 1018 |
| BF16 DeepSelect, 1024-row chunks | 4 | 514.00 | 516 | 1030 |

The live peak is capture max_memory_allocated minus the pre-capture live
allocation. It includes the transient source/destination overlap and small
operator metadata. Private-pool reservation is a different allocator metric.
The nominal 512-MiB score budget omits row padding, allocator rounding and
metadata; the DeepSelect chunk therefore measures slightly above 512 MiB.

For the tested W=64K,256K,1M shapes, the measured BF16 row stride is W+512.
At W=512K,H=32 it is W+1024. Padding must be read from the actual output.
`as_fp32_logits`
allocates a separate FP32 tensor with the same stride. Its storage covers
`(R-1)*stride + W` elements, omitting unused padding after the final row.
During conversion both allocations are active. Valid-row conversion reads
only row_end columns, although both buffers reserve capacity-width storage.
BF16->FP32 adds representation width; it does not recover BF16 rounding loss.

`memory_operator_summary.jsonl` records first-chunk tensor storage sizes,
the allocator block/pool owning each pointer, and snapshot SHA256 hashes.
The overlap snapshots show two independent allocations on the legacy path;
DeepSelect's producer and consumer pointers are the same BF16 tensor.

The additional `memory_operator_v41_dense.jsonl` validates the V4.1 candidate
source's dense shape R=8192,W=1048576,H=32,D=128,k=512 (operator reproduction,
not a V4.1 model run). All 12 cases have finite valid scores and preserved
selected values on replay. The unchunked legacy live peak is 49176 MiB,
DeepSelect 16392 MiB; 80-row legacy chunks peak 480.23 MiB (103 chunks), and
256-row DeepSelect chunks peak 512.25 MiB (32 chunks). These numbers demonstrate
why a sparse consumer's narrow candidate matrix cannot be substituted for
the dense candidate source when estimating V4.1 peak memory.
The matching pre-candidate C2 shape (W=524288,H=32,R=8192) is in
`memory_operator_v41_c2.jsonl`: actual stride 525312, unchunked legacy peak
24624 MiB, unchunked DeepSelect 8208 MiB, chunked DeepSelect 514.001 MiB.

V4.1 has additional workspace beyond the logits budget. Six actual
`select_candidate_blocks` producer probes at R=4096,W=1M,H=32 confirm that
a 256-row BF16 chunk peaks at **576.252 MiB**: about 512.25 MiB logits plus
64 MiB block scores (W/8 columns), valid-block counts and small metadata.
The unchunked producer peaks at 9220.016 MiB. Three dense-consumer fallback
`apply_candidate_mask` probes peak at 544.252 MiB with the same chunk: its
uint8 flags require about 32 MiB. These are actual operator reproductions
(`memory_candidate_source.jsonl`, `memory_candidate_mask.jsonl`), all finite
and replay-preserving; they are not a full V4.1 model run. The default narrow
sparse-consumer path is separate and must not be assigned this dense mask cost.
These figures are increments above preallocated inputs and outputs. The full
candidate-index and top-k-index buffers also remain live; they are not part of
the score-matrix budget or the transient increment reported here.

## What FULL and breakable capture change

The reproduction uses the actual `eager_break_during_capture` and
`BreakableCUDAGraphCapture`. In FULL, the indexer executes with stream capture
active and its transient logits reserve graph-private-pool blocks. After
capture, tensors can have zero active bytes while the pool retains those
blocks for replay. Releasing the graph and emptying the cache releases the
reservation in this controlled experiment.

In breakable capture the indexer reports capturing=False and has no
indexer allocation in the graph-private pool. It still allocates the BF16
and FP32 tensors at eager runtime. In this reproduction capture and replay
use different CUDA streams; caching allocator stream ownership creates a
second set of cached default-pool blocks after replay. This directly shows
why 'outside graph' does not mean 'zero memory', but it does **not** establish
the cause of two similarly sized segments in a full-model snapshot.

The real-model two-equal-FP32-segment claim was not reproduced in the
FULL_DECODE_ONLY or FULL_AND_PIECEWISE cases below.
Segment counts cannot establish logical tensor counts:
one allocation can occupy multiple blocks and multiple calls can reuse a block.
No formula here attempts to match total graph-pool reservation within 2%.

## Model boundaries and host parameters

Static MRV2 evidence: `vllm/v1/worker/gpu/model_states/default.py:195-199`
sets capture max_seq_len to the original uncompressed max_model_len. It is
passed to metadata construction at line225 and inherited by indexer metadata.
V4's compressed output capacity W=M/4 is therefore different from the host
max_seq_len used to specialize a captured legacy top-k. Actual dynamic hook
records are required to confirm this in each successful launch.
The metadata builder also executes GPU work before the model graph:
`vllm/v1/attention/backends/mla/indexer.py:1744-1751` computes/copies the paged
scheduler metadata. FULL replay does not eliminate that outer metadata build.
Chunk boundaries/backend choices in Python are fixed at capture, while the
captured GPU scheduling/scoring kernels read updated per-step tensor contents.
Exactly which scheduler kernels lie inside the model graph depends on whether
they are in this outer builder or the chunk loop.

PIECEWISE uses explicit eager-break decorators around model attention/indexer
regions on this branch. `splitting_ops=[]` alone does not specify whether an
indexer is captured. Compiled Inductor partition and breakable capture are
different mechanisms. See `SHAPES.md` for V4.1 source/consumer differences and
latest-main default changes.

## Real-model measurements

TP4 DeepSeek-V4-Flash-DSpark ran on Slurm40470, `gb200-rack1-10`, with
gpu_memory_utilization=0.85. All six cases completed bursts of 8,128,1024
requests with 48 output tokens each. This is allocation evidence, not a
timing or accuracy benchmark. API concurrency did not force an exact row count.

| Mode / M | Final private-pool reserved GiB | Private-pool allocated GiB | Indexer allocations in private pool |
| --- | ---: | ---: | --- |
| FULL_DECODE_ONLY / 262144 | 5.1855 | 0 | BF16 and FP32 |
| PIECEWISE / 262144 | 2.0879 | 0.3105 | none |
| FULL_DECODE_ONLY / 1048576 | 9.6856 | 0 | BF16 and FP32 |
| PIECEWISE / 1048576 | 2.0879 | 0.3105 | none |
| FULL_AND_PIECEWISE / 262144 | 6.2246 | 0.3105 | BF16 and FP32 in FULL captures |
| FULL_AND_PIECEWISE / 1048576 | 10.7246 | 0.3105 | BF16 and FP32 in FULL captures |

Rank0 recorded all 21 target indexers. FULL captures reached R=4096;
their host max_seq_len was the uncompressed M while capacity W=M/4.
R=64 selected cooperative top-k, larger captured shapes selected persistent.
The same BF16 pointer and same FP32 pointer were reused across all 21 layers
at the largest shape: these are transient buffers reused by many calls.
At M=1M the largest FP32 block is 4104 MiB (requested 4103.998 MiB), from
`as_fp32_logits`, and the BF16 block is 2052 MiB. Smaller graphs reuse these
blocks; there is one 4104-MiB private segment in both FULL-only and mixed-mode
1M experiments. Therefore the proposed explanation for **two** equally large
segments is not established by this dataset. Different graph grids, versions
and allocator histories would require a separate reproduction; this audit
does not reinterpret an older screenshot as two permanently live tensors.

PIECEWISE metadata/top-k/logits hooks consistently report capturing=False,
including during graph construction and runtime. Its dummy host max_seq_len
values are 1/4/8, then actual uncompressed lengths at runtime, rather than M.
Allocation-event stacks contain no indexer BF16 or FP32 allocation in a graph
private pool. Graph totals include other model allocations and cannot be
predicted by the score-matrix formula alone. Some graph blocks are reused for
both indexer and attention/vocabulary-logits allocations at different times;
the analyzer tracks segment creation epochs to avoid misattributing a reused
virtual address from an older, already released segment.

The model's log line named **"CUDA graph pool memory" is not the snapshot's
graph-private-pool total**. `GPUModelRunner.capture_model` measures global
free-device-memory change around capture, including default-pool allocations
cached by eager breaks. Measured reported/estimated figures:

| Mode / M | Capture free-memory delta, GiB | Profile estimate, GiB | Private pool, GiB |
| --- | ---: | ---: | ---: |
| FULL_DECODE_ONLY / 256K | 5.7813 | 6.1699 | 5.1855 |
| PIECEWISE / 256K | 7.0391 | 7.7188 | 2.0879 |
| FULL_DECODE_ONLY / 1M | 10.2813 | 10.6699 | 9.6855 |
| PIECEWISE / 1M | 16.0391 | 16.7188 | 2.0879 |
| FULL_AND_PIECEWISE / 256K | 7.1992 | 7.7891 | 6.2246 |
| FULL_AND_PIECEWISE / 1M | 11.6992 | 12.2891 | 10.7246 |

PIECEWISE graph profiling on this MRV2/grid executes paged decode at R=8192,
outside stream capture. Its full BF16+FP32 allocation contributes to the
default-pool cache and capture-memory estimate. **This configuration does not
reserve only the indexer's 512-MiB dummy budget.** The default .90 stress log
says its effective KV allocation equals utilization .8093 with graph profiling
disabled and gives 98.36 GiB of KV memory. A 512-MiB ordinary dummy reserve
and full MRV2 graph profiling are separate stages; older-main/grid OOM reports
cannot be applied to this case without reproducing its profiling behavior.

PIECEWISE runtime reached 5944 rows at M=256K and 5936 at M=1M. In the latter,
the BF16+FP32 storage simultaneously visible during conversion is about
8.71 GiB, exceeding the nominal 0.5-GiB ordinary indexer dummy budget. The allocator reported
one retry; all requests completed. This .85 run does not establish default-.90
OOM behavior. FULL's 48-token workload stayed within captured decode shapes,
so it is not a shape-matched large-eager comparison.

`results/memory_model_allocation.jsonl` stores per-pool totals, large-block
allocation history/call sites, capture events and hashes of snapshots and
immutable trace extracts. Full logs/snapshots are in the external results
directory. The per-case copied `hooks/sitecustomize.py` and `hooks_loaded`
events identify the exact imported hook; a source-hash discrepancy in the last
main launch record is explicitly corrected by its `hooks_provenance` event.

Three real BF16 logits samples come from layers 2/4/6 after a 10253-token
prompt, during separate PIECEWISE decode. Each is 8x2565 with per-row valid
ends, source full shape 8x65536 and stride 66048. All valid scores are finite;
values include negative scores and many BF16 ties. Paths, hashes and value
summaries are in `results/memory_real_logits.jsonl`. These are checkpoint
activations, unlike the synthetic operator inputs.
The exact logged host bound is 10261, not 2565*4=10260: visible raw lengths
10254..10261 are compressed to 2563,2563,2564,2564,2564,2564,2565,2565.
The indexer Q/K use MXFP4 with UE8M0 scales; the separate attention cache uses
`fp8_ds_mla`. These formats must not be conflated with the BF16 score dtype.

The initial TP2 attempt failed before weight loading because this checkout
lacked generated FlashAttention CuTe Python artifacts. Failed/aborted startup
is retained in JSONL and makes no model-memory claim. Real-logits copies are
an explicitly separate diagnostic workload and excluded from runtime memory
and performance inference. Shutdown may emit
EngineDeadError after a recorded complete event because the runner terminates
its owned server/worker process group; completed requests must be distinguished
from shutdown cleanup logs.
`results/memory_runner_sources.jsonl` archives both exact operator source
versions (matching every operator record's SHA256), current runners/analyzers
and the exact per-case hook versions used by successful model launches.

## Default-utilization runtime stress

Two PIECEWISE 1M runs at gpu_memory_utilization=.90 each completed all 1024
requests with 256 output tokens; neither OOMed. Both reported 11 allocator
retries. The corrected second run measured 95 first-layer indexer shapes,
distinguishing decode and prefill counts instead of only the padded hidden
shape (the first diagnostic hook had conflated those cases).

The largest observed mixed batch had 8072 decode and 120 prefill tokens,
8192 total. Its first indexer peak rose by 12,722,494,464 B
(12133.116 MiB); simultaneously live BF16+FP32 storage was 12,720,953,344 B
(12131.646 MiB). The 1,541,120 B difference is about 0.012%, small metadata
and allocator rounding. Before/after-op live allocation returned to the same
159,207,621,632 B: score tensors were transient. This validates the live
buffer formula at an actual model step, while the broader graph-profiling
reserve explains why this case survived despite exceeding 512 MiB.

This is a mixed batch, not an exact uniform 1024x8 model step. The exact
8192-row uniform shape is covered by native-operator probes; API concurrency
alone is not evidence of that model shape. No default-utilization OOM is
claimed. `memory_runtime_peaks.jsonl` contains the 95 per-op measurements,
matched dtype storage, trace hash and hook provenance;
`memory_model_stress_v2.jsonl` records requests and configuration.

## DeepSelect configuration ablation

A separate same-.85, same-PIECEWISE-1M launch explicitly selected
`kernel_config.sparse_indexer_topk_backend=deep_select`. It completed the same
8/128/1024-request workloads. Paged decode consumed BF16 directly; the ordinary
prefill path still calls `as_fp32_logits` and its prefill selector, independently
of this decode backend (`sparse_attn_indexer.py:727-772`). Therefore this is not
a claim that the entire model has no FP32 score tensors.

| Metric | Auto | Explicit DeepSelect |
| --- | ---: | ---: |
| Graph-private pool, GiB | 2.0879 | 2.0879 |
| Capture free-memory delta, GiB | 16.0391 | 8.0234 |
| Profile estimate, GiB | 16.7188 | 8.7031 |
| Available KV memory, GiB | 89.14 | 97.16 |
| KV token capacity | 10147974 | 11060540 |

The private pool is unchanged, while the profile/global-cache footprint falls
by about 8.02 GiB and KV allocation rises. This directly demonstrates that
reducing indexer temporaries can affect profiling/KV allocation even when
the indexer itself stays outside the CUDA graph. It is an explicit-backend
ablation, **not the final PR's auto/chunk policy** and not an accuracy/performance
comparison. `memory_backend_ablation.jsonl` contains log hashes and separates
paged decode upcasts from the still-present prefill conversions.
