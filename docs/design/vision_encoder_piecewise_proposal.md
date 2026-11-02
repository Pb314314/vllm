# Vision encoder piecewise graphs: research and proposed design

Original design, 2026-10-29; subsequent implementation and experiment updates
are recorded below. Source initially inspected at
`596e53067548fab65de6fc3abdcd020f752e7ae7`.

Upstream main was additionally checked at
`4b47db922b833b83f8f1e067a9f5ac9fed16c769` before specifying the implementation
route. Its break decorator no longer has the older import-time environment
guard, but retains the decoder FULL-context guard, fixed-output contract and
weak-reference recording. Do not confuse this with PR #63406's pinned head.

## Decision

Prototype **encoder-owned piecewise execution using the existing breakable
capture primitive**. Retain full encoder graphs and whole-batch eager as
baselines. Evaluate compiler-based PW separately, sharing the attention boundary
contract and encoder planning logic. Do not change the default based on the
current evidence.

The strongest motivation is compatibility: some encoder attention paths require
runtime host work that cannot be recorded into a reusable full graph. Performance
is a hypothesis, not a consequence of splitting. Moving attention outside the
graph does not eliminate padded QKV/MLP computation.

PW describes segmented execution. Breakable and compiler-based splitting are two
ways to implement it; they should not require two independent encoder managers.

## Evidence and prior work

- `EncoderCudaGraphManager` captures `encoder_cudagraph_forward` with
  `torch.cuda.graph`; the token budgets, path selection, capture axes, packing,
  input copying, output scattering and encoder DP handling already exist.
- Qwen3-VL patch embedding and vision blocks have `support_torch_compile`
  decorators with `is_encoder=True`. Compilation is not equivalent to graph
  replay: `CUDAGraphWrapper.__call__` runs the underlying callable when no
  `forward_context` exists, explicitly citing a vision encoder as an example.
- Compiler PW wrappers select their mode from the common compilation config and
  use decoder `BatchDescriptor`/forward context. The default splitting-op list
  does not include ViT attention wrappers. Encoder compilation ranges have an
  encoder exception, but this does not establish encoder PW graph ownership.
- `BreakableCUDAGraphCapture` already supports graph/eager/graph sequences. Its
  `add_eager` replays callables for side effects: downstream captured consumers
  must read stable output addresses. The higher-level wrapper is coupled to
  decoder context; the lower-level primitive can be reused without that wrapper.
- The inspected local/PR-era decorator reads the global enable flag when
  decorated; upstream main has removed that guard. Both versions suppress a
  break in decoder FULL context. Encoder execution policy must be independent
  of these decoder choices.
- Qwen varlen SDPA performs a GPU length `.tolist()` and Python `torch.split`.
  An eager boundary alone permits this code to execute but does not make its
  synchronization efficient. Real host lengths should be prepared once per
  encoder batch and reused across layers.
- Dynamic FP8 scale updates mutate a Python ring index. A break below that
  update is insufficient; quantization and state changes would need to be part
  of the eager boundary. Initial scope is BF16, not dynamic FP8.

Primary references:

- [vLLM full ViT graph PR #35963](https://github.com/vllm-project/vllm/pull/35963)
  includes FlashInfer/cuDNN support and its benchmark. Full encoder graphs are
  not exclusive to FA3/FA4.
- [SGLang encoder graph design](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/cuda_graph_for_multi_modal_encoder.mdx)
  describes a separate ViT graph runner. Its example combining ViT graphs with
  prefill breakable graphs refers to **decoder prefill**, not proof that the ViT
  itself uses PW.
- [SGLang breakable design](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/breakable_cuda_graph.mdx)
  provides a precedent for selective eager boundaries and documents their
  additional replay/Python overhead. It is not a ViT performance result.

## Why prefill is a useful but incomplete analogy

Both execute transformer blocks over many tokens and encounter changing batch
shapes. Capturing non-attention operators can save launch overhead while leaving
attention planning flexible. Unlike typical LLM prefill, Qwen ViT attention has
no paged KV cache, uses noncausal per-image/per-frame sequences, and runs dense
blocks rather than the MoE/indexer workloads motivating some decoder gains.
Large-image necessary GPU work can dominate both eager and graph execution.

Qwen's graph budget is in output visual tokens; attention operates on patch
tokens. With 2x2 spatial merging, there are four patch tokens per visual token.
This conversion is model-specific and must not be built into the shared manager.

Current FULL already uses real `cu_seqlens`, so it does not attend to other
images or unused tail rows. The potential benefit of eager attention is runtime
planning/launch bounds and backend compatibility, not removal of valid attention
work. Captured projections and MLPs still operate on bucket-sized buffers.

## Execution design

```text
effective encoder config + model shape constraints
                     |
existing encoder manager: select items/path/axis/budget
                     |
prepare metadata once; copy static GPU inputs; set scoped replay metadata
                     |
     +---------------+----------------+
     |               |                |
    FULL             PW          whole-batch eager
     |               |
one encoder     graph: norm/QKV/RoPE
graph replay    eager: real-shape attention -> fixed output buffer
                graph: projection/residual/MLP/next-layer QKV
                ...
                     |
existing per-item output assembly/cache + LM prefill
```

Whole-batch eager must be selected before graph-cap-driven splitting if the goal
is to compare it against several small replays. Initially this is an explicit
experimental arm, not a new automatic cost model. Same-packing eager is a
separate control for measuring graph overhead.

### Shared attention boundary contract

Introduce one encoder attention operation with a caller-owned output, conceptually
`encoder_attention_with_output(q, k, v, output, layer_key)`. This is a proposed
contract, not an existing API. It must have an accurate mutation schema and fake
implementation for compiler splitting.

The boundary must:

1. Obtain real lengths, active patch rows, item/frame counts, offsets and actual
   max sequence length from the current encoder replay context. Never capture a
   Python scalar/list from the warmup and reuse it as runtime metadata.
2. Keep host lengths on the host and GPU metadata in stable GPU buffers. Reuse
   already available input metadata once per batch; do not add per-layer GPU
   `.item()`/`.tolist()` synchronization. Different window/full attention layers
   may need distinct metadata within that batch.
3. Slice/use real rows inside the eager operation when required, while writing
   results to the same bucket-sized output allocation consumed by later graphs.
   Preserve layout/strides; do not assume every backend accepts a supplied view.
4. Use a backend's `out` API when supported. Otherwise compute a temporary output
   and copy into the stable buffer; measure this cost. Clear unused output rows
   deterministically so stale values/NaNs cannot contaminate later operations.
5. Preserve per-image/per-frame boundaries, RoPE, dtype, quantization semantics
   and backend-specific contiguous requirements. No cross-image attention.

For BF16 the first boundary can surround the attention core. For future dynamic
FP8 it must include the relevant scale updates and quantization, with reductions
over the same valid rows as the eager reference. GPU-data-dependent pruning is
not automatically solved by this design and is outside the first prototype.

### Breakable route: first prototype

- Keep the existing encoder manager, packing and output ownership.
- Allow a capture entry to hold a monolithic graph or a segmented capture with
  an explicit replay contract. Avoid a second encoder runner.
- Use `BreakableCUDAGraphCapture` inside encoder capture and intercept the shared
  attention boundary. Scope activation to the encoder; do not depend on the
  decoder's FULL/PW mode or a globally applied attention decorator.
- Allocate boundary outputs before ending their producer capture segment, so
  graph-pool storage remains valid through the capture entry's lifetime. Record
  transient boundary arguments through the same weak-reference mechanism as
  main; do not retain an extra strong-reference list of every layer's activation
  for every budget. Explicit persistent input/output buffers remain entry-owned.
  Validate allocator reuse across eager boundaries and budget transitions.
- Disable compiler-managed nested graph capture in this first experiment.
  Compilation can later be combined with breakable execution, but keep it fixed
  when comparing graph mechanisms.
- Preserve pool/stream handling and output cloning before another replay can
  overwrite the storage. A shared pool does not itself prove safe lifetime reuse.

### Compiler PW route: separate implementation of the same boundary

- Reuse Qwen's existing compiled patch/block entry points and shared FX splitting.
- Register the encoder output-writing attention op as a splitting boundary.
- Give encoder compiled segments an encoder-owned capture key/context and
  encoder policy. Do not merely set a fabricated decoder batch size or mutate
  the shared decoder compilation config.
- Ensure initialization and cached/AOT reconstruction both preserve the same
  split/graph policy. No outer full capture around eager attention boundaries.
- Existing per-block compilation can create roughly two captured pieces per
  block. A continuous whole-forward segmented capture with one attention break
  per layer can instead have roughly L+1 nonempty segments. Actual counts must
  be observed; fusion, empty pieces and extra operators change them.

The compiler route may benefit from fusion. Compare compiled eager against
compiled PW as well as uncompiled eager against breakable; otherwise a reported
PW benefit may actually be a compilation benefit.

## Configuration and compatibility

Retain existing `encoder_cudagraph_token_budgets`, item/frame capacities and
per-model paths/axes. Do not reuse decoder prefill/decode/SD/AV sizes.

For the prototype, select the capture implementation internally in the test
harness. If evidence warrants a public API, add an encoder-specific mode with
NONE/FULL/PIECEWISE semantics; keep the existing Boolean's behavior when the mode
is unspecified and validate contradictions. Naming and precedence are proposals,
not implemented flags. Capture implementation is a separate concern from policy;
avoid exposing two competing graph managers or flags before needed.

Default FULL remains available for known-compatible configurations. PW is opt-in
until measured. An unsupported FULL backend can use PW only after the full eager
boundary/output contract has been validated; otherwise eager is the safe path.
Do not recover from arbitrary CUDA capture errors as if they were clean fallback.

Backend compatibility is finer than a GPU vendor: FlashAttention/FlashInfer/
Triton/AITER/SDPA each have version, shape, dtype and model constraints. PW can
relax attention capture constraints but cannot make an unsupported kernel,
unstable collective, or data-dependent output shape automatically supported.

## Implementation route before changing runtime code

### Stage 1: encoder context and one attention boundary

Introduce a scoped encoder replay context carrying the capture mode and metadata
per attention layout. For the Qwen BF16 image prototype it includes actual patch
rows, item count, host sequence lengths/max length and GPU sequence offsets.
The model adapter supplies these facts; the shared manager never assumes a 4:1
patch/output-token ratio. Clear the context on normal and exceptional exit.
An eager callback reads the current context during every replay, not a mutable
Python argument accidentally frozen into its capture-time closure.

Register one output-writing encoder attention custom op and route the selected
encoder PW path through it. Leave normal eager and FULL dispatch intact. The op
trims attention inputs to valid rows and metadata to active sequences. It writes
the valid result into the fixed padded output and initializes the unused tail.
First prove that contract on FA4 and varlen SDPA. Keep any temporary-output copy
visible in timing; support direct backend output writing only where the layout
contract permits it.

Reuse main's weak-argument/eager-call recording through a small shared helper
extracted from the existing decorator if needed. Keep the decoder decorator's
existing guards and behavior. The encoder caller applies its own scoped policy
before using that helper, so a decoder FULL context or environment default does
not select encoder behavior. Do not duplicate the graph-segment engine.

Deliverable: varying-shape attention calls write correct results at stable
addresses, and decoder breakable primitive tests still pass.

### Stage 2: segmented entries in the existing encoder manager

Keep capture keys `(path, token_budget, axis_keys)` for a manager configured to
one execution mode. Extend capture metadata to hold either a monolithic graph
or `BreakableCUDAGraphCapture`, plus observable graph-segment/break counts. Do not
capture FULL and PW simultaneously in the same manager for the first prototype.

Startup sequence: allocate static inputs, prepare host/GPU metadata, warm the
eager attention path, rebuild any metadata mutated during warmup, enter the
manager's capture stream/pool, capture the encoder with attention breaks, copy
final embeddings into the existing persistent output, then release transient
strong references. Prewarm measurement shapes before steady-state timing;
unseen eager attention specializations may still cause first-use JIT costs and
must not silently contaminate timings.

Runtime sequence: retain the current packing, prepare real metadata once, update
static inputs, install the scoped encoder context, replay the segment list in
order, and clone/scatter valid outputs before the next replay overwrites them.
Respect the existing stream dependencies and pool lifetime; do not add routine
synchronization per attention layer. No nested monolithic/segmented capture.

First enable this through an explicit internal experimental mode used by the
test/benchmark harness. Existing public settings and FULL defaults stay intact.
A public encoder mode is a later, separately reviewed interface change after the
prototype passes; it must not toggle global decoder compilation or breakable
environment variables. Initial BF16 Qwen support is explicit, not a promise for
every model implementing the existing full-encoder protocol.

Deliverable: actual encoder segmented capture/replay, graph-count visibility and
stable memory across repeated inputs. Both decoder NONE and FULL settings work.

### Stage 3: correctness, attribution and serving acceptance

Run the correctness and controlled comparison matrix below before changing
packing, token grids, automatic caps or adding cost-based dispatch. Separate
kernel profiling from uninstrumented timing. Retest image TTFT/text stalls with
the production decoder mode after the encoder-only attribution is understood.

Only after these gates pass consider direct-output optimization, additional
backends/models and a public opt-in. Traditional compiler PW is a subsequent
comparison using the same boundary/context contract, not part of this first
breakable implementation. GPU-dependent pruning, dynamic FP8 state, multi-path
encoders and TP/DP collective capture each require their own validation.

## Necessary tests and measurements

First scope: Qwen3-VL-4B, BF16, TP1 images. FA4 for comparison with existing data;
varlen SDPA for the compatibility case; Triton if available. Actual AMD/AITER
testing and one second model (Gemma4 with its additional shape axes) are required
before cross-platform/general-model claims. Later stages cover video, encoder
TP/DP, OCR multi-path and quantization independently.

The module's contract is replaying different valid encoder inputs into correct
per-item embeddings with bounded graph buffers. The main failures are stale
metadata, address changes, padding leakage and repeated retained allocations.
Extend existing encoder/breakable tests for these behaviors; use compiler tests
for mutation/split semantics and E2E tests only for output/serving integration.

### Correctness

- Change item count and lengths within the same bucket across replay calls; use
  both large-to-small and small-to-large transitions and different compositions
  with the same total tokens. Compare every valid item's embeddings to eager.
- Test zero-length unused slots, exact boundaries, nonuniform images, cap overflow
  fallback, output ordering and repeated calls with poisoned unused buffers.
- Check tensor tolerances, finite outputs and greedy generation on representative
  inputs; use model evaluation before general quality claims. Do not assume that
  previous FULL correctness results cover new PW execution.
- Assert eager attention executes once per layer per replay with current metadata
  and that adjacent graph segments really replay; no accidental all-eager pass.
- Test encoder mode with decoder NONE and FULL to establish independence.
- Run the changed path with `VLLM_GPU_SYNC_CHECK=error`; distinguish existing
  manager-level metadata reads from new per-layer synchronization.

### Small performance matrix

1. Single-image visual sizes 256, 1024, 9216 and 16384. Mixed two-image batch
   1024+512. Sixteen images of 1024 each for batching effects.
2. For attention-boundary comparisons hold packing, budget, item capacity and
   compilation constant: same-packing eager, exact-size FULL where safe,
   existing padded FULL, and breakable PW. Include exact-shape compiled eager
   and compiler PW as a second comparison when that implementation is ready.
3. For 16x1024 separately compare one whole-batch eager forward, four 4096 FULL
   replays and one 16384 FULL replay; add PW without changing multiple variables.
4. Same GPU/node, warmed kernels, balanced arm order, at least three independent
   repeats for promising points. Profile separately from uninstrumented timing.
5. Fix KV capacity and LM policy. Measure encoder GPU stream span, CPU preparation,
   attention versus non-attention kernels, graph launches/breaks, copy/zero time,
   padding and execution mode. Then confirm TTFT and mixed-text emission stalls
   with production decoder settings, not only decoder graphs disabled.

### Memory and economic decision

Record capture duration, budget count, actual CUDA graph objects/segments,
device-free delta, allocated/reserved graph-pool memory, persistent I/O and
boundary buffers, eager temporary peak and steady-state growth. These overlap:
do not add pool, tensors and total-device figures as independent costs.

With B budgets and J attention breaks, a continuous segmented path can have about
B*(J+1) graph objects instead of B, before path/axis factors. This does not imply
the same factor in GPU memory: pool reuse and buffer lifetimes decide memory.
Conversely, excluding attention does not guarantee lower memory when its eager
workspace and stable bridge outputs are considered.

At equal bucket size, the expected timing tradeoff is:

    FULL: padded dense work + captured attention + one replay + preparation
    PW:   padded dense work + eager attention + segment replays + bridge work
    eager: actual dense work + eager attention + exposed host launch gaps

PW wins over FULL only if better attention execution/compatibility outweighs
extra boundaries; it wins over eager only if saved host gaps outweigh padding,
replays and bridge copies. These are latency components, not permission to sum
overlapped CPU and GPU times blindly.

Keep PW as a compatibility option if it enables a relevant previously all-eager
backend with acceptable cost. Promote it as a performance option only with
repeatable E2E gains and acceptable memory. If full-compatible large-image cases
do not improve, retain eager/FULL rather than forcing PW. No automatic
cost-based dispatch or new default grid is justified by the existing results.

## Existing data and remaining uncertainty

Earlier Qwen GB200 serving results (decoder graphs disabled, different nodes per
arm, six samples per image size) found 9216-token TTFT 414.2 ms with old oversize
eager, 434.9 ms with a 16384 padded graph, and 415.4 ms with a denser graph grid.
At 16384 tokens the corresponding TTFTs were 922.0, 922.7 and 917.5 ms. These
results motivate better selection; they are not a PW experiment.

Local source data:
`/home/bopang/cg-project/results/vision_ttft_20261029/SUMMARY.json` and
`REPORT_ZH.md`. Earlier encoder-only graph memory data lives in
`/home/bopang/cg-project/results/vision_budget_20261029/`.

At proposal time, unanswered questions included attention/non-attention shares;
benefit of reducing per-sequence launch bounds; bridge-copy cost; PW memory;
cross-backend correctness; and whether compiler PW adds value over compiled
eager or compiled breakable. Subsequent prototype findings are recorded below.
Duplicate-work and human-review requirements apply before submission.

## Prototype results, October 29, 2026

The local prototype implements the independent encoder context, output-writing
attention boundary, shared weak-reference recording helper and segmented entries
in the existing manager. It is an internal opt-in, limited to BF16 TP1 images,
with encoder compilation disabled. Public arguments and defaults are unchanged.

On GB200 with Qwen3-VL-4B-Instruct, all eight tested input layouts produced
bitwise-equal embeddings against each backend's eager reference, for both FA and
SDPA. The encoder and shared breakable suites pass all 91 tests. These checks do
not replace a multimodal task-quality evaluation.

SDPA small-image serving TTFT improved from 71.10 to 54.46 ms at 256 visual
tokens. FA FULL remained faster than breakable: 48.04 versus 58.17 ms. At 9216
tokens, rounding to 16384 still made graph execution slower than eager. The
metadata-only eager ablation did not reproduce the small-image graph benefit.

Capture produced nine FULL graphs or 225 breakable segments (24 attention breaks
per budget). Both used a 2.445 GiB reserved graph pool and 1.076 GiB persistent
I/O storage. Matching FULL's pre-capture allocator cleanup reduced the prototype's
FA capture device-memory delta from about 5.36 to 3.623 GiB, close to FULL's
3.609 GiB. The earlier excess was primarily warmup allocator cache, not a
225-fold increase in graph storage.

Keep FULL as the default where it works well; retain breakable as an experimental
compatibility option. Exact kernel attribution, compiled eager/PW comparisons,
broader models/backends, task-quality evaluations and automatic cost-based
dispatch remain future work. All measurements are initial, bounded experiments.

The full Chinese report, raw results, serving traces, source snapshots and
reproduction scripts are preserved locally under
`.cache/vision-breakable-20261029/`; the report is `REPORT_ZH.md`.
The reusable encoder benchmark is
`benchmarks/multimodal/benchmark_encoder_breakable.py`. No PR was updated.

## FlashInfer follow-up, October 29, 2026

The prototype now supports the existing FlashInfer cuDNN vision wrapper. It
preserves element-based Q/K/O offsets, the interleaved V stride, padded sequence
slots and the shared 128 MiB workspace. The same BF16 TP1 image-only scope
applies. This is not evidence for every FlashInfer version or execution path.

With FlashInfer 0.7.2, cuDNN backend 9.24 and Qwen3-VL-4B on GB200, existing FULL
capture worked correctly. Both FULL and breakable produced bitwise-equal encoder
outputs against same-backend eager for eight layouts, including dynamic image
lengths/counts and repeated small/large transitions. The regression suites pass
93 tests. Synthetic serving image first-token outputs matched in 28/28 requests
per graph arm; a multimodal task-quality evaluation remains outstanding.

At 256 visual tokens, serving TTFT was 68.45 ms eager, 51.08 ms FULL and 56.98 ms
breakable. At 9216 tokens padded to 16384 it was 402.38, 424.57 and 422.01 ms.
Breakable did not remove non-attention padding or outperform FULL consistently.
Mixed-text emission stalls also remained worse than eager for the 9K image.
These are initial serving measurements, not claims about the performance frontier.

Separate-process capture memory deltas were 3.613 GiB FULL and 3.633 GiB
breakable, with the same 2.445 GiB reserved graph pool and 1.076 GiB persistent
I/O. The shared workspace is allocated before capture. A reversed-order run
showed a larger FULL device-level delta without increased PyTorch reserved
memory; this did not reproduce in separate processes and remains unattributed.

Keep FlashInfer FULL as the default. Breakable needs a demonstrated compatibility
case or repeatable performance gain before promotion. Remove unused breakable
metadata preparation from the FULL path before proposing a public implementation.
Scripts, raw measurements, failures and the Chinese report are preserved under
`.cache/vision-flashinfer-20261029/`. No PR was updated.

## Multi-image batching versus repeated replay: experiment plan

Updated October 29, 2026. The experiment has exactly two execution arms:
whole-batch eager and the existing FULL graph packing/replay policy. There is
no split-eager arm, padded-eager arm, forced larger graph or capture-cap sweep.
Runtime policy is unchanged; these choices are isolated in the benchmark.

### Located behavior and question

The scheduler admits uncached encoder items within encoder compute and cache
budgets. `DefaultModelState.prepare_inputs_embeds` invokes the encoder before
assembling LM embeddings. `EncoderRunner.execute_mm_encoder` groups compatible
items by modality and invokes the captured manager when enabled; otherwise it
calls the model's existing whole-batch multimodal path.

`EncoderCudaGraphManager._execute_local` sorts by output-token count and packs
under each path's graph capacity and item limit. It may also close a group early
when summed capture budgets become smaller. This heuristic compares token
budgets, not measured latency. A fitting group replays; an oversized single
image falls back to eager. Images are not cut into independent attention chunks.
Each replay prepares and copies metadata/inputs, then clones/scatters outputs
before another replay overwrites the buffer. The loop does not yield to LM decode.

The actual packing method was dry-run through existing test helpers. With cap
4096 and item capacity 64, 16x1024 uses four replays and 64x256 uses four replays;
8x768+8x1280 uses five. Even 1024+512 uses two replays because the heuristic
prefers summed budgets 1536 to a single padded 2048. Original plans are retained
in `.cache/vision-batching-20261029/packing_probe.json`. Graph hit counters count
images, not calls; the benchmark records calls separately.

The key question is whether whole-batch eager is faster and remains within the
actual memory budget across image-size distributions. For independent Qwen image
attention, batching preserves the mathematical attention work proportional to
sum(patch_length_i^2); it changes execution efficiency, overhead and intermediate
lifetimes. A graph saves host submission work but can add padding, copies and
smaller forwards. Compute-bound behavior is evaluated by size, not assumed for
every input. Prior FlashInfer timings were 83.10 ms eager / 83.63 ms one FULL
replay for 16x1024, but 20.18 / 2.92 ms for 256 tokens. Those results did not test
four 4K replays.

### Fixed configuration and distribution coverage

Use GB200, Qwen3-VL-4B-Instruct BF16 TP1, fixed 8 GiB KV and encoder compilation
disabled. Test FlashInfer and FlashAttention in separate processes on the same
node/GPU. Keep cap 4096, powers-of-two sizes from 64, item capacity 64, decoder
settings and scheduling budgets fixed. Both arms retain the same graph pool:
this measures a hybrid dispatch policy, not the memory benefit of disabling
all encoder graphs. No breakable execution is included.

The reusable `benchmark_encoder_breakable.py --suite batching` generates 42
cases, with 1-64 images and 256-16384 total visual tokens:

- Equal-sized controls at total 1024/4096/8192/16384, with per-image 256/1024/4096
  where feasible; a small256 control.
- Same 16K total: 16x1024, 64x256, 4x4096, 8x768+8x1280, one 4K plus 48x256,
  and mixed aspect ratios. Image boundaries and aspect ratios are preserved.
- Size-distribution families: small-heavy, bimodal and long-tail at total4K/8K/16K,
  two fixed random seeds each. Shape lists are fixed and shared between arms.
- Boundary cases 4096±64,8192±64,16384-64, and 1024+512.
- Explicit fallback controls: single 9216/single 16384 and one 8192 plus 32x256.
  Report these separately from batches fully handled by multiple graph replays.

These are reproducible synthetic distributions, not measured production
frequencies. Report per-family and per-case results; do not claim a production
weighted average. A real trace can later replace the shape list without changing
the two execution arms. Model-derived encoder/cache budgets are logged, and
cases beyond the effective budget are excluded. Microbench budget compliance
does not establish that the serving scheduler forms every batch in practice.

### Execution, correctness and memory

Both arms use exactly the same raw uint8 patch tensors, positions and image
order, with real model weights. Whole eager bypasses graph-driven packing and
returns per-image views. Graph execution uses the existing manager and its real
copies/output ownership; no artificial concatenation is added to timed calls.
Correctness compares every image against whole eager before timing, checking
finite outputs, bitwise equality, maximum absolute error and existing BF16
rtol/atol 0.01. No tolerance changes are permitted to hide grouping errors.

Record actual call membership, each budget, number of FULL/eager calls and
padding. This diagnostic call runs outside timing. Warm both arms, then collect
three alternating-order blocks with five samples each. Record wall time and
GPU stream span; the latter includes host submission gaps, not just kernel time.
Shuffle case order deterministically, and record GPU clocks/power/processes.
Repeat only uncertain or decision-critical points; small changes inside timing
spread are not evidence of a reliable win.

In separate untimed memory passes, retain all per-image outputs and record
allocated/reserved baselines, their peaks, and device-used memory. Log capture
device delta and graph-pool reserved separately. PyTorch peak counters do not
measure every external cuDNN allocation, and sampled device readings are not
an exact continuous device peak. Keep memory scopes explicit and do not add
overlapping categories. Report OOM/infeasibility rather than silently shrinking
an input or freeing the shared graph pool for eager.

Scheduler admission and existing startup encoder profiling make whole eager a
legitimate baseline. They do not prove every input-size distribution has the
same backend workspace peak. Test few-large, many-small and mixed images near
the admitted budget, and confirm production KV/cache occupancy before changing
a default. The first fixed 8 GiB-KV run is not that final deployment proof.

### Serving confirmation and decision

After the encoder comparison, retain the same two execution choices for a small
set of representative multi-image requests, alone and with four active text
streams. Keep LM graph/KV and scheduling settings identical. Record actual
scheduler-selected items, encoder batch sizes, image TTFT and text output stalls.
A 16-image request is not proof that one encoder call contains all 16 images.
If the original 8192 LM-token budget does not form a target batch, report that;
a separately labeled controlled larger-LM-budget cohort may be used, but never
change scheduling budget between execution arms.

Report eager/replay latency ratios, paired-block variation, memory and correctness
by distribution and replay count. Distinguish zero/one/multiple replay and eager
fallback cases. If whole eager consistently wins at acceptable memory, select
it before graph packing for the supported workload range. Retain graphs where
they win; do not infer a universal threshold from total tokens alone. A model/
backend-specific policy may need image count, maximum image length, distribution
and predicted replay/padding counts. No default change or PR publication is part
of this experiment.

### Completed two-arm measurements

Both backends completed all 42 cases in two independent runs (jobs 41413/41415).
All 168 case comparisons had bitwise-equal per-image outputs. At fixed cap 4096,
FlashInfer wall times were 83.20 ms whole eager versus 87.38 ms replay for 16x1024,
and 85.14 versus 97.35 ms for 8x768+8x1280. The analogous FlashAttention values
were 84.68/88.01 and 86.13/98.86 ms. Across the tested 8K/16K multi-replay cases,
replay/eager ratios ranged approximately 1.01-1.22. These are synthetic shape
coverage results, not production-weighted speedups or a universal threshold.
For 1024+512, two replays remained much faster than whole eager.

The memory tradeoff is material: for 16K inputs, extra PyTorch allocated peak
above the pre-forward baseline was about 1.742 GiB whole eager versus 0.33 GiB
replay. Both arms retained the same 0.617 GiB graph pool and fixed 8 GiB KV. This
does not establish safety at full KV occupancy or for other models. The current
experiment measured encoder latency; new TTFT/ITL measurements remain pending.

Raw samples, actual call plans, correctness, allocator/device measurements,
reproduction scripts, an 84-row summary CSV and a plot are in
`.cache/vision-batching-20261029/`; `REPORT_ZH.md` explains scopes and limitations.

## Dispatch update: one replay per local path (2026-10-30 PDT)

The local prototype now defaults `encoder_cudagraph_oversize_policy` to `eager`.
For each batch after encoder DP sharding, it selects the smallest captured budget
that holds the whole batch on each active path. It preserves item order and does
not split to reduce the sum of padded budgets. If any path exceeds its token cap,
or the batch exceeds captured item capacity, the whole local batch runs eager.
DeepSeek V4.1 uses its packed encoder implementation for this fallback. Explicit
`split` retains the earlier greedy packing for comparison and memory constraints.

Capture-axis misses and replay inputs larger than their captured buffers also
fall back safely, without trying several graphs or copying oversized metadata.
This checks dimensions on the host; it does not read tensor contents back from
the GPU. MiniMax's attention padding and NaN protections are unchanged.

The replay count here is per active path and local batch. Independent model
paths or different DP ranks still execute separately; a breakable graph may
itself contain multiple segments. The decoder's grid and scheduler admission
budgets are unchanged.

### Grid and measurements

More capture sizes and a larger maximum size are separate choices. We tested
the same cap with more sizes to isolate padding, and a lower cap with whole-batch
eager above it. Grids remain explicit `encoder_cudagraph_token_budgets`; the
automatic grid generator has not been replaced with a model-independent optimum.

- Coarse 8K: `[128, 256, 512, 1024, 2048, 4096, 8192]` (7 graphs).
- Fine 8K: `[128, 256, 512, 640, 768, 896, 1024, 1152, 1280, 1408,
  1536, 1664, 1792, 1920, 2048, 2560, 3072, 3584, 4096, 4608, 5120,
  5632, 6144, 6656, 7168, 7680, 8192]` (27 graphs).
- Fine 4K: the first 19 entries of Fine 8K; batches above 4096 run eager.

These candidate grids use regular intervals, not a captured graph for each test
image. The 1152 entry, for example, serves the measured 1088-token batch with
64 padded visual tokens. A separate exact-size arm is an experimental control,
not part of these deployment grids. All three configurations use a captured item
capacity of 128; the token budget is in model-defined output visual tokens, not
pixels or attention patch rows.

GB200, BF16, FA4, real DeepSeek-V4.1-Flash and MiniMax-M3.1-Flash vision weights,
TP1, encoder only. Three randomized blocks of ten warmed samples; median GPU
stream span includes metadata preparation, input copies, padding, dispatch and
output assembly. The latency run retains a common superset of graphs. No LM or
KV cache is loaded, so these are not TTFT/ITL measurements.

| Model / batch | Old split, coarse 8K | One replay, coarse 8K | One replay, fine 8K | Fine 4K / eager above cap |
| --- | ---: | ---: | ---: | ---: |
| MiniMax, mixed 1088 tokens | 15.10 ms (2 replays) | 16.80 ms | 10.23 ms | 10.22 ms |
| MiniMax, mixed 1600 tokens | 15.36 ms | 15.35 ms | 12.83 ms | 12.84 ms |
| MiniMax, 65 small images / 4160 tokens | 83.46 ms | 83.39 ms | 31.70 ms | 28.11 ms (eager) |
| MiniMax, 256 small images / 16384 tokens | 111.86 ms (2 replays) | 103.85 ms (eager) | 103.82 ms (eager) | 103.70 ms (eager) |
| DeepSeek, 8 small images / 1472 tokens | 44.76 ms | 44.75 ms | 34.44 ms | 34.47 ms |
| DeepSeek, 32 small images / 5888 tokens | 173.41 ms | 173.25 ms | 136.03 ms | 124.30 ms (eager) |
| DeepSeek, 64 small images / 11776 tokens | 269.11 ms (2 replays) | 240.08 ms (eager) | 240.29 ms (eager) | 240.09 ms (eager) |

Removing splits alone can regress performance: the MiniMax 1088-token case
pads to 2048 on the coarse grid. Adding 1152 reduces both replay count and padded
work. Graphs still do not win everywhere: MiniMax at 4096 tokens measured
27.16 ms packed eager versus 28.45 ms with Fine 8K. A universal threshold or
end-to-end speedup is not established by this sweep.

Fresh-process capture memory, GiB/GPU, with only the indicated grid captured:

| Model | Grid | Device-used delta | Graph pool reserved | Persistent inputs | Persistent outputs |
| --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek | Coarse 8K | 3.318 | 2.744 | 0.196 | 0.155 |
| DeepSeek | Fine 8K | 4.869 | 2.744 | 0.986 | 0.778 |
| DeepSeek | Fine 4K | 2.330 | 1.373 | 0.367 | 0.289 |
| MiniMax | Coarse 8K | 2.594 | 1.924 | 0.199 | 0.186 |
| MiniMax | Fine 8K | 4.068 | 1.924 | 0.999 | 0.933 |
| MiniMax | Fine 4K | 1.893 | 0.963 | 0.372 | 0.347 |

The device-used delta includes more than the graph pool; the breakdown columns
are not an exhaustive additive partition. With a fixed maximum, most of the
extra retained memory is per-size input/output storage. Whole-batch eager can
increase runtime workspace even when a smaller capture grid saves memory;
production KV occupancy still requires validation.
For DeepSeek's 64-small-image batch, the extra PyTorch allocated peak above
post-call live allocation was 0.222 GiB with splitting versus 2.299 GiB with
whole-batch eager. For MiniMax's 256-small-image batch it was 0.259 versus
1.776 GiB. These include transient outputs and workspace, and are measured with
the shared graph superset retained; they are not total deployment peak memory.

The final dispatcher test suite passed 90 tests, covering one-replay packing,
sparse-grid small-image floods, added intermediate sizes, original output order,
multi-path and idle paths, capture-axis handling, oversized metadata fallback,
and output lifetime across replay/eager transitions. Real-model embedding checks
use BF16 `rtol=0.01, atol=0.01`, not task-level accuracy. Across 29 real-model
layouts, all candidate single-dispatch policies used zero or one replay and
produced finite outputs. Twenty-eight layouts passed the embedding tolerance.
The MiniMax 336x336 input reproduced the existing discrepancy: both old and new
dispatch had relative L2 error 0.016063 against packed eager. It is not a NaN and
is not resolved by this dispatch change. The other four additional unaligned
layouts passed; this does not establish task-level accuracy for the failing case.

Reproduce the main encoder comparison with:

```bash
.venv/bin/python benchmarks/multimodal/benchmark_encoder_crossmodel.py \
  --model minimax --dispatch-ablation --max-items 128 \
  --blocks 3 --repeats 10 --output minimax-dispatch.jsonl
# Repeat with --model deepseek. For independent capture memory, add:
# --memory-only --policy single-fine-8K (or single-coarse-8K / single-fine-4K).
.venv/bin/python -m pytest tests/v1/cudagraph/test_encoder_cudagraph.py -q
```

The harness currently resolves pinned checkpoints from the cluster HF cache;
their IDs and revisions are recorded in each output. Source snapshots, before-
change dispatch/config, explicit configuration JSON, test logs and independent
memory runs are retained in `.cache/vision-dispatch-20261031/`. Latency and
embedding data are in `.cache/vision-crossmodel-20261030/`, jobs 41733/41734 and
boundary jobs 41738/41739. The former two precede the final extra metadata bounds
check; boundary jobs and the final 90-test run include it. No PR is published by
this update. Unaligned MiniMax checks are in job 41740; the merged JSON summary
and latency CSV, including failures, are in `.cache/vision-dispatch-20261031/`.

## Padding-cliff regression sweep (2026-10-30 PDT)

A targeted follow-up tested batches just above 256, 512, 1024, 2048 and 4096
visual tokens, with both identical small images and a small image mixed with
larger images. The latter matters: the old split policy can reduce padding much
more effectively for a mixed batch than for a flood of equal-sized images.
The baseline is the same-build `split-8K` policy with captured item capacity 128,
not an independently installed upstream binary or an end-to-end serving run.

Two independent startups per model used different randomized case/arm orders,
five blocks of ten timed calls per arm, GB200/BF16/FA4 and the pinned real vision
weights. There were 54 case-runs representing 29 distinct model/input layouts;
four additional model/layout combinations were introduced only in the second
startup. Compilation, warmup and correctness checks were excluded from timing.
Metadata, staging, padding, dispatch and output assembly remained included.

### Regressions found and corrected configuration

The previous 27-entry Fine 8K grid is not regression-free. MiniMax reproduced:

| Input | Old split plan | Fine 8K plan | Fine 8K latency regression, two startups |
| --- | --- | --- | ---: |
| 4 x 512 + 64 = 2112 visual tokens | 128 + 2048 | 2560 | +1.36%, +2.57% |
| 8 x 512 + 64 = 4160 visual tokens | 128 + 4096 | 4608 | +2.29%, +1.49% |

At 2112 tokens, the new coarse step raised executed patch rows from 8704 to
10240 (8448 are real), outweighing the saved replay. At 4160, it raised rows
from 16896 to 18432 (16640 real). The five block medians had the same regression
direction in each startup; these findings are retained, not averaged away.
An added held-out composition, 8 x 256 + 64, also regressed 1.61% with Fine 8K.

The corrected candidate uses `[128, 256, ..., 4096]`, 32 sizes, one replay per
local path, and whole-batch packed eager above 4096. This uses existing encoder
arguments. It adds the missing 2176 entry and other regular intermediate sizes;
it does not special-case the measured image dimensions. With the current
128-item dummy capture construction, these increments correspond to distinct
physical buffer sizes. Sub-128 increments alone can duplicate an allocation
shape rather than reduce padding.

Representative first-startup medians and independent-repeat latency reductions:

| Model / input | Old split | Fine 8K | Corrected candidate | Reduction vs old, two startups |
| --- | ---: | ---: | ---: | ---: |
| MiniMax, mixed 2112 | 19.484 ms | 19.749 ms | 17.543 ms, graph 2176 | 9.96%, 7.85% |
| MiniMax, mixed 4160 | 34.708 ms | 35.503 ms | 32.722 ms, packed eager | 5.72%, 6.05% |
| MiniMax, 65 small images / 4160 | 83.179 ms | 31.446 ms | 27.923 ms, packed eager | 66.43%, 65.21% |
| DeepSeek, 12 small images / 2208 | 86.882 ms | 57.946 ms | 53.055 ms, graph 2304 | 38.93%, 39.32% |
| DeepSeek, mixed 4174 | 169.084 ms | 103.808 ms | 86.410 ms, packed eager | 48.90%, 49.05% |

No reproducible slowdown was observed for the corrected candidate over these
cases. Its largest positive median difference was +0.39% for MiniMax's 4096-token
control; that case was -1.14% in the other startup, with mixed signs across
blocks. DeepSeek's largest positive difference was +0.23%, also at a control
that uses the same graph as the baseline. This is a bounded regression check,
not proof for all configurations or all image distributions.

Every tested embedding comparison passed the existing BF16 tolerance
`rtol=0.01, atol=0.01`; all candidate plans used zero or one replay. The separate
336x336 MiniMax numerical discrepancy documented above was not resolved or
reclassified by this workload sweep. No new TTFT/ITL or multimodal task-accuracy
claim is made.

### Memory and reproducibility

Fresh-process capture measurements for the 32-entry candidate, GiB/GPU:

| Model | Device-used delta | Graph pool reserved | Persistent input | Persistent output |
| --- | ---: | ---: | ---: | ---: |
| MiniMax | 2.789 | 0.963 | 0.828 | 0.773 |
| DeepSeek | 3.199 | 1.373 | 0.817 | 0.645 |

Relative to the earlier 19-entry Fine 4K grid, adding sizes increased capture
device-used memory by approximately 0.90 GiB for MiniMax and 0.87 GiB for
DeepSeek. It remains below the earlier Fine 8K measurement, because the maximum
capture shape is smaller. Whole-batch eager still has the runtime workspace
tradeoff documented above. Memory columns are not an exhaustive additive sum.

The corrected configuration is saved at
`.cache/vision-cliffs-20261031/candidate-linear-4K.json`; automatic grids and
global model defaults have not been replaced on the strength of this sweep.
Raw per-call samples, replay plans, physical padding, block deltas, correctness,
source snapshots and failure/preemption logs are retained. Completed latency
jobs are 41750/41751 and 41757/41753; 41752 was preempted before producing usable
measurements and excluded. Independent memory jobs are 41754/41755.

```bash
.venv/bin/python benchmarks/multimodal/benchmark_encoder_crossmodel.py \
  --model minimax --scenario padding-cliffs --dispatch-ablation \
  --max-items 128 --blocks 5 --repeats 10 --seed 4721 \
  --timing-arms split-8K,single-fine-8K,single-linear-4K,packed-eager \
  --output minimax-padding-cliffs.jsonl
# Repeat with --model deepseek; use a different seed for the independent startup.
# Memory: add --memory-only --policy single-linear-4K.
```

The merged comparison is `.cache/vision-cliffs-20261031/comparison.csv`, with
the full checks in `summary.json`. This is a local experiment/configuration
update; no PR or remote branch was changed.

### Exact boundary-plus-one follow-up (2026-11-02 UTC)

The previous cliff sweep did not systematically cover the smallest legal input
above each capture boundary: its MiniMax increments were commonly 64 tokens.
Its no-reproducible-regression finding must not be extended to all `B + 1` inputs.

A follow-up targets 129, 257, 513, 1025, 2049, 3073, 3969 and 4097 output visual
tokens. It compares old same-build `split-8K`, the 32-size `single-linear-4K`
candidate, and whole-batch packed eager, with both few-image and many-image
compositions. Real image-processor preflight passed for 27 distinct layouts.
DeepSeek's first target uses 170 tokens, the next reachable size in the enumerated
legal-image catalog; it is not reported as an exact 129-token case.

Jobs 42969 (MiniMax) and 42970 (DeepSeek) completed on GB200/BF16/FA4, with five
randomized blocks of ten calls per arm. All 27 candidate encoder outputs were
bitwise equal to packed eager; comparisons against the native eager and old
split-policy outputs also passed `rtol=0.01, atol=0.01`. This is an encoder-output
check, not a new multimodal task evaluation or a resolution of the separate
336x336 numerical issue above. All candidate plans used zero or one replay.

First-startup medians, including metadata, staging and output assembly:

| Model / input | Old split policy | Candidate | Packed eager | Candidate vs old | Candidate vs eager |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniMax, mixed 1025 | 16.860 ms | 9.953 ms | 20.758 ms | -41.0% | -52.1% |
| MiniMax, mixed 3073 | 32.325 ms | 25.692 ms | 24.966 ms | -20.5% | +2.9% |
| MiniMax, many-small 3969 | 29.900 ms | 29.759 ms | 28.482 ms | -0.5% | +4.5% |
| MiniMax, many-small 4097 | 84.452 ms | 29.737 ms | 29.659 ms | -64.8% | +0.3% |
| DeepSeek, few-image 1025 | 37.238 ms | 31.749 ms | 32.075 ms | -14.7% | -1.0% |
| DeepSeek, many-small 2049 | 87.137 ms | 50.479 ms | 48.275 ms | -42.1% | +4.6% |
| DeepSeek, many-small 4097 | 169.431 ms | 94.118 ms | 94.091 ms | -44.4% | +0.03% |

The candidate's largest positive latency difference against the old policy was
+0.25% (MiniMax mixed 3969); both policies use the same 4096 graph there. A second
startup with a different randomized order completed as jobs 42996/42997. Across
both startups, all 54 candidate outputs were bitwise equal to packed eager and
all reference comparisons passed. MiniMax mixed 3969 was +0.10% in the repeat;
there was no larger positive difference across the tested inputs. This bounded
result is not proof of zero regressions for all inputs.

Against packed eager, the candidate still leaves measurable headroom at some
2K--4K boundary inputs. DeepSeek many-small 2049 has 17964 real patch rows but
its 2176 graph executes 19584 rows, +9.0%; all five blocks were 4.5--4.7% slower
than packed eager. MiniMax 3969 executes 16384 rather than 15876 rows, +3.2%.
The eager advantage reproduced: DeepSeek many-small 2049 was 4.78% slower than
eager in the second startup (first: 4.57%), and MiniMax many-small 3969 was 4.85%
slower (first: 4.48%).
These measurements are consistent with padding costs exceeding saved host work,
but they do not isolate a kernel-level attribution. A 4096 dispatch threshold
is therefore a candidate, not an established per-model optimum. In contrast,
small inputs still benefit strongly from graph replay despite proportionally
larger padding. Token count alone does not describe image composition or the
physical patch workload.

Capture uses a shared superset of 48 entries to compare policies within one
process; it is not the candidate's deployment graph count or deployment memory.
The candidate still specifies 32 sizes. At 4097 tokens, whole-batch eager raises
the measured transient allocated peak above post-call live allocations: MiniMax
0.083 to 0.444 GiB, DeepSeek 0.114--0.117 to 0.848--0.851 GiB, depending on image
composition. These are shared-superset runtime deltas, not deployment totals or
capture-memory deltas; they must be checked against available serving headroom.
Physical capture-buffer shapes, actual
patch rows, replay plans, per-call timing, runtime allocator measurements and
all correctness checks are retained in `.cache/vision-one-above-20261102/`, with
`analyze.py`, `comparison.csv`, `summary.json` and source snapshots. The pinned
checkpoint revisions are unchanged; required vision shards were restored to an
owned directory after shared-cache read access became unavailable. No production
dispatch change or PR update was made for this follow-up.

### Open work: MiniMax padding correctness and dispatch tuning (Nov 1, PDT)

Keep these two issues separate:

| Issue | Evidence | Next action |
| --- | --- | --- |
| Padding sequence computes extra attention to protect correctness | MiniMax's `pad_cu_seqlens` registers unused rows as a separate nonempty sequence. Removing it while poisoning unused attention output with NaN reproduced nonfinite valid outputs in job 41622. | Retain the protection. Establish whether an unmodified production path can create the invalid-tail state; evaluate a backend fix or a cheaper finite-tail guarantee with stress and ordinary-input tests. |
| Graph is slower than packed eager for some covered inputs | Two startups reproduced approximately 4--5% slower encoder latency for MiniMax 3969 and DeepSeek 2049 token cases. | Tune per-model capture limits using existing arguments; validate both sides of the switching boundary and exact-hit controls, including many-small and mixed-image batches. |

The NaN stress result is not evidence that default MiniMax spontaneously
produces NaN, nor evidence that it needs an additional clearing pass today.
Jobs 41690/41691 demonstrate finite output under injected-tail stress with extra
output or QKV/output clearing, but the cost and production necessity remain
separate questions. The ordinary run 41692 is retained as a different category.
The known unaligned numerical discrepancy must still be tracked separately from
NaN. Dispatch tuning does not remove or resolve the existing NaN guard.

The local implementation now avoids concatenating all patch rows again when
MiniMax or DeepSeek selects the entire batch in its original order. Subsets,
permutations, duplicate selections and empty inputs retain their existing paths.
The encoder test suite covers these selection contracts. Screen jobs 43282/43283
measure threshold neighborhoods with the same real model revisions; candidate
limits and independent validation are reported in the next section. Artifacts are under
`.cache/vision-dispatch-tuning-20261102/`; old measurements are preserved.

### Implemented dispatch refinement and validation (Nov 1, PDT)

The 127-case screen (MiniMax 63, DeepSeek 64) showed why a lower cap alone is
insufficient: DeepSeek's exact 2048-token graph can beat eager, while 2049 tokens
padded to 2176 can lose. Dropping the entire range creates a new regression.
The local manager therefore has an optional padding rule, disabled by default:
at or above `encoder_cudagraph_eager_min_tokens`, use whole-batch packed eager
if `(fitting budget - tokens) / tokens` exceeds
`encoder_cudagraph_max_padding_ratio`. Exact and nearly exact graphs remain
available. The existing `split` policy ignores this rule. Admission limits and
model-specific padding correctness logic are unchanged.

Measured GB200/BF16/FA4 candidate configurations, not universal defaults:

| Model | Capture grid | Additional dispatch rule |
| --- | --- | --- |
| MiniMax-M3.1-Flash | 128-token steps through 2944; 23 entries | Whole-batch eager above 2944. |
| DeepSeek-V4.1-Flash | 128-token steps through 4096; 32 entries | From 1536 tokens, use eager if token padding exceeds 2%; otherwise use the fitting graph. |

The settings are saved as `candidate-minimax.json` and `candidate-deepseek.json`
in `.cache/vision-dispatch-tuning-20261102/`. A separate fixed-seed holdout used
new image-size combinations: 18 MiniMax cases and 17 DeepSeek cases. One DeepSeek
128-image control exceeded the fixed 16384-token admission budget and was
explicitly excluded, not silently resized. Jobs 43301/43302 completed these
35 cases; 43310/43311 retested ten original boundary cases, and 43320 repeated
two exact-hit controls with nine blocks of twenty samples. All 47 candidate
encoder outputs were bitwise equal to packed eager, with all native/split
reference checks also passing the existing BF16 tolerance. The encoder unit/GPU
suite passed 105 tests. This does not establish end-to-end TTFT or task accuracy.

Representative measured encoder latency, including preparation and output copies:

| Model / input | Old split-8K | Previous 4K candidate | Refined candidate | Change vs previous 4K |
| --- | ---: | ---: | ---: | ---: |
| MiniMax, many-small 3969 | 29.661 ms | 29.500 ms | 28.328 ms | -3.97% |
| MiniMax, mixed 3969 | 33.172 ms | 33.177 ms | 32.027 ms | -3.47% |
| MiniMax, held-out mixed 3413 | 31.789 ms | 27.149 ms | 26.601 ms | -2.02% |
| DeepSeek, many-small 2049 | 87.260 ms | 50.444 ms | 48.375 ms | -4.10% |
| DeepSeek, few-image 2049 | 67.835 ms | 63.657 ms | 61.218 ms | -3.83% |
| DeepSeek, held-out mixed 3464 | 91.745 ms | 82.053 ms | 75.384 ms | -8.13% |

The previously observed 4--5% loss against packed eager is removed on the
targeted inputs by choosing that path. Small graph-benefiting inputs remain on
graphs. A MiniMax 2048-token control measured +0.60% against the old policy
despite using the same graph and less host preparation time; job 43320's longer
independent repeat measured -0.16%. The four-wide-image 2048-token control was
-0.29% initially and +0.27% in the repeat. Paired block differences have both
signs, so these controls do not establish a reproducible regression. Retain the
samples; this is still bounded evidence, not proof of universal equivalence.
These policies are conservative candidates; image composition still affects
the best choice, so a global optimum is not claimed.

MiniMax's fresh 23-entry capture measurement (job 43309) is 1.771 GiB/GPU in
device-used delta: graph pool reserved 0.701, persistent inputs 0.433, persistent
outputs 0.404 GiB. A same-build, fresh-process 32-entry 4K capture (job 43332)
measured 2.789 GiB, confirming a 1.018 GiB / 36.5% capture-device-delta reduction.
Its pool reserved / persistent input / persistent output were 0.963 / 0.828 /
0.773 GiB respectively; these categories are not an exhaustive additive sum.
Earlier eager dispatch increases runtime workspace:
the 3413-token MiniMax holdout rose from 0.040 to 0.372 GiB transient allocation,
and the 3464-token DeepSeek holdout from 0.064 to 0.666 GiB. These runtime deltas
and capture deltas have different scopes and are not a deployment-total sum.
The NaN guard remains an explicitly open optimization issue; this change does
not add clearing kernels or remove the protective trailing attention sequence.

Raw samples, source versions, per-arm replay plans, correctness checks and
runtime memory remain in the experiment directory. `scan_caps.py` is restricted
to the two screening jobs; held-out data does not enter its selection estimates.
`analyze_validation.py` produces `validation-summary.json` and `validation.csv`.
No public PR or remote branch was changed.

### DeepSeek regression attribution: patch capacity and empty sequence slots

Two independent encoder-only runs (jobs 43412 and 43413) isolated the remaining
DeepSeek holdout regressions. They use the same pinned real weights, GB200,
BF16, FA4, TP1, and deterministic processed images as jobs 43302/43311. The
dispatch/model files from the prior validation were overlaid on an isolated
source snapshot. No production dispatch or NaN protection was changed.
Each arm has three warmups and five randomized-order blocks of eight samples.
All 30 arm/case correctness checks across the two starts were bitwise equal to
packed eager. These are encoder checks, not serving/task evaluation results.

**Finding:** the output-token padding percentage understated the actual ViT
buffer padding. Extra patch-row work is the dominant cost; unused sequence
slots add a smaller attention scheduling cost. Tightening `max_seqlen` alone
does not materially help these cases.

| Input | Output tokens / selected budget | Output-token padding | Actual / captured patch rows | Patch-row padding |
| --- | ---: | ---: | ---: | ---: |
| `heldout-13-0`, 13 images | 2636 / 2688 | 1.97% | 21295 / 24192 | 13.60% |
| `heldout-13-1`, 13 images | 2794 / 2816 | 0.79% | 22325 / 25344 | 13.52% |

`DeepseekV4VLEncoderCudaGraphMixin.prepare_encoder_cudagraph_capture_inputs`
allocates `r*r*ceil(budget/max_items)*max_items` patch rows. Here `r=3` and
`max_items=128`. The output-token count includes image delimiters and row
separators, while ViT work uses pre-merge patch rows. Consequently, even adding
exact output-token entries 2636/2794 leaves the same patch-buffer capacities.

Representative complete-encoder medians from the independent repeat, job
43413, include preparation, staging, graph/eager execution and output assembly:

| Diagnostic arm | 2636 tokens | 2794 tokens |
| --- | ---: | ---: |
| Current graph path | 61.772 ms | 65.319 ms |
| Whole-batch packed eager | 58.004 ms | 60.695 ms |
| Same buffers, only tighter attention `max_seqlen` | 61.720 ms | 65.352 ms |
| Same buffers, only remove empty sequence slots | 60.907 ms | 64.533 ms |
| Both attention metadata changes, same patch buffers | 60.994 ms | 64.466 ms |
| Also trim patch/cos/sin capacity to actual rows | 55.130 ms | 58.293 ms |
| Also trim merge/output rows to actual rows | 55.066 ms | 58.221 ms |

The additional buffer-matched custom-path control measured 61.753/65.323 ms,
close to the production-path baseline; custom preparation code is not the
source of the large gains. The first start reproduced the direction and size
of the component effects. The specialized graphs are diagnostic controls for
these exact inputs, not deployable general-purpose capture plans.

To attribute work inside the graph separately from preparation, both runs also
timed already-prepared forwards. The sequential ablation contributions were:

| Component | 2636 tokens, two starts | 2794 tokens, two starts |
| --- | ---: | ---: |
| Extra patch capacity | 5.739–5.749 ms | 6.165–6.168 ms |
| Attention empty slots/bounds combined | 0.729–0.732 ms | 0.746–0.747 ms |
| Extra merge capacity | 0.011–0.020 ms | 0.018–0.020 ms |
| Net graph benefit once shapes are matched | 2.875–2.914 ms saved | 2.428–2.512 ms saved |
| Difference in surrounding preparation/copy/output costs | 0.161–0.201 ms | 0.206–0.369 ms |

For example, the repeat's first case closes the latency balance, approximately:
`5.739 + 0.732 + 0.011 + 0.161 - 2.875 = 3.768 ms`, matching
`61.772 - 58.004`. These are ordered interventions and differences of medians;
they are not a unique additive cost model for arbitrary workloads. Graph's net
benefit includes its existing output-copy and correctness-guard costs, rather
than estimating pure launch overhead in isolation.

CUPTI-backed PyTorch traces corroborate the attribution. On the first case,
trimming patch rows after attention metadata was tightened reduced GEMM kernel
time by about 0.84 ms, dtype-copy kernels by 1.42 ms, and elementwise/reduction/
other kernels by 3.60 ms; attention kernel time was essentially unchanged.
These are sums of kernel durations from instrumented runs, not the unprofiled
wall-latency numbers above. The large cost is not solely GEMM: unfused per-row
operations also process the padded capacity throughout the 32 ViT layers.

The empty-slot effect is visible in actual CUDA launch grids. The original
`cu_seqlens` has 130 entries (129 sequence slots), versus 14 entries for the
13 real images. For these FA4 kernels, the observed query tile is 256 rows and
there are 16 heads. `SingleTileVarlenScheduler.get_grid_shape` computes the
conservative grid as `floor((total_q + num_slots * 255) / 256) * 16` CTAs:

| Capture layout | 2636-token case | 2794-token case |
| --- | ---: | ---: |
| Original patch capacity and 129 slots | 3552 CTAs | 3632 CTAs |
| Same patch capacity, 13 slots | 1712 CTAs | 1776 CTAs |
| Actual patch rows, 13 slots | 1536 CTAs | 1600 CTAs |

Tightening `max_seqlen` from 24192/25344 to 2304 alone leaves these CTA grids
unchanged. Unused slots remain zero-length: they do not perform full fake-image
attention, but conservative launch/scheduling work is not free. The first
case's attention time across 32 layers fell from approximately 7.47 to 6.67 ms
after removing unused slots. Effective image boundaries and outputs remained
unchanged. This is distinct from MiniMax's nonempty padding-sequence NaN guard.

**Implementation implication:** first evaluate actual input-row capacity in
dispatch, not only output-token padding. This can avoid the two slow choices
without startup autotuning. Improving capture layouts/slot bucketing is a
separate optimization: a specialized 13-image graph cannot safely replace a
general 128-item graph without coverage checks. Simply reducing the overall
cap bypasses the problem; simply adding output-token grid entries does not
remove the internal capacity rounding demonstrated here.

Reproduction: `benchmarks/multimodal/benchmark_deepseek_encoder_attribution.py`.
Frozen sources, Slurm command, raw samples, traces, `summary.json`, and
`latencies.csv` are retained under
`.cache/vision-deepseek-attribution-20261102/`. `analyze.py` regenerates the
tables and records source hashes. The benchmark passes Ruff and Python syntax
checks. No public PR or production configuration was updated by this analysis.
