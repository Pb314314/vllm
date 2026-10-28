# Top-k benchmark method

- Worktree: `/home/bopang/cg-project/wt-indexer-topk-breakdown`; source branch
  `cg/indexer-topk-breakdown-results` (plan input: `cg/indexer-topk-breakdown`). Each JSONL line records source SHA, dirty state,
  runner SHA256, hostname, GPU, versions, telemetry and timing configuration.
- Interpreter: shared `.venv/bin/python`, PyTorch 2.14.0+cu132; compiled vLLM
  artifacts supplied by the parent from the same `cff6224872` product sources.
  CUPTI-python 13.4.0 is installed (PyTorch CUDA runtime 13.2). The runner
  requires CUPTI-python >=13 and refuses the known FlashInfer
  version-based CUDA-event fallback. CUPTI kernel names verify actual tracing.
- Initial smoke/probe: `gb200-rack1-01`, physical GPU0 only. Parent reserved
  other GPUs for independent pipeline/memory tasks. SM clock locking was denied
  by driver permissions; before/after clocks, temperature and power are sampled
  for every record. Treat results as unlocked clocks, not guaranteed 2062MHz.
- Main scan: dedicated Slurm40469, `gb200-rack1-04`, GPU0. A separate cold-L2
  and sentinel smoke confirmed actual CUPTI traces on this node before scanning.
  Main/core trial groups do not combine dev01 probe samples with dedicated04.
- After the memory agent fully released GPUs2/3, GPU2 was assigned to saved
  real-score/control and supplementary comparisons, with CPU70–85 pinned and
  explicit UUID. GPU0 continues primary core scans. Each A/B comparison and
  three-trial group stays on one GPU. A GPU2 anchor repeats GPU0 shapes before
  comparison; absolute times from different devices are not pooled.
- Core graph scan leaves CPU affinity unchanged. Later supplement/sample/width
  scripts pin their inherited process affinity to NUMA-local cores0–15; the
  pipeline agent uses cores32–39 for its separate eager tests. New records
  include `cpu_affinity`. Eager conclusions use the pinned supplement rather
  than interpreting unpinned core host samples as serving latency.
- Triton cache: shared `/home/bopang/cg-project/design/idx-breakdown-cache/topk`.
  No product code is modified by these experiments.

- At11:52 UTC the completed pipeline measurements released GPUs1/3. The full
  P0 row grid for V3.2/k2048 uses GPU1/CPU16–31; V4-Pro/k1024 uses
  GPU3/CPU104–119. GPU0's width/small-row extension stays on CPU0–15,
  GPU2's real/supplement scan stays on CPU70–85. Each process has its own
  CUDA-visible device and result file, with no same-GPU concurrent timings.
- The64K short-cap completion keeps all45 P0 row counts and all six length
  profiles, but omits the two standalone FP32-only counterfactual schemes to
  preserve the overnight time box. DS-bf16, UP+COOP, UP+PERS, UP+DS-fp32 and
  UP-only remain100 invocations x3 trials throughout. The primary model-width
  grids retain both FP32-only schemes; their per-kernel traces also separate
  conversion from selection. Optional FI/PERROW keep the plan's sparse grid.
- Later P0 completion uses `--resume-from` only for already measured exact
  case keys on the same host/physical GPU and historical native environment.
  Input generation is seeded independently of shape ordering. Reused raw data
  stays in its original JSONL; each new record retains source-file SHA256 and
  byte count, and the summary spans both files. Unsupported keys can also be
  reused. This does not treat another GPU or PR-native result as the same trial.
- A separate PR-native anchor runs from `wt-b4a0fix`, exact HEAD
  `4e5b4f8ce9b067d92e7373d950727d8238e9a289`, using
  `venv-b4a0fix/bin/python`. New records include actual imported Python root and
  SHA256/size of the loaded native library and DeepSelect extension. Historical
  records before this addition remain identified by the environment freeze and
  sibling pipeline artifact manifest; they are not relabeled as PR-native.
  Targeted CV reruns select the original result's native tree/interpreter.

| Native artifact | Historical cff environment SHA256 | PR4e5 environment SHA256 |
| --- | --- | --- |
| `_C_stable_libtorch.abi3.so` | `92a34d708922249ccc5daf8def36b69b112f1a3a0afc44aeab2b9c9a8d82283a` | `5c9bbcf71062119fb8e0365f53e29035b25c7328e734cf42d595e9f3920e128b` |
| `_deepselect_C.abi3.so` | `b348addeaeb75a13cba06d4a776b307844e486030367e465585589b0711623bc` | same binary |

The legacy native selector library differs between versions; DeepSelect is
byte-identical. Historical cff timing is therefore not relabeled as PR4e5
timing even when an exported operator has the same name.

- FlashInfer0.7.2 build commit is
  `d84335d8c54e0c8f9f7e22faa15c7e34a58abf94`. A large-stride clusters-path
  failure is isolated and documented in TOPK.md; debug runs are not performance
  evidence. The GPU core omits global/shared/local/const memory and CPU core
  dumping was disabled. `topk_fi_core_focus.log` preserves offline cuda-gdb
  backtrace/disassembly, and the manifest retains the full core's SHA/path.
  The [official vLLM debugging workflow](https://vllm.ai/blog/2025-08-11-cuda-debugging)
  was used without modifying the installed third-party implementation.

## Input and workload contract

Start is BF16 logits, logical shape `[rows, capacity]`, with padded row stride.
Pipeline's actual DeepGEMM output is capacity+512 at W256K and1M, but
**capacity+1024 at W512K** (confirmed over both quantizations, all measured
rows and contexts). C2 had not started when discovered and uses1024 from its
first record; C1 and V4 use512. Saved files preserve the original stride.
Other widths default to512 unless explicitly overridden and are flagged as
artificial layout if not verified by a producer. `seq_lens` gives effective
columns per row and never exceeds capacity. Upcast allocates full FP32 capacity
but reads/writes only valid columns. Thus FP32 allocation is approximately
`4 * rows * stride`, while useful conversion traffic is `6 * sum(seq_lens)`.
Capacity and traffic must not be conflated. All lengths are compressed score
columns, not uncompressed context-token lengths.

`short=4096`, `mid=32768`, `long=131072`, `full=capacity` (all clamped to
capacity); ragged lengths are log-uniform. Spec profiles expand original token
positions before integer division by compression ratio. Padding includes
length zero and lengths below top-k to check the -1 convention.

The top-k-only runner passes already expanded per-query row ends as `[R,1]`
and `next_n=1`; cooperative/persistent consume the same flattened lengths as
native `[B,q]`, and DeepSelect's `_row_ends` is a view when there are R values.
Ordinary uniform/ragged profiles do not claim to reproduce a full native
speculative batch merely because the CLI's default `query_len` is8. That
parameter affects the synthetic `spec` profile and the explanatory request
count. A fractional `request_count_if_uniform` is not a legal capture shape.
For a given q, interpret R=B*q only when R is divisible by q and the request
count appears in the independently generated capture grid; otherwise R is a
raw row-boundary probe. Production q1/2/3/4/6/8 is measured separately in the
pipeline runner, which includes its metadata construction.
Ragged visible-history lengths are not an end-to-end benchmark of variable
per-request query counts q_i; the top-k stage only receives flattened row ends.

Warm means repeated access to the same logits. Large inputs need not fit L2;
this is **not** a claim to reproduce GEMM producer locality. Cold explicitly
flushes at least twice reported L2 before each invocation. Full producer/top-k
locality is measured by the separate pipeline runner.

An implementation audit found that FlashInfer's default public timing helper
rounds `2*L2` down to whole MiB. On this GPU2*L2 is258.5MiB, so its automatic
buffer is258MiB. The runner now explicitly requests259MiB for the public span;
the custom kernel trace/host timing use an exact258.5MiB buffer. Flush work is
outside the measured CUPTI correlation interval in both paths. The first
capacity1M modes process had already started, so its cold rows are retained as
an old-protocol control and excluded from formal cold comparisons/CV reruns.
`topk_cold_rounding_fix.sh` repeats that entire cold grid into
`topk_modes_cache_1048576_ceiling.jsonl`; subsequent capacity256K runs use the
corrected source from startup. Per-record `public_cold_flush_bytes` and summary
`cold_flush_meets_explicit_2xl2` distinguish the protocols without relabeling
old data. Warm rows in the earlier file are unaffected.

The production FULL capture host bound is uncompressed `max_model_len`, not
score capacity: this branch's GPU model runner overrides `max_seq_len` during
capture, indexer metadata preserves it, and dense top-k receives it directly.
Main V4 and V4.1 scans explicitly use 1048576 for capture bounds, including
capacity262144/compression4 and capacity524288/compression2. Actual-bound arms
instead pass the maximum effective compressed row length: this is a deliberate
**compressed-bound counterfactual**, not production eager's host argument.
Production eager also preserves the uncompressed context. The added
`raw_actual` arm passes `compressed_max * compress_ratio`, or the saved sample's
exact `raw_context_tokens` metadata. Compression1 actual/raw coincide except
for a saved partial speculative group. The initial dev01
probe used capacity as a synthetic capture bound and is not production evidence.

Real V4 checkpoint samples require an exact raw host override10261: compressed
max2565 multiplied by compression4 gives10260 and loses the residual. The
saved native visible lengths are10254..10261 compressed to
[2563,2563,2564,2564,2564,2564,2565,2565]. This is why the runner supports
`--raw-max-seq-len` and does not infer an exact original length from a floor.

`--prune-unused-bounds` avoids duplicate capture-bound measurements for
all schemes except persistent whose code ignores this host argument; it also
removes exact duplicate bounds. Persistent arms retain both meaningful host
bounds. Initial V4.1 compression1 core additionally kept cooperative's second
bound as a negative control before pruning was extended to that proven-unused
parameter.

## Timing

Each core record uses 100 invocations and three independently interleaved
trials. The runner records all samples, median/p10/p90. Scheme order is
randomized per shape/trial. A separate custom CUPTI trace uses 100 invocations
to retain each kernel's duration and invocation count, plus the sum of GPU
activity durations. Runtime/driver correlation selects the activities belonging
to each invocation; MEMSET and MEMCPY are included as well as kernels.

The public FlashInfer CUPTI helper reports the span from first GPU activity
start to last GPU activity end. An eager span can include a CPU-induced gap
between launches. It is not the sum of kernel durations. Host enqueue and
synchronized wall time are separately recorded. Graph mode captures the entire
scheme into a fresh private pool, then times `graph.replay`; eager mode directly
calls the Python scheme. The public helper is told not to capture again.
Recorded activity intervals can overlap: a few small persistent cases have
activity-duration sums slightly above their first-to-last spans. The sum is
cumulative activity time, not an elapsed-time substitute. Do not label
`span - sum` as exact CPU time; host enqueue and wall measurements provide
separate evidence of launch cost. Whole-scheme GPU latency uses the span.

Smoke/probe records with fewer repeats are labeled by their recorded repeat
count and are excluded from final core statistics. Trial-offset supports
additional reruns without overwriting earlier observations.
Every 200 records a fixed R64/W256K/k2048/UP+PERS graph sentinel is repeated.
Its drift versus the first sentinel is recorded, with a flag above 3%; later
analysis identifies unstable trial groups for targeted reruns.
The online reference resets per process/file. Absolute sentinel medians and
timestamps are retained to check drift across files on the same physical GPU;
different GPUs' absolute samples are checked separately.
`--rerun-unstable` reads completed successful records, identifies configurations
with trial CV>3%, and appends three fresh trials with higher trial numbers.
Original observations are retained. Later runner revisions archive their
complete source in `results/topk_runner_sources.jsonl`, keyed by SHA256;
per-record metadata identifies the revision used. Early C1 started before the
archive option was added; its orchestration later gained CV reruns, pruning of
the unused cooperative host argument, saved-prefix loading, and extra metadata.
The timed operations and correctness comparison are unchanged by those options.

## Allocation and correctness

Input BF16 logits, output indices and warmed shared radix workspace are already
live before resetting peak statistics. FP32-only arms additionally exclude the
prepared FP32 input, which is reported explicitly. Incremental allocated peak
and global reserved delta during fresh private-pool capture are reported
separately. Global reserved delta is an allocator observation, not a precise
sum of tensor sizes and not interchangeable with live peak.

Every row is checked for valid count, bounds, duplicate indices and -1 padding.
Exact torch reference uses all rows for small shapes and an evenly distributed
128-row sample for larger shapes; JSONL explicitly labels sampled reference.
Both index-set disagreement and selected-value-multiset disagreement are
recorded. Equal selected values with different tied indices are tracked
separately. No blanket claim that persistent's selection is exact is made.

## Finalization

Final write target is `cg/indexer-topk-breakdown-results`; the original
`cg/indexer-topk-breakdown` plan branch is read-only. Parent coordinates staging
and push. A pre-commit stash race affected active model-native raw/attempt
files; root restored saved patch rows and missing completed cases were rerun
with identical specs in fresh processes. The recovery manifest preserves
file hashes, retry keys and missing old attempt keys; new raw timing keeps its
actual pid/time. Timed raw keys are unique and model-native groups complete.

Large performance logs remain in this shared worktree and are ignored by git.
The two small offline debug evidence files, `topk_fi_core_focus.log` and
`topk_fi_core_address.log`, are explicitly included by parent. The55,864,329-byte
GPU core stays in shared cache with its hash/path manifest; it is not committed.
