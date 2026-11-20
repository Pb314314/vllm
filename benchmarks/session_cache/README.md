# Session cache diagnostics

These scripts investigate the experimental session-only scheduler. They are not
an agent workload suite or production throughput evidence. Use the same checkout,
GPU, and configuration for both modes. Current defaults use Qwen3-0.6B, one GPU,
TRITON_ATTN, eager execution, synchronous scheduling, and forced output tokens.
Neither Mooncake nor speculative decoding is enabled in these measurements.

## Observed behavior on the current prototype

The [raw observations](observations.json) were captured on 2026-11-20 with
upstream base `54bf2e700e` and prototype `b3a4cd99e1`, on a GB200. These are
focused synthetic diagnostics, not Claude Code traces. Each range below contains
two measured trajectories after a warmup. APC ran before session mode; the small
sample and fixed ordering do not establish a causal speedup percentage.

| Case | APC token reuse | Session token reuse | APC output tokens/s | Session output tokens/s |
| --- | --- | --- | --- | --- |
| One resident session, serial turns | 99.52% | 99.61% | 26.3–38.7 | 39.7–40.6 |
| Four resident sessions, batched turns | 99.52% | 99.61% | 137.9–142.6 | 159.2–160.6 |
| Four new sessions sharing a system prefix | 74.71% | 0% | 25.9–26.1 | 14.2–14.6 |
| Two alternating sessions exceeding KV capacity | 70.19% | 0% | 27.7–28.0 | 13.4–13.9 |

The first two cases had 100% request hit rates in both modes. Shared-prefix
fanout had 75% versus 0%; the pressure case had 100% versus 0%. Thus an APC
request hit rate of 100% under pressure still required recomputing 29.81% of the
prompt tokens. In four measured pressure turns, APC recomputed 19,680 tokens,
whereas session recomputed 66,016 and evicted four whole sessions. APC evicted
1,230 blocks while preserving useful prefix fragments.

Shared-prefix fanout left 1,040 reusable/owned blocks with APC and 4,116 with
session. Both used the same 12 GiB physical KV pool; this is about 4x cache
occupancy, not 4x allocated device memory. In continuation cases, session kept
the partial last block and saved 15 recomputed tokens per turn (80 versus 65).

The separate 16K profile found:

- APC input preprocessing performed 1,034 block hashes; session performed none.
  cProfile counted 12,443 versus 30 input-processing calls. Instrumented input
  time was 4.67 versus 0.15 ms; these are not uninstrumented latency estimates.
- Prefix lookup, reacquiring block references, and releasing them at turn end
  were absent from the resident session continuation. Binding, resume, and
  retention each took tens of microseconds in the instrumented capture.
- Both modes still passed 16,624 historical token IDs and 1,039 block IDs to a
  newly constructed worker request. The worker staged full history and uploaded
  block tables again. In this executor the worker shares the engine process;
  there is no additional engine-to-worker IPC to optimize.
- Both emitted 6,123 GPU kernels over the profiled turn. Summed kernel time was
  72.78 versus 72.77 ms, while first-to-last GPU activity spanned 654 versus
  712 ms under torch profiling. The larger session span came from gaps, not an
  increase in kernel count or measured kernel time. The profile does not establish
  that session bookkeeping caused those gaps.

These findings prioritize explicit shared-root snapshots/forks and a pressure
policy that avoids all-or-nothing recomputation. Session ownership can coexist
with an atomically updated shorter resident prefix; it need not imply evicting
every block at once. Stateful/hybrid models would require recoverable checkpoints
at that boundary. Worker persistence is a separate optimization, and must not
reserve active batch slots for every idle session. Keep async scheduling and
CUDA graph compatibility in the design before drawing production conclusions.

## HBM-only eviction comparison

[Recorded observations](hbm-prefix-observations.json) on `c4e77d9b0e` compare APC,
whole-session eviction, and prefix-preserving session eviction at equal absolute
KV budgets. The workload alternates two distinct 16K histories; each continuation
adds 64 input and 16 forced output tokens. Each configuration has one warmup and
two measured four-turn trajectories. Qwen3-0.6B, GB200 x1, TRITON_ATTN, eager/sync;
no offload, SD, or profiler. Light counters remain enabled. The measured region is
`LLM.generate`, not TTFT. Token reuse is identical across the two trajectories:

| KV budget | APC token reuse | Session whole | Session prefix |
| --- | --- | --- | --- |
| 2 GiB | 13.48% | 0% | 13.48% |
| 3 GiB | 70.19% | 0% | 70.19% |
| 4 GiB | 99.52% | 99.61% | 99.61% |

Request hit rates are 100% for APC/prefix at every budget, illustrating why that
metric alone hides substantial recomputation. Whole-session hit rates are 0% at
2/3 GiB and 100% at 4 GiB. At 3 GiB, prefix eviction reduces four-turn recomputation
from 66,016 to 19,680 tokens, matching APC. Both session variants recompute 260
tokens at 4 GiB versus APC's 320 because they preserve resident partial tails.

Observed output tokens/s ranges (two trajectories, not confidence intervals):

| KV budget | APC | Session whole | Session prefix |
| --- | --- | --- | --- |
| 2 GiB | 15.88–15.90 | 13.14–13.62 | 13.75–14.51 |
| 3 GiB | 24.78–26.65 | 12.82–13.06 | 21.31–26.63 |
| 4 GiB | 40.69–41.45 | 40.60–41.37 | 37.50–38.31 |

The prefix policy recovers partial reuse without content matching, but does not
demonstrate better serving throughput than APC. At 4 GiB neither session policy
evicts or trims anything; their timing difference cannot be attributed to work
in the trim path. These sequential small samples have run variation and were
not randomized: 3 GiB ran APC/whole/prefix; 2/4 GiB ran prefix/APC/whole. More
repetitions and interleaving are needed for small throughput claims. This workload
has no cross-session shared root and no tool-return predictions, so it does not
validate explicit sharing or lifecycle-aware policies.

Reproduce with `measure_reuse.py --regime pressure --kv-cache-mib <MiB>
--mode <apc|session> --output <path>`; select `--eviction-policy whole` or `prefix`
for session. The raw records retain per-turn timings and counters; per-step
scheduler traces are omitted. Full traces are emitted by the reproduction script.

## CPU cost by phase

`measure_cpu_phases.py` separates resident continuation admission, steady decode,
and turn completion using the real synchronous schedulers with fake GPU outputs.
It does not run a worker, GPU forward, IPC, Mooncake, or speculative decoding.

```bash
HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/session_cache/measure_cpu_phases.py \
  --output /tmp/cpu-phases.json
```

[Per-round observations](cpu-phase-observations.json) were captured on prototype
`22bb235a95`, base `54bf2e700e`, with Python 3.12.3 on an ARM Neoverse-V2 host.
Each request has a distinct 16K history, 16 generated setup tokens, and a 64-token
continuation. Three same-process rounds alternate APC/session ordering. Each
round warms 100 decode steps and measures 512; turn completion is separate.
The synthetic KV pool and token budget admit every continuation together, without
eviction. This is not a claim that these batches fit a real GPU configuration.

Values are medians of the three per-round measurements, in milliseconds:

| CPU phase, entire batch | APC, 64 requests | Session, 64 | APC, 256 | Session, 256 |
| --- | --- | --- | --- | --- |
| Continuation admission, once per turn | 202.09 | 4.04 | 804.96 | 15.75 |
| Steady decode schedule, per step | 0.422 | 0.439 | 1.814 | 1.839 |
| Steady decode update, per step | 0.174 | 0.158 | 0.745 | 0.672 |
| Steady decode combined median, per step | 0.606 | 0.603 | 2.572 | 2.517 |
| Steady decode combined mean, per step | 0.626 | 0.608 | 2.668 | 2.553 |
| Completion schedule + update, once per turn | 7.60 | 0.86 | 31.40 | 3.48 |

Independent medians need not add up. The ranges of per-round decode combined
medians were 0.601–0.606 versus 0.601–0.608 ms at 64 requests, and 2.538–2.645
versus 2.503–2.557 ms at 256. Small differences do not establish a stable speedup.
All requests have aligned token positions, so block-boundary costs are synchronized;
the means include those costs, which a median alone can underrepresent.

Admission includes Request/SamplingParams construction, add_request, first
schedule, and first result update. APC receives full history while session receives
a delta. These are aggregate batch CPU costs, not hash-lookup-only time, GPU
prefill latency, TTFT, or cold/chunked-prefill measurements. Prefill schedules
80 versus 65 tokens per request. Decode schedules one token per request in both
modes. Creating fake GPU output objects and model/config setup are outside timing.

The main saving is at turn boundaries. Decode still traverses requests, checks
allocation, constructs scheduler outputs, and updates sampled-token state. APC
also checks for incremental hashes and publishes full cache blocks; session skips
that work. The session allocator currently wraps the normal allocation path, so
removing APC does not remove per-step allocation checks. Faster turn completion
retains owned KV; releasing it is deferred until session close or eviction, which
is outside the completion timing, not eliminated from the session lifetime.

[Upstream #63934](https://github.com/vllm-project/vllm/pull/63934) independently
optimizes no-op decode allocation while retaining APC. Its M4 Pro/async scheduler
numbers are not directly comparable to this synchronous ARM-host probe. That
optimization is absent from this baseline. A later comparison should include an
optimized APC baseline and equivalent no-op handling for session mode. Session
ownership alone does not replace decode-loop optimization or demonstrate a
serving-throughput gain.

### After decoupling capacity checks from APC

Commit `9b854a18f6` lets exact `FullAttentionManager` instances with no incoming
prefix blocks use the existing capacity-only calculation without an APC marker.
Other manager types and prefix adoption retain their previous paths. This fixes
the session path's additional generic capacity checks; it is narrower than the
no-op coordinator optimization in #63934, which remains absent.

The same CPU probe was rerun on `c4e77d9b0e` (which also adds the optional prefix
eviction policy). This no-pressure run uses the default whole eviction policy.
[Per-round records](cpu-phase-fixed-observations.json) use the same methodology
as above. Values are medians across three rounds, in milliseconds per batch step:

| Phase | APC, 64 | Session, 64 | APC, 256 | Session, 256 |
| --- | --- | --- | --- | --- |
| Schedule median | 0.436 | 0.414 | 1.804 | 1.717 |
| Update median | 0.179 | 0.160 | 0.753 | 0.682 |
| Combined median | 0.617 | 0.575 | 2.566 | 2.403 |
| Combined mean | 0.641 | 0.581 | 2.668 | 2.432 |

Combined per-round median ranges were 0.609–0.620 versus 0.574–0.583 ms at 64,
and 2.552–2.567 versus 2.375–2.409 ms at 256. This small CPU-only probe observes
about 6–7% lower combined medians for session; it does not establish a serving
throughput gain. The older capture is not an interleaved before/after experiment.

Validation: the capacity behavior test passes both before and after the change
(this is a performance change, not a previously incorrect capacity result).
Allocator/prefix/scheduler suites: 638 passed. Ordinary/ngram model correctness:
2 passed. Prefix-eviction follow-up: 9 session scheduler tests and 1 real-model
pressure/recompute comparison passed. No new SD performance measurement was made.

## Function and GPU profiling

Run from the repository root, with its dependencies installed:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/session_cache/profile_turn.py \
  --mode apc --output /tmp/session-profile/apc
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/session_cache/profile_turn.py \
  --mode session --output /tmp/session-profile/session
.venv/bin/python benchmarks/session_cache/summarize_profiles.py \
  --root /tmp/session-profile
```

The first two turns warm execution. The third captures separate cProfile files
for frontend, engine input processing, and the engine loop. The fourth captures
a PyTorch CPU/CUDA trace and inclusive host spans. The two modes use identical
token trajectories. Hook classes are diagnostic-only and run in the engine
process; production scheduler behavior is inherited unchanged.

Do not sum nested spans, interpret frontend result waiting as IPC overhead, or
treat instrumented wall time as serving latency. Frontend cProfile does not
cover every background serialization thread. CPU profiling and torch tracing
have different overhead and must not be mixed into one latency decomposition.
GPU busy time is the union of kernel/copy intervals; GPU user annotations include
idle gaps and are not additional kernel execution.

## Reuse, pressure, and throughput counters

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/session_cache/measure_reuse.py \
  --mode apc --regime resident --output /tmp/reuse-apc-resident.json
```

Repeat for `--mode session` and for `--regime pressure`. Run GPU processes
sequentially. `resident` uses 12 GiB KV and covers serial continuation, four
independent sessions with a shared prefix, and batches of four continuations.
`pressure` uses 3 GiB KV and alternates between two distinct 16K histories whose
combined KV exceeds capacity. Each scenario excludes one warmup trajectory and
records two measured trajectories. This is enough to inspect behavior and expose
large regressions, not establish confidence intervals or small speedups.

Profilers are off for throughput measurement. Small scheduler/eviction counters
remain enabled in both modes. The timed region is `LLM.generate`; tokenization,
client history construction, metric snapshots, and setup turns are outside it.
The shared-prefix case includes all four first turns because sharing across new
sessions is the behavior being tested. Other cases measure continuations after
initialization. All cases force 16 output tokens, preserving equal future work.

Reported metrics:

- **Token reuse rate:** sum of reused prompt tokens divided by sum of logical
  full-history prompt lengths. A session's submitted delta is not the denominator.
- **Request hit rate:** fraction of measured turns with at least one reused token.
  A tiny partial hit counts here; use token reuse and recomputation alongside it.
- **Recomputed prompt tokens:** logical prompt tokens minus reused tokens.
- **Throughput:** completed turns or generated output tokens divided by summed
  timed durations. Cached input tokens are not counted as generated throughput.
- **Eviction:** APC reports evicted cached blocks; session mode reports entire
  sessions evicted. These counts have different units and are not interchangeable.
- **Resident blocks:** non-null blocks with a cache hash or an outstanding owner.
  This includes reserved tails and is not total GPU memory allocated by PyTorch.
- **Scheduled tokens:** engine work including prefill and decode. The setup is
  full attention without speculation or connectors, so discarded draft work and
  transferred tokens do not enter this counter.

APC's built-in hash-hit metric does not describe session reuse. In session mode,
the hash lookup is absent; use the common logical-token denominator above.

## Design questions these probes can answer

The next cache-policy comparison is HBM-only: no host KV tier or Mooncake.
Compare whole-idle-session eviction with dropping a block-aligned suffix while
preserving an explicitly recorded valid prefix. The `prefix` policy now supports
one full-attention KV group and frees the allocation shortfall from idle tails.
Measure at identical absolute KV
byte budgets, including retained partial tails. Then compare lifecycle-aware
policies with the same hints available to APC. Record token reuse, recomputation,
resume latency, active batch size, and completed work under a resume-latency
target. A smaller prefill can admit more requests when the token budget binds;
it does not guarantee a larger decode batch when KV capacity or max sequences
binds instead. Lifecycle-aware policies are not implemented yet.

Use `measure_reuse.py --mode session --regime pressure --eviction-policy prefix`
with `--output <path>` to select the new policy. `--kv-cache-mib` sets the absolute
KV budget for either mode. Compare with `--eviction-policy whole` and `--mode apc`
at the same budget. `session_evictions` counts entire idle states evicted under
pressure; `session_trims` counts partial-trim operations, and
`session_evicted_blocks` counts freed physical blocks in both session policies.
Repeated trims of one session count separately. Close/reset are excluded.

The first sharing extension will be an explicit shared-root snapshot with private
child sessions, without automatic sub-session matching. Hybrid follow-up must
inventory attention KV, window state, recurrent/convolution state and worker
buffers, then establish a recoverable boundary before adding eviction or fork.
Existing APC checkpoints and supported sub-block reuse belong in that baseline.
SD must distinguish committed state from speculative scratch; rejected drafts
are never valid shared prefix state in either design. Stateful SD and offload
follow the HBM-only comparison, with transfer and restore costs measured separately.

Explicit continuation can remove history hashing, prefix searches, and per-turn
block reference reacquisition/release. It does not remove physical paging,
committed-token boundaries, or worker input preparation. With the single-GPU
executor, worker metadata is passed in-process; full history is still staged and
block tables are still uploaded on each turn.

Cross-session sharing and partial survival under pressure are separate from hash
lookup overhead. Whole-session eviction may lose a useful prefix that block LRU
would preserve. Consider explicit snapshot/fork for known shared roots, and
compare whole-session offload against retaining a valid prefix under a session
handle. A shorter resident prefix could still be published as one atomic session
state update; hybrid/draft state would need matching recovery boundaries.

Persistent worker state must not pin an execution slot for every idle agent.
Keep logical session ownership separate from active batch slots. Changes to async
scheduling and stateful draft models require completion fences and valid state
boundaries before enabling them. The eager, synchronous profiles here cannot
predict performance with CUDA graphs, async scheduling, MTP/EAGLE, or Mooncake.
