# Session cache diagnostics

These scripts investigate explicit session continuation, with or without APC.
They are not an agent workload suite or production throughput evidence.
Use the same checkout,
GPU, and configuration for both modes. Current defaults use Qwen3-0.6B, one GPU,
TRITON_ATTN, eager execution, synchronous scheduling, and forced output tokens.
Neither Mooncake nor speculative decoding is enabled in these measurements.

## Automatic turn-prefix lookup, with APC retained

Implementation `3b22930a5e` adds the intermediate version for ordinary full-history
Chat Completions requests. It requires no session IDs or new HTTP fields. Enable
`VLLM_TURN_PREFIX_CACHE=1` with APC and the default scheduler; the switch is off by
default. This is separate from the explicit delta/session protocol below. See
[supported inputs and fallback behavior](../../docs/features/automatic_prefix_caching.md#experimental-turn-prefix-matching-for-chat-completions).

The first version optimizes **lookup only**. Previous-turn positions select at
most two checkpoint keys; all recorded block identities are validated, then
ordinary APC continues through the suffix. The existing allocator, GPU reference
counts and eviction rules remain authoritative. Checkpoints contain completed
full blocks and own no GPU references. Historical hashing is unchanged.

The Chat API test caught a real integration issue: the server supplies the
`cohere_format` template default to non-Cohere models too. Rejecting every default
template kwarg silently disabled the shortcut. The implementation accepts that
irrelevant default for the allowlisted Qwen template and retains conservative
fallback for other overrides. The test asserts actual shortcut hits, not just
successful generation.

### Validation

[Raw validation output](turn-prefix-validation.txt) covers:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/v1/core/test_prefix_caching.py tests/v1/core/test_scheduler.py \
  -k 'prefix or session or preempt_during_execution' -q
# 309 passed, 278 deselected in 107.38s (0:01:47)
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/renderers/test_hf.py tests/v1/core/test_prefix_caching.py -k turn_ -q
# 14 passed, 328 deselected in 12.54s
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=. \
  LD_LIBRARY_PATH="/usr/local/cuda/compat:$LD_LIBRARY_PATH" \
  CUDA_VISIBLE_DEVICES=2 VLLM_GPU_SYNC_CHECK=error VLLM_BATCH_INVARIANT=1 \
  .venv/bin/python -m pytest \
  tests/entrypoints/openai/chat_completion/test_chat_completion.py \
  -k turn_prefix -v -s
# 1 passed, 9 deselected in 124.23s (0:02:04)
```

Real-model validation uses Qwen2.5-1.5B-Instruct, GB200, TRITON_ATTN,
eager execution and the default async scheduler with chunked prefill. It compares
identical rendered tokens, cached-token counts and output/logprobs with the flag
off/on. It covers streaming, non-streaming, branches, edited history, compaction,
raw token input and reset. Core tests additionally exercise recycled block IDs,
an evicted middle block, cache salt, concurrent ownership, in-flight cancellation,
stale preemption output and metadata limits. This is output-correctness evidence;
it is not a task-accuracy evaluation or coverage of SD, hybrid or connectors.

Pre-commit passes, including mypy and test tethering. Existing Buildkite
Chat Completions, core and renderer suites select the tests. The feature test
explicitly selects the Python frontend, including when the surrounding CI job
uses Rust. Rust callers continue through ordinary APC.

### CPU observations

[Per-round raw CPU data](turn-prefix-cpu-observations.json) uses real synchronous
schedulers and simulated GPU outputs on ARM Neoverse-V2, pinned to logical CPU
16. The scheduler fixture uses Qwen3-0.6B configuration; no model executes. Each
request has a distinct 16K history and appends 64 tokens. Three rounds rotate
the mode order, with 32 warmup and 128 measured decode steps. Capacity fits all
synthetic requests without eviction. This does not imply a real GPU fits the
same batch.

Median milliseconds for the **whole batch of 64**:

| Phase | APC | Turn lookup + APC | Ordinary requests with switch on |
| --- | ---: | ---: | ---: |
| Construct request, including hashing | 141.96 | 143.53 | 141.83 |
| First schedule, including lookup and allocation | 62.15 | 20.77 | 62.91 |
| First result update, including checkpoint registration | 0.87 | 5.27 | 0.83 |
| Total continuation admission | 205.41 | 170.22 | 204.54 |
| Final decode schedule + result update | 7.55 | 12.96 | 7.63 |
| Steady decode schedule + result update, per step | 0.659 | 0.679 | 0.666 |

All modes schedule the same 5,120 prefill tokens. Admission ranges across rounds
are 202.49–206.37, 166.17–170.68 and 203.83–207.14 ms. Decode ranges are
0.658–0.669, 0.678–0.685 and 0.653–0.690 ms. The shortcut therefore trades lookup
time for registration work and shows a small decode cost in this capture.
Admission plus completion improves about 14% at batch 64; initial cold-turn
registration, frontend work, IPC and workers are outside this timing boundary.

Batch-1 measurements are noisy: median admission is 3.39, 3.75 and 4.62 ms,
including substantial variation in the unchanged hashing phase. They do not
establish low-concurrency non-regression. Do not extrapolate the batch-64 result
to all request sizes.

```bash
PYTHONPATH=. taskset -c 16 .venv/bin/python \
  benchmarks/session_cache/measure_cpu_phases.py \
  --modes apc turn-apc plain-on-turn --batches 1 64 --context 16384 \
  --warmup 32 --steps 128 --rounds 3 --output /tmp/turn-prefix-cpu.json
```

[A separate cProfile capture](turn-prefix-cpu-profile.txt) shows the remaining
linear block-validity checks, reference acquisition and checkpoint registration.
Profiling perturbs timings and includes the seed trajectory; it is not the source
of the unprofiled table. [Frontend observations](turn-prefix-frontend-observations.json)
measure 50 warmed HF template/tokenization and boundary-extraction calls on the
same history. At about 4K/16K/64K tokens, boundary extraction medians are
13/72/261 microseconds versus 5.4/22.5/98.0 ms for HF render + tokenize. These
exclude API validation and thread-pool queuing. Delimiter checks still read
historical text; they are not constant-time history verification.

### Serving observations and unresolved performance concern

[Raw per-request serving observations](turn-prefix-serving-observations.json)
use Qwen2.5-1.5B-Instruct on one GB200, TRITON_ATTN, eager/default async execution,
16-token blocks and identical 12 GiB KV pools. Both server processes stay
resident on the same GPU, but measured workloads execute sequentially. SD,
Mooncake and batch-invariance mode are off. This is a shared host.

Eight concurrent conversations each execute eight dependent turns, with about
16K initial tokens and 16 output tokens per turn. The initial context is synthetic
repeated text, not a coding-agent trace. Every measured run includes
its cold first turns after a cache reset and follows one warm trajectory. An
interactive run uses real generated answers; subsequent runs replay exactly the
same full requests. All six replay input digests match. Mode order is off/on,
on/off, off/on across three pairs.

| Metric | APC | Turn lookup + APC |
| --- | ---: | ---: |
| Requests/s, three runs | 8.58, 9.01, 9.45 | 8.91, 9.00, 7.92 |
| Median continuation TTFT per run, ms | 113.51, 101.47, 99.14 | 108.89, 107.75, 127.02 |
| Server process tree CPU seconds per run | 9.33, 8.94, 8.62 | 9.14, 9.05, 9.91 |
| Reused / queried tokens, every run | 925,552 / 1,058,016 | 925,552 / 1,058,016 |
| Eligible continuation checkpoint hits | N/A | 56 / 56 |

Token reuse is **87.48% for both**. Every enabled run skips 57,840 block lookups;
the index ends with 120 entries and a 10.11 MiB metadata estimate. The separate
interactive run also hits all 56 continuations. Full TTFT p50/p95/p99 and raw
request timings are included in the data; cold first turns dominate overall p95.

**There is no stable serving speedup in this capture.** One enabled run is
materially slower. Hashing, rendering and model execution remain; fewer cache
lookups alone are insufficient. CPU admission benefits are not evidence that
overall CPU use, TTFT or throughput improve. Keep this experimental and default
off; isolate the variation and collect profiles of real serving before deciding
whether to optimize hashing, reduce registration further, or expand support.

Start each variant with the same command, changing only the feature flag and
port (the capture used ports 18635/off and 18634/on):

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 \
  VLLM_TURN_PREFIX_CACHE=0 .venv/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 --attention-backend TRITON_ATTN --enforce-eager \
  --max-model-len 20000 --max-num-seqs 16 --block-size 16 \
  --kv-cache-memory-bytes 12884901888 --enable-prefix-caching \
  --enable-prompt-tokens-details --port 18635
```

Once both `/health` endpoints are ready, capture one interactive trajectory on
the enabled server. Then replay that file on each server, alternating their order
across independent runs. `--server-pid` records API and descendant engine CPU
time; supply the corresponding server PID. The benchmark's development-only
reset endpoint requires `VLLM_SERVER_DEV_MODE=1` as above.

```bash
PYTHONPATH=. .venv/bin/python benchmarks/session_cache/measure_turn_prefix_api.py \
  --url http://127.0.0.1:18634 --sessions 8 --turns 8 --context 16384 \
  --rounds 1 --output /tmp/turn-prefix-interactive.json
PYTHONPATH=. .venv/bin/python benchmarks/session_cache/measure_turn_prefix_api.py \
  --url http://127.0.0.1:18635 --sessions 8 --turns 8 --context 16384 \
  --rounds 1 --replay /tmp/turn-prefix-interactive.json \
  --output /tmp/turn-prefix-replay-off.json
```

## APC compatibility

`SessionScheduler` now also accepts `enable_prefix_caching=True`. Ordinary
requests use APC, explicit delta turns resume retained state, and both use the
same physical block pool. New sessions can hit ordinary requests' prefixes;
ordinary requests can hit completed session blocks. Sessions retain their hash
chain and cache-registration cursor, hashing new tokens rather than reconstructing
history hashes on each turn. This adds incremental hashing and index maintenance
compared with the APC-disabled experiment. The input path can also hash a delta
before the scheduler replaces those hashes with the stored chain.

`mode="replace"` on a versioned session command replaces history under the same
ID, for explicit compaction. It reuses eligible APC prefixes and keeps versions
monotonic. It does not infer compaction from a delta. Ordinary `session_id` hints
do not enable the delta protocol. See the
[feature documentation](../../docs/features/automatic_prefix_caching.md#experimental-session-cache-with-apc).

The CPU phase script compares four modes: `apc` (default scheduler), `session`
(APC disabled), `session-apc` (explicit continuation with APC enabled), and
`apc-on-session` (ordinary requests on the session scheduler). Modes rotate across
rounds; it measures real scheduler calls with simulated GPU outputs, not serving
throughput. Older captures below use the APC-disabled prototype unless stated.

### Compatibility validation and CPU cost

Implementation `24d9e721c9` passes 18 selected scheduler tests and 9 real-model
session tests, including both APC settings, ngram SD, mixed batches, explicit
forks, pressure replay and same-ID replacement. The mixed-batch test was rerun
after adding a cold-output comparison for the replacement itself: 1 passed.
The [validation output excerpts](apc-compat-validation.txt) includes the expected
pre-fix failure. Model tests use Qwen3-0.6B, GB200, TRITON_ATTN, eager/sync and
batch invariance; they establish token-output correctness on these cases, not
task accuracy or production feature parity.

[CPU observations](apc-compat-cpu-observations.json) compare four modes on the
same ARM Neoverse-V2 host. There are three rounds with rotated ordering per batch,
100 warmup and 512 measured decode steps per round. Each request has a distinct
16K history and appends 64 tokens. The synthetic KV capacity admits all requests
without eviction; this does not establish that the batch fits a real GPU.
Values below are medians of per-round measurements in milliseconds, for the
entire batch. Admission includes request construction, add_request, first schedule
and result update. It excludes tokenization, worker work and GPU compute.

| Mode | Admission, 64 requests | Decode per step, 64 | Admission, 256 | Decode per step, 256 |
| --- | --- | --- | --- | --- |
| APC, default scheduler | 197.59 | 0.606 | 928.19 | 2.948 |
| Session, APC disabled | 4.25 | 0.575 | 17.85 | 2.378 |
| Explicit session with APC | 5.78 | 0.635 | 27.28 | 2.671 |
| Ordinary APC on session scheduler | 196.84 | 0.639 | 866.76 | 2.667 |

Explicit continuation retains the admission benefit, but compatibility is not
free. It restores incremental hashing and cache-index maintenance. At 64 requests,
decode medians across rounds range from 0.605–0.618 ms for default APC,
0.630–0.637 ms for explicit session with APC, and 0.633–0.639 ms for ordinary
requests on the session scheduler. This is a measured CPU regression to investigate,
not evidence of zero overhead. At 256 requests the ranges overlap: respectively
2.529–3.016, 2.652–2.757, and 2.640–2.700 ms; the noisy medians do not establish
a decode improvement. No profiler, GPU model, SD or Mooncake is used here.

```bash
HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/session_cache/measure_cpu_phases.py \
  --batches 64 256 --context 16384 --warmup 100 --steps 512 --rounds 3 \
  --output /tmp/apc-compat-cpu.json
```

### APC-compatible resident GPU diagnostic

[Per-turn observations](apc-compat-gpu-observations.json) on the same implementation
use Qwen3-0.6B / GB200 / TRITON_ATTN / eager synchronous, a 12 GiB KV pool and
16 forced output tokens per request. There is one warmup and two measured
trajectories per case. APC runs first, then session with APC; no SD or Mooncake.
Times cover `LLM.generate` with lightweight counters and exclude setup turns,
except for the shared-prefix case, which includes the first cold request.

| Case | APC token reuse | Session + APC token reuse | APC output tokens/s | Session + APC output tokens/s |
| --- | --- | --- | --- | --- |
| Serial continuation | 99.52% | 99.61% | 33.87–38.24 | 42.61–42.63 |
| Four new sessions with shared system prefix | 74.71% | 74.71% | 24.47–25.66 | 26.89–27.80 |
| Four sessions, batched continuation | 99.52% | 99.61% | 133.20–138.73 | 120.25–135.62 |

Request hit rates are 100% for both continuation cases and 75% for shared-prefix
fanout in both modes. Shared-prefix resident/reusable blocks are 1,040 versus
1,044: APC-compatible sessions recover automatic sharing, with four private
partial tails. All configurations have the same 7,020 usable physical blocks.
No explicit fork or caller-provided shared-root reference is used in this capture.

These results do not establish throughput non-regression: the batched case includes
a slower session trajectory, and the sample size, fixed ordering and run variation
prevent attributing the difference. Serial admission improvements do not guarantee
batched serving gains. Mixed ordinary/session traffic under capacity pressure,
async scheduling and CUDA graphs still need evaluation.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" VLLM_GPU_SYNC_CHECK=error \
  .venv/bin/python benchmarks/session_cache/measure_reuse.py \
  --mode apc --regime resident --output /tmp/apc-compat-baseline.json
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" VLLM_GPU_SYNC_CHECK=error \
  .venv/bin/python benchmarks/session_cache/measure_reuse.py \
  --mode session-apc --regime resident --output /tmp/apc-compat-session.json
```

## Historical session-only observations

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

## Explicit shared-root comparison

[Recorded observations](shared-root-observations.json) on `760248a7ee` compare
APC, independent sessions, and explicit shared-root forks. Four serial requests
share 16,384 input tokens, each adding a distinct 64-token suffix and 16 forced
output tokens. The first request is cold in every mode; later forks select only
its shared 16K prefix. Each mode has one warmup and two measured trajectories.
All use a 12 GiB KV pool, Qwen3-0.6B / GB200 x1 / TRITON_ATTN / eager synchronous,
no SD, offload, or profiler. Sequential order was APC, fork, independent sessions.

| Mode | Token reuse | Request hits | Recomputed input tokens | Resident physical blocks | Output tokens/s |
| --- | --- | --- | --- | --- | --- |
| APC | 74.71% | 75% | 16,640 | 1,040 | 26.66–28.67 |
| Independent sessions | 0% | 0% | 65,792 | 4,116 | 11.55–14.07 |
| Explicit shared root | 74.71% | 75% | 16,640 | 1,044 | 28.31–28.32 |

The fork path recovers APC's reuse without content matching. Its four extra
blocks retain the private partial tails. All modes allocate the same GPU KV pool;
these block counts describe its occupied/reusable contents, not process VRAM
allocation. Fork times are included: 0.632–0.677 ms per fork across six measured
forks, totaling 1.940–1.950 ms per four-request trajectory. This includes the
utility round trip, CPU token copies and block-reference acquisition.
The observed throughput ranges overlap between APC and fork. Two fixed-order
trajectories do not establish a speedup over APC. This serial probe measures
prefill reuse and occupancy, not cascade attention or batched decode bandwidth.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_GPU_SYNC_CHECK=error PYTHONPATH="$PWD" \
  .venv/bin/python benchmarks/session_cache/measure_reuse.py \
  --mode session --regime resident --scenario shared_prefix --shared-root \
  --output /tmp/shared-root-session.json
```

Omit `--shared-root` for independent sessions; use `--mode apc` without that flag
for APC. The same script records per-turn timings, fork costs, physical blocks
and reuse. The fork implementation and accounting passed 12 session scheduler
tests and 6 session-related real-model tests, including batch-invariant divergent
branches, parent closure, partial-tail replay and existing ngram continuation.

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

The first sharing extension is `LLM.fork_session`: full resident blocks are shared
by reference, CPU history is copied, and each child replays any partial tail.
The caller selects a source version and prefix length; there is no automatic
sub-session matching. Both eviction policies count actual physical capacity
released, not discarded references. They are not yet optimized for shared roots:
trimming one owner's reference can lose its reuse without freeing any HBM.

Use `measure_reuse.py --mode session --regime resident --scenario shared_prefix
--shared-root --output <path>` to compare explicit sharing with APC and independent
sessions (omit `--shared-root`). All three paths include the first cold request
and four outputs in their timed totals. The fork path creates later sessions from
the first session's 16K prefix, excluding its private suffix and generated tokens.
Each fork utility call, including token-buffer copying and reference acquisition,
is included in its request timing and reported separately as `fork_seconds`.
`resident_blocks` counts distinct physical blocks in all modes, including reusable
unreferenced APC blocks. This probe measures prefill reuse and HBM occupancy,
not shared attention-read bandwidth.

The rebased upstream baseline includes removal of cascade attention in
[PR #61126](https://github.com/vllm-project/vllm/pull/61126). Shared physical block
IDs could support a future common-prefix attention path, but this prototype does
not implement or measure one. Shared allocation alone does not ensure the GPU
reads each prefix only once across requests.

Hybrid follow-up must
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
would preserve. Measure explicit fork for known shared roots, and
compare whole-session offload against retaining a valid prefix under a session
handle. A shorter resident prefix could still be published as one atomic session
state update; hybrid/draft state would need matching recovery boundaries.

Persistent worker state must not pin an execution slot for every idle agent.
Keep logical session ownership separate from active batch slots. Changes to async
scheduling and stateful draft models require completion fences and valid state
boundaries before enabling them. The eager, synchronous profiles here cannot
predict performance with CUDA graphs, async scheduling, MTP/EAGLE, or Mooncake.

## Routing and lifecycle follow-up

Session affinity already exists in routers. For example, Production Stack offers
[session-ID routing](https://github.com/vllm-project/production-stack/blob/main/src/vllm_router/README.md)
and [load-aware routing](https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/loadaware-routing.html)
that balances cached-prefix benefit against in-flight requests. Dynamo's
[KV-aware router](https://docs.nvidia.com/dynamo/latest/knowledge-base/concepts/system-architecture/kv-aware-routing)
combines cache events with active prefill and decode load; its
[filtering and queueing](https://docs.nvidia.com/dynamo/knowledge-base/modular-components/router/router-filtering)
controls admission independently of candidate scoring.

The proposed session contract would let a known continuation locate an owner and
version directly. It still needs resident length, physical occupancy, reclaimable
capacity, queued/active work, and the new input length to estimate cost. Session
count or turn number alone cannot predict work. Missing state on another worker
requires replay or transfer; a large cluster does not pool all GPU HBM for free.
Shared roots need explicit IDs or frontend discovery, not per-turn engine hashing.
The prototype has idle/active ownership and explicit close, but no router
integration, tool-state hints, workload predictor, or migration protocol.

Compare APC, APC with identical lifecycle hints, and session ownership with
explicit roots under the same HBM budget. Vary context lengths, tool-return
bursts, shared-root popularity, and placement skew. Record recomputed tokens,
physical KV, resume P95, and completed work within a latency target. This separates
benefits from new information and admission policy from benefits of removing APC.

## Regression audit and acceptance gates

The next objective is to find regressions and choose the least complex remedy
that preserves performance in the supported workload. Keep two comparisons:
identical engine settings to isolate cache mechanics, and production-capable
settings to reveal missing async/graph/SD/parallel support. The current eager,
synchronous results establish neither production parity nor global superiority.
Report CPU per turn/step, token reuse, actual physical capacity, TTFT/resume P95,
TPOT and completed work at a latency target. Agree on workload-specific noise
bounds before treating a timing difference as a regression.

| Risk | Evidence/status | Candidate remedy and cost |
| --- | --- | --- |
| Unrelated sessions with identical prefixes | Independent sessions demonstrably lose reuse; explicit roots recover the measured shared-system case | Frontend root registration/discovery plus engine handles. Without an explicit relationship or content discovery, arbitrary APC hits cannot be preserved. |
| A source loses its cache handle while a descendant still owns the prefix | Reproduced below | Coordinate shared-tail eviction or keep an explicit shared-prefix residency record. Prefer existing ownership metadata; measure pressure-path scan cost before adding an index. |
| Excessive eviction/recomputation | Whole-session eviction lost all reuse in the 2/3 GiB sweep; prefix eviction recovered APC's token reuse but not a stable throughput advantage | Reclaim actual physical capacity and preserve valid prefixes. Shared ownership means dropping one reference may free nothing; group bookkeeping has a cost. |
| Finish/close/restart loses future reuse | Close immediately releases this session's references; no anonymous cache or handle recovery exists | Explicit reusable-root lifetime, bounded retention and recovery protocol. These retain metadata/state and require application lifecycle signals. |
| Large branching fanout | Fork copies CPU history and references; only an idle source is supported | Measure branch creation and host RSS. Shared immutable CPU history or active snapshots add ownership and execution-fence complexity. |
| Partial boundaries or edited histories | Fork currently shares complete blocks only; main has partial-hit/CoW machinery in supported configurations. Deltas cannot express arbitrary edits in place | Explicit prefix snapshots, bounded replay and possibly tail CoW. Price GPU copy/replay and metadata costs against the saved prefill. |
| Idle-state memory and smaller effective batches | Two persistent int32 histories per session plus retained tails; shared-root probe holds 1,044 blocks versus APC's 1,040 | Bound logical state and reclaim idle capacity. Measure active-batch size and preemptions; do not interpret raw cache usage identically across modes. |
| Production feature gaps and placement | Async is disabled; graphs are unverified by the current eager measurements; stateful SD, hybrid, parallel and connectors are rejected | Restore capabilities independently, then compare production configurations. Track state bytes/copies, migration/replay cost, and locality versus load balance. |

APC content matching handles branch-like requests without requiring a logical
session tree: matching prefixes can reuse available indexed blocks, followed by
independent suffix allocation. Explicit session fork replaces discovery with a
declared source/version/boundary; it still needs refcounts, immutable shared
blocks and mutable-state isolation. Reduced discovery work is not evidence that
APC sharing was unsafe or that total system complexity has already decreased.
Lifecycle policy must also be compared with APC receiving the same lifecycle hints.

Restoring cascade attention is outside these gates: upstream removed it in
[PR #61126](https://github.com/vllm-project/vllm/pull/61126).
The pre-removal code supported ordinary full attention; its graph
restriction was CUDA Graph compatibility, including FULL mode. Physical prefix
sharing must remain effective, but cross-request attention-read optimization is
a separate feature, not a regression introduced by disabling APC here.

### Proposed common representation for system roots and forks

Use the same explicit shared-prefix representation for a system template and an
agent branch. Frontend registration selects the exact token prefix and cache
namespace, optionally doing content discovery once when creating a root. Engine
continuations use handles; the engine does not rediscover their contents.
`fork(source, version, boundary)` references committed immutable prefix blocks
and gives the child a private writable tail. Both cases use the same allocator,
reference counting and valid-computation boundaries.

Separate logical identity, shared ownership and residency. A registered root
must not depend on the source request remaining alive, and a missing parent's
local block list must not hide still-resident shared state. Eviction should update
the shared residency boundary consistently, reclaim physical capacity and leave
the logical handle usable for bounded replay. Closing one session drops its
ownership; it does not invalidate other owners. A live handle is not a promise to
pin HBM indefinitely.

This is a design target, not implemented functionality. First evaluate coordinated
tail reclamation using existing references before adding a persistent shared-root
directory. Keep complete-block sharing and private-tail replay as the small first
implementation; compare with main's supported partial-hit/CoW path before deciding
whether another tail-copy mechanism is justified. CPU-history sharing and active
snapshots also require evidence: current fork copies history and waits for an
idle source. Count frontend bookkeeping and recovery logic in complexity reviews,
not only the engine code removed.

### Reproduced: eviction loses a reusable source prefix

On `8564031d94`, a four-usable-block pool and the `prefix` policy can discard a
parent's references before reclaiming a shared block from its child. The child
retains one physical block, but a subsequent fork through the parent cannot
discover it. Identical 17-token inputs schedule 17 versus 1 token depending on
which handle supplies their first 16 tokens. No GPU or model weights are used;
this is evidence of extra scheduled prefill, not a latency measurement or an APC
throughput comparison. The implementation is unchanged by this audit.

```bash
HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" .venv/bin/python - <<'PY'
from tests.v1.core.utils import create_scheduler
from tests.v1.core.test_scheduler import _session_turn
from vllm.v1.core.sched.session_scheduler import SessionScheduler
from vllm.v1.outputs import ModelRunnerOutput

s = create_scheduler(
    scheduler_cls=SessionScheduler, block_size=16, num_blocks=5,
    additional_config={"session_cache": {"eviction_policy": "prefix"}},
)
for sid, length in [("root", 33), ("c", 17), ("d", 1)]:
    s.add_request(_session_turn(sid, sid, 0, [1] * length))
    out = s.schedule()
    s.update_from_output(out, ModelRunnerOutput(req_ids=[sid], sampled_token_ids=[[2]]))
    if sid == "root":
        s.fork_session("root", "child", 1, num_tokens=32)
print("retained", {sid: state.num_computed_tokens for sid, state in s.sessions.items()})
s.close_session("c")
s.close_session("d")
s.fork_session("root", "from-root", 1, num_tokens=16)
s.fork_session("child", "from-child", 0, num_tokens=16)
for sid in ["from-root", "from-child"]:
    s.add_request(_session_turn(sid, sid, 0, [3]))
out = s.schedule()
print("scheduled", out.num_scheduled_tokens)
print("same_input", list(s.requests["from-root"].prompt_token_ids)
      == list(s.requests["from-child"].prompt_token_ids))
s.update_from_output(out, ModelRunnerOutput(
    req_ids=["from-root", "from-child"], sampled_token_ids=[[4], [4]],
))
for sid in ["root", "child", "from-root", "from-child"]:
    s.close_session(sid)
print("free_blocks_after_close", s.kv_cache_manager.block_pool.get_num_free_blocks())
PY
```

Observed counts (labels shortened from the original capture):

```text
retained {'root': 0, 'child': 16, 'c': 17, 'd': 1}
scheduled {'from-root': 17, 'from-child': 1}
same_input True
free_blocks_after_close 4
```

Prioritize this shared-ownership/pressure case, then cold shared-root bursts,
fanout and close/return patterns. After the ownership and reclaiming behavior is
sound, validate graph/async parity before extending to stateful SD, hybrid state
and distributed recovery. Avoid using an unmeasured policy change or an oracle
for future tool returns as evidence that sessions intrinsically outperform APC.
