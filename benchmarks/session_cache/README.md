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
