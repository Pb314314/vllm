# Automatic Prefix Caching

## Introduction

Automatic Prefix Caching (APC in short) caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part.

!!! note
    Technical details on how vLLM implements APC can be found [here](../design/prefix_caching.md).

## Enabling APC in vLLM

Set `enable_prefix_caching=True` in vLLM engine to enable APC. Here is an example:

[examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py](../../examples/features/automatic_prefix_caching/automatic_prefix_caching_offline.py)

## Hybrid Mamba models

With prefix caching enabled, Mamba state is stored only on the Mamba block grid, so a prefix-cache hit can resume only at a block boundary. `--enable-mamba-shared-prefix-checkpoint` also stores a checkpoint at the shared-prefix junction, the point where an earlier request with the same prefix stopped. Requests whose shared prefix ends inside a block can then reuse it.

This helps when many requests share a long system prompt and then diverge. It is off by default, and takes effect only when all of the following hold:

- prefix caching enabled
- EAGLE/MTP speculative decoding on the Mamba group
- `--prefix-match-unit` smaller than the Mamba block size
- the model does not use multi-module MTP

```bash
vllm serve <hybrid-model> \
    --prefix-match-unit 64 \
    --enable-mamba-shared-prefix-checkpoint
```

`--prefix-match-unit` is required. It sets the granularity at which prefix-cache keys are computed. When unset it defaults to the greatest common divisor of the prefix-cacheable KV cache group block sizes. For hybrid Mamba models that is the block size itself, so no sub-block boundary exists and the flag has no effect.

Choose a value that divides the block size of every prefix-cacheable KV cache group, and that is a multiple of the per-state compression ratio for models that use one, such as sparse MLA. vLLM validates both at startup and names the offending sizes in the error. Read the served block size from the startup log. 64 is a reasonable starting point.

### FlashInfer ReplaySSM

FlashInfer ReplaySSM caches recent inputs to reduce full-state writes during Mamba2 decoding. It is disabled by default. Both Model Runners V1 and V2 support prefix caching with ReplaySSM, including MTP speculative decoding.

This example enables ReplaySSM and prefix caching with three MTP draft tokens:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --tensor-parallel-size 2 \
    --max-model-len 12288 \
    --dtype bfloat16 \
    --mamba-backend flashinfer \
    --mamba-ssm-cache-dtype float16 \
    --enable-mamba-cache-stochastic-rounding \
    --mamba-cache-philox-rounds 5 \
    --use-replayssm \
    --replayssm-buffer-len 16 \
    --enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

`--enable-prefix-caching` selects Mamba cache mode `align`. To use ReplaySSM without prefix caching, replace this flag with `--no-enable-prefix-caching`. This selects mode `none`. To disable speculative decoding, remove `--speculative-config`.

FlashInfer ReplaySSM has these limits:

- `align` requires BF16 model inputs for its replay rings. The SSM state dtype is independent. The example uses FP16 state with stochastic rounding.
- The logical replay buffer cannot exceed 16 tokens. For speculative decoding, it must contain at least `1 + num_speculative_tokens` tokens.
- Pipeline parallelism, microbatching, and KV connectors are not supported.

With `--mamba-backend triton` (for example on ROCm), ReplaySSM also supports speculative decoding with or without prefix caching. It has the same speculative-decoding buffer requirement and parallelism limits, has no 16-token buffer cap, and does not support stochastic rounding.

Prefix reuse still requires a shared prefix with a saved Mamba checkpoint. The matching rules for hybrid Mamba models above also apply.

## Example workloads

We describe two example workloads, where APC can provide huge performance benefit:

- Long document query, where the user repeatedly queries the same long document (e.g. software manual or annual report) with different queries. In this case, instead of processing the long document again and again, APC allows vLLM to process this long document *only once*, and all future requests can avoid recomputing this long document by reusing its KV cache. This allows vLLM to serve future requests with much higher throughput and much lower latency.
- Multi-round conversation, where the user may chat with the application multiple times in the same chatting session. In this case, instead of processing the whole chatting history again and again, APC allows vLLM to reuse the processing results of the chat history across all future rounds of conversation, allowing vLLM to serve future requests with much higher throughput and much lower latency.

## Experimental turn-prefix matching for Chat Completions

`VLLM_TURN_PREFIX_CACHE=1` enables an experimental lookup shortcut for ordinary
`POST /v1/chat/completions` requests containing full conversation history. It is
**disabled by default**. Clients do not supply session IDs, checkpoint IDs, or
cache hints, and the HTTP request and response schemas are unchanged.

```bash
VLLM_TURN_PREFIX_CACHE=1 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --enable-prefix-caching
```

The first version uses the Python API frontend and recognizes the shipped
Qwen2.5-Instruct text chat template.
It supports streaming and non-streaming responses, a single engine/GPU, one
full-attention KV cache group, and local GPU caching. Requests with tools,
multimodal content, reasoning parsers, custom templates, prompt truncation, or
explicit session-cache commands use the original path. Hybrid models,
speculative decoding, parallel execution, and KV connectors also retain their
original paths. Raw text and token-ID requests continue to use ordinary APC.

The server derives the previous assistant turn's position from the final
rendered tokens without changing the prompt or retokenizing each historical
turn. At most two adjacent complete-block boundaries are probed. A match must
have the same chained prefix hash, including the cache salt, and all recorded
blocks must still have their original cache-registration identities. It then
continues ordinary APC lookup after the checkpoint, preserving longer prefix
hits. A missing, invalid, or inapplicable checkpoint falls back to ordinary APC.
Compaction and edited histories therefore require no new client protocol.

Checkpoints are recorded after prefill and at successful generation completion,
using only completed full blocks. The index neither copies KV tensors nor holds
GPU usage references, so allocation and eviction retain their existing rules.
It is bounded to 1,024 entries and a conservative 64 MiB metadata estimate;
eviction and cache reset can invalidate entries. Partial tails are recomputed.

This version **does not reduce historical block hashing**. It trades block hash
table lookups for a linear block-validity check and checkpoint bookkeeping. Its
benefit depends on history length, hit rate, and registration cost; fewer hash
table queries alone do not establish an end-to-end speedup.

When enabled, `/metrics` additionally exposes:

- `vllm:turn_prefix_cache_queries_total` and `vllm:turn_prefix_cache_hits_total`:
  requests probing and hitting the shortcut (scheduler retries may probe again).
- `vllm:turn_prefix_cache_skipped_block_lookups_total`: block lookups bypassed by
  valid checkpoints, excluding the suffix still queried through APC.
- `vllm:turn_prefix_cache_invalidations_total`: stale entries rejected on lookup.
- `vllm:turn_prefix_cache_entries` and `vllm:turn_prefix_cache_metadata_bytes`:
  current index size and conservatively accounted metadata, not GPU usage.

Use normal prefix-cache token metrics alongside these counters: turn hit rate
is not token reuse rate. See the reproducible CPU and serving diagnostics in
[the session-cache benchmark notes](https://github.com/Pb314314/vllm/blob/codex/session-only-cache/benchmarks/session_cache/README.md).

## Experimental session cache with APC

`SessionScheduler` supports ordinary APC requests and explicit multi-turn sessions
in the same engine and KV block pool. Keep `enable_prefix_caching=True` for this
mixed mode. Ordinary requests use APC; requests with the explicit `session_cache`
command retain their state and resume from token deltas. A new session can hit
ordinary APC blocks, and ordinary requests can hit a session's complete cached
blocks. Shared blocks use the same allocator and reference counting.

It is opt-in and currently supports the Python `LLM` entry point, a single GPU,
text-only full-attention models, synchronous scheduling, and optional `ngram`
speculative decoding. Stateful draft models, hybrid attention, LoRA, connectors,
parallel execution, and arbitrary rollback are not supported yet. Explicit prefix
fork is supported for one full-attention KV group.
This is not a replacement for the default scheduler's production feature set.
`enable_prefix_caching=False` retains the session-only experiment, in which
ordinary requests execute but do not get automatic prefix reuse.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    enable_prefix_caching=True,
    async_scheduling=False,
    scheduler_cls="vllm.v1.core.sched.session_scheduler.SessionScheduler",
    additional_config={"session_cache": {"max_sessions": 1024}},
)
tokenizer = llm.get_tokenizer()
for version, new_text in enumerate(["The capital of France is", ". Explain why:"]):
    output = llm.generate(
        {"prompt_token_ids": tokenizer.encode(new_text, add_special_tokens=False)},
        SamplingParams(
            max_tokens=32,
            extra_args={"session_cache": {"id": "agent-1", "version": version}},
        ),
    )[0]
    if output.outputs[0].finish_reason == "error":
        raise RuntimeError(output.outputs[0].stop_reason)
    print(output.outputs[0].text)
llm.close_session("agent-1")

# Ordinary requests on the same engine continue to use APC.
llm.generate("The capital of France is", SamplingParams(max_tokens=8))
```

Each prompt is a token **delta**, appended after all tokens generated by the
previous turn. The application is responsible for rendering a valid token-level
continuation, including role delimiters and tool messages. Sending full history
again duplicates it. Decoding and re-tokenizing generated text is not equivalent
to preserving its token IDs. `RequestOutput.prompt_token_ids` describes the
submitted delta; the engine retains the full history.
The optional `mode` defaults to `"delta"`. A plain `session_id` without the
`session_cache` command does not opt into delta semantics; it follows ordinary APC.
Full-history validation and automatic detection of conversation changes are not
implemented by this scheduler.

Version 0 creates a session; every successfully completed turn increments its
version. Concurrent writers, stale versions, unknown nonzero versions, changes to
`cache_salt`, and context overflow are rejected without changing the existing session.
Only `n=1` is supported; prompt logprobs and streaming-input requests are rejected.
Cancellation or a failed active turn ends the session. There is no retry/deduplication
protocol: an interrupted caller must establish whether its turn completed before
attempting a continuation.

For compaction or edited history, send the complete replacement token input with
`{"id": "agent-1", "version": current_version, "mode": "replace"}`. The ID stays
unchanged. The replacement requires an idle session and its current version; a
successful turn increments that version as usual. It releases the old session's
ownership, lets APC find any still-reusable prefix, and computes the new suffix.
Other owners of shared blocks remain valid. Subsequent delta turns append to the
replacement history. Invalid versions or input validation failures leave the old
state intact; cancellation or failure after admission ends the session, including
when the request is a replacement. No previous-history rollback is promised.

The session retains the original block lists and token buffers between requests.
Resume transfers ownership of these lists without hashing the history, looking
up APC entries, or acquiring per-block references. The last sampled token may
still need computation. With APC enabled, it also retains the hash chain and the
number of blocks already indexed. New inputs and accepted outputs are hashed
incrementally so their complete blocks remain discoverable by ordinary requests.
The normal input path may hash the submitted delta before binding; those hashes
are replaced with hashes chained from the stored history. Compatibility therefore
adds hashing and index maintenance compared with session-only mode.
Idle sessions use no running request slot. Worker request
metadata is currently rebuilt on resume, including full token-history staging
and block-table upload. The single-GPU in-process executor passes this metadata
directly to the worker; there is no additional engine-to-worker IPC in that setup.

Under allocation pressure, the oldest idle resident session loses **all** of its
KV. Its token history and version survive, so its next turn replays the history.
Queued turns remain evictable until admission. This policy can over-evict and
cause large latency spikes; it is a baseline for agent-aware eviction experiments.
To compare a finer eviction unit on models with one full-attention KV group, set
`additional_config={"session_cache": {"eviction_policy": "prefix"}}`.
This drops references from the oldest idle session's tail until enough physical
blocks become free, retaining a valid prefix and replaying its missing suffix on resume.
Multiple idle sessions may be trimmed or fully evicted if necessary. The default
`"whole"` policy remains available for comparison. Neither policy predicts tool
completion or offloads KV; both operate only on the GPU KV pool. Prefix eviction
does not support hybrid or multiple KV groups in this prototype.
There is no automatic sharing between different sessions, including identical
system prompts, when APC is disabled. With APC enabled, new sessions use ordinary
prefix lookup; a resident continuation directly resumes its retained prefix.
Close releases session ownership; it does not flush the shared APC index.
Physical KV allocation remains paged. Idle sessions are retained until pressure,
close, or reset; there is no retention TTL. This can change the hit rate of
ordinary traffic sharing the pool, so mixed-traffic performance needs validation.

### Explicit shared prefixes

Use `LLM.fork_session` to create independent branches of an idle session without
matching token contents. For example, after the first successful turn of a
session named `root`, share only its system prompt, excluding generated tokens:

```python
llm.fork_session("root", "child", version=1, num_tokens=len(system_prompt_token_ids))
output = llm.generate(
    {"prompt_token_ids": new_user_token_ids},
    SamplingParams(
        max_tokens=32,
        extra_args={"session_cache": {"id": "child", "version": 0}},
    ),
)[0]
```

The destination ID must be unused. `version` checks the source version; the child
starts at version 0. `num_tokens` selects a nonempty prefix, or defaults to the
source's entire history, including its generated output. The child inherits the
source's cache namespace. Only fully computed, resident **complete blocks** are
shared by reference. Each child recomputes its partial tail and allocates its own
new blocks. Closing the source does not invalidate a child's shared blocks.
Forking an evicted source copies its token history; missing KV is recomputed.

Fork copies CPU token buffers and block references, so its cost grows with the
selected history. It does not copy the shared GPU KV. An explicit fork chooses
its source directly; it needs no content discovery. APC can separately discover
shared system prompts for new requests when enabled.
Shared blocks occupy HBM once, but releasing one owner's reference may free no
physical memory. Pressure eviction continues until enough blocks become free or
no idle candidates remain. This policy is not yet optimized for shared roots.
Sharing physical KV does not guarantee attention-kernel reuse of HBM reads across
requests; cascade attention is absent from this prototype's upstream baseline.

### Lifecycle and accounting

`max_sessions` (default 1024) bounds logical sessions, including those whose KV
was evicted. There is no TTL in this prototype. Call `LLM.close_session(id)` to
release an idle session's KV and CPU history; closing an active session raises
an error. Two int32 token buffers retain approximately eight bytes per historical
token, excluding object overhead. `reset_prefix_cache()` discards idle session KV
but preserves token histories for recomputation, including after a weight change.
Use the existing draining/reset protocol before replacing model weights.

`num_cached_tokens` reports tokens reused by a resumed turn. APC lookup metrics
cover ordinary lookups, not direct session resumes. Compare session reuse,
recomputed tokens, whole-session
evictions, partial trims, CPU overhead, and latency at the same KV budget. The exploratory
benchmark is `benchmarks/benchmark_session_cache.py`; a synthetic single-session
result is not evidence for general agentic serving performance.

## Limits

APC in general does not reduce the performance of vLLM. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vLLM spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).
