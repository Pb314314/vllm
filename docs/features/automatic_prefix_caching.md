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

## Experimental session-only cache

`SessionScheduler` explores explicit multi-turn reuse without engine-side APC.
It is opt-in and currently supports the Python `LLM` entry point, a single GPU,
text-only full-attention models, synchronous scheduling, and optional `ngram`
speculative decoding. Stateful draft models, hybrid attention, LoRA, connectors,
parallel execution, and snapshot/fork are not supported yet.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    enable_prefix_caching=False,
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
```

Each prompt is a token **delta**, appended after all tokens generated by the
previous turn. The application is responsible for rendering a valid token-level
continuation, including role delimiters and tool messages. Sending full history
again duplicates it. Decoding and re-tokenizing generated text is not equivalent
to preserving its token IDs. `RequestOutput.prompt_token_ids` describes the
submitted delta; the engine retains the full history.

Version 0 creates a session; every successfully completed turn increments its
version. Concurrent writers, stale versions, unknown nonzero versions, changes to
`cache_salt`, and context overflow are rejected without changing the existing session.
Only `n=1` is supported; prompt logprobs and streaming-input requests are rejected.
Cancellation or a failed active turn ends the session. There is no retry/deduplication
protocol: an interrupted caller must establish whether its turn completed before
attempting a continuation.

The session retains the original block lists and token buffers between requests.
Resume transfers ownership of these lists without hashing the history, looking
up APC entries, or acquiring per-block references. The last sampled token may
still need computation. Idle sessions use no running request slot. Worker request
metadata is currently rebuilt on resume, including full token-history staging
and block-table upload. The single-GPU in-process executor passes this metadata
directly to the worker; there is no additional engine-to-worker IPC in that setup.

Under allocation pressure, the oldest idle resident session loses **all** of its
KV. Its token history and version survive, so its next turn replays the history.
Queued turns remain evictable until admission. This policy can over-evict and
cause large latency spikes; it is a baseline for agent-aware eviction experiments.
To compare a finer eviction unit on models with one full-attention KV group, set
`additional_config={"session_cache": {"eviction_policy": "prefix"}}`.
This releases only the missing number of blocks from the oldest idle session's
tail, retaining a valid prefix and replaying its missing suffix on resume.
Multiple idle sessions may be trimmed or fully evicted if necessary. The default
`"whole"` policy remains available for comparison. Neither policy predicts tool
completion or offloads KV; both operate only on the GPU KV pool. Prefix eviction
does not support hybrid or multiple KV groups in this prototype.
There is no automatic sharing between different sessions, including identical
system prompts. Physical KV allocation remains paged.

`max_sessions` (default 1024) bounds logical sessions, including those whose KV
was evicted. There is no TTL in this prototype. Call `LLM.close_session(id)` to
release an idle session's KV and CPU history; closing an active session raises
an error. Two int32 token buffers retain approximately eight bytes per historical
token, excluding object overhead. `reset_prefix_cache()` discards idle session KV
but preserves token histories for recomputation, including after a weight change.
Use the existing draining/reset protocol before replacing model weights.

`num_cached_tokens` reports tokens reused by a resumed turn. APC hash-hit metrics
do not measure this mode. Compare session reuse, recomputed tokens, whole-session
evictions, partial trims, CPU overhead, and latency at the same KV budget. The exploratory
benchmark is `benchmarks/benchmark_session_cache.py`; a synthetic single-session
result is not evidence for general agentic serving performance.

## Limits

APC in general does not reduce the performance of vLLM. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vLLM spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).
