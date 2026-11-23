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
- `--prefix-match-unit` smaller than the Mamba block size

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

`--enable-prefix-caching` also caches Mamba states at block boundaries. To use ReplaySSM without prefix caching, replace this flag with `--no-enable-prefix-caching`. To disable speculative decoding, remove `--speculative-config`.

FlashInfer ReplaySSM has these limits:

- Prefix caching requires BF16 model inputs for its replay rings. The SSM state dtype is independent. The example uses FP16 state with stochastic rounding.
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

The experimental Python frontend recognizes assistant markers from ChatML,
role-specific `<|im_assistant|>`, `]~b]` and `<｜Assistant｜>` tokenizer formats,
without checking model names or template hashes. Markers only propose positions;
they never authorize reuse. It supports streaming and non-streaming responses,
including text tool continuations whose rendered generation header is recognized.
Requests with multimodal content, custom templates, template overrides, or prompt
truncation use the original path. Raw text and token-ID
requests continue to use ordinary APC.

The shortcut seeds standard full-attention cache groups within the normal cache
coordinator. Other groups retain their original matching and alignment rules;
speculative decoding retains its normal proof, lookahead and replay rules.
Tensor/data parallel execution and connectors can use these local shortcuts.
Pipeline and context parallel execution remain on the ordinary path. A local
checkpoint does not bypass remote-cache validation or KV transfer. These are
experimental paths; target-model serving and performance validation are required
before using a newly supported combination in production.

The server derives the previous assistant turn's position from the final
rendered tokens without changing the prompt or retokenizing each historical
turn. It considers the previous assistant's input and output boundaries, with at
most two neighboring complete-block positions per boundary and cache group.
The input boundary is useful when reasoning is rewritten between turns. A match must
have the same chained prefix hash, including the cache salt, and all recorded
blocks must still have their original cache-registration identities. Each group's
lookup continues after its checkpoint, followed by the original cross-group
reconciliation, preserving ordinary APC's hit length. A missing, invalid, or
inapplicable checkpoint falls back to ordinary APC.
Compaction and edited histories therefore require no new client protocol.

Checkpoints are recorded after prefill and at successful generation completion,
using only completed full blocks. The index neither copies KV tensors nor holds
GPU usage references, so allocation and eviction retain their existing rules.
When duplicate physical blocks exist for the same content, publication selects
the copy ordinary APC would return, preserving its physical sharing. The existing
duplicate registry bypasses this extra resolution when no duplicates exist.
When a shorter checkpoint is still valid, registration reuses its canonical
blocks and hash identities, resolving only the added suffix. This preserves
both prefill and output checkpoints without repeating per-block Python work
over the shared history. The flat metadata tuples still require copying their
references; registration is not constant-time in history length.
It is bounded to 1,024 entries and a conservative 64 MiB metadata estimate;
eviction and cache reset can invalidate entries. Partial tails are recomputed.

An additional prototype, `VLLM_TURN_HYBRID_SNAPSHOTS=1`, preserves the exact
completed turn endpoint for full-attention plus prefix-cached Mamba groups. It requires
`VLLM_TURN_PREFIX_CACHE=1`, synchronous execution, no speculative decoding,
no KV events and no ReplaySSM. `MooncakeConnector` can preserve local snapshots
while holding producer blocks until transfer completion. Other connectors,
including `MooncakeStoreConnector`, currently retain ordinary turn lookup/APC.
The transport does not make a decoder's completed snapshot discoverable on the
prefiller; each engine's index describes its own resident state. Other configurations retain
ordinary turn lookup/APC. Validate each deployment before enabling this prototype.
The content key combines the existing hash chain with the remaining token tail
and exact position; a full hash block is required before that tail. Successful
request completion publishes the attention tail and recurrent/conv state at that
same position, excluding in-flight or uncomputed tokens. Existing copy-on-write
copies writable tails on continuation or fork. Entries share the existing index
budget and hold no GPU usage references. A missing group invalidates the joint
hit. This prototype does not yet export exact turn snapshots to a remote Store.

Without the additional hash-reuse switch, this version does not reduce
historical block hashing. A pool generation
avoids repeating block-validity checks while cached contents have not been
invalidated; after an invalidation, lookup checks all recorded block identities.
Block-list materialization and checkpoint bookkeeping remain. Its
benefit depends on history length, hit rate, and registration cost; fewer hash
table queries alone do not establish an end-to-end speedup.

`VLLM_TURN_PREFIX_HASH_REUSE=1`, together with `VLLM_TURN_PREFIX_CACHE=1`,
experiments with reusing the historical block-hash chain. Before initializing a
request, the engine fingerprints the complete candidate prefix's int32 token
buffer with SHA-256, including speculative hash lookahead tokens, and checks the
cache salt. A match supplies the existing hashes; the ordinary hasher computes
the remaining suffix. A miss uses the original hasher. Frontend-supplied hashes
retain priority. Clients need no new fields, and modified histories cannot reuse
an inconsistent hash chain.

This CPU-only memo stores no KV blocks or token buffers. Its separate budget is
1,024 entries and a conservative 64 MiB. Publication after prefill and generation
completion still scans the retained prefix. Input processing and the scheduler
share a metadata lock; fingerprinting happens outside it. Cache reset clears the
memo. `vllm:turn_prefix_cache_reused_hashes_total` records the avoided hashes, and
`vllm:turn_prefix_cache_hash_metadata_bytes` reports its metadata estimate.
The complete history still must be read: this is not constant-time validation.
Candidate fingerprints share one scan of the token bytes, including when the
newest candidate misses and an earlier turn must be checked.
Measure fingerprinting, publication, misses and memory together with serving
performance before enabling it.

`VLLM_DEFER_TURN_PREFIX_CACHE=1`, together with `VLLM_TURN_PREFIX_CACHE=1`,
experimentally defers checkpoint publication. The scheduler captures an immutable
block-reference tuple without pinning GPU memory. Pending work is published while
waiting for an already dispatched model execution, or after the engine becomes
idle. Processing stops between entries when the model output is ready, with at
most eight entries per wait. One entry cannot be interrupted mid-publication.
The queue holds at most 32 entries and an additional conservative 8 MiB metadata
budget. Overflow drops optional publications. Cache invalidation between
capture and publication requires checking the captured block identities;
invalid entries are discarded. Pending or discarded checkpoints fall back to
ordinary APC.
This mode may reduce work on the scheduling critical path, but adds queueing
cost and can miss immediate continuations. It does not guarantee lower total
CPU time or better throughput, and remains disabled by default.

When enabled, `/metrics` additionally exposes:

- `vllm:turn_prefix_cache_queries_total` and `vllm:turn_prefix_cache_hits_total`:
  cache-group probes and valid checkpoint hits. One request can probe several
  groups, and scheduler retries may probe again.
- `vllm:turn_prefix_cache_skipped_block_lookups_total`: group/block lookups actually
  bypassed in the coordinator, excluding the suffix still queried through APC.
  A valid checkpoint can save no queries if another group prevents its use.
- `vllm:turn_prefix_cache_invalidations_total`: stale entries rejected on lookup.
- `vllm:turn_prefix_cache_entries` and `vllm:turn_prefix_cache_metadata_bytes`:
  current index size and conservatively accounted metadata, not GPU usage.

Use normal prefix-cache token metrics alongside these counters: turn hit rate
is not token reuse rate. Initial diagnostics show faster CPU lookup but no
stable serving speedup. See the reproducible CPU and serving diagnostics in
[the session-cache benchmark notes](https://github.com/Pb314314/vllm/blob/ed18d3e4cef5b03cc523367662db080fb390e004/benchmarks/session_cache/README.md).
The [large-model AgentX study](https://github.com/Pb314314/vllm/blob/ed18d3e4cef5b03cc523367662db080fb390e004/benchmarks/session_cache/agentx_results.md)
separates lookup savings from publication costs, ordinary frontend acceleration,
memory pressure, and end-to-end serving results.
The subsequent [overhead iteration](https://github.com/Pb314314/vllm/blob/ed18d3e4cef5b03cc523367662db080fb390e004/benchmarks/session_cache/turn_overhead_results.md)
records lookup/maintenance CPU reductions, cold-request costs and
deferred-publication results.

## Limits

APC in general does not reduce the performance of vLLM. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vLLM spends most of the time generating answers to the queries (e.g. when the length of the answer is long), or new queries do not share the same prefix with any of existing queries (so that the computation cannot be reused).
