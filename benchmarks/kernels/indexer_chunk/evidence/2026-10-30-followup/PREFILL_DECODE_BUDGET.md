# Shared prefill/decode budget: code audit, not performance validation

The user proposed sharing a 2 GiB workspace between prefill and decode. No production code is changed and no new experiment is launched in this consolidation.

## Current accounting differs

| Path | Setting | Current default | Budget interpretation |
| --- | --- | ---: | --- |
| Prefill | VLLM_SPARSE_INDEXER_MAX_LOGITS_MB | 512 MiB | Planner bounds M*N*4, representing FP32 logits; also constrained by the gathered-K workspace |
| PR2 decode | VLLM_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB | 2048 MiB | Planner bounds Q rows *configured compressed K capacity* 2, representing BF16 logits payload |

[Prefill planner](https://github.com/vllm-project/vllm/blob/d51f40aeffe014e4ee1f77759b22501893f84d5e/vllm/v1/attention/backends/mla/indexer.py#L1369) and [decode planner](https://github.com/vllm-project/vllm/blob/d51f40aeffe014e4ee1f77759b22501893f84d5e/vllm/model_executor/layers/sparse_attn_indexer.py#L337).

On the audited SM100 prefill auto path, DeepGEMM returns BF16 and the caller explicitly invokes as_fp32_logits. The conversion allocates an independent FP32 tensor. At a 2 GiB FP32 payload limit, BF16 plus FP32 can therefore reach approximately 3 GiB during conversion before alignment, candidate scratch, and K gathering. The planner is conservative for some compact-output shapes; this is an accounting bound, not a claim every batch allocates 3 GiB. Explicit DeepSelect has different conversion behavior. [Prefill conversion](https://github.com/vllm-project/vllm/blob/d51f40aeffe014e4ee1f77759b22501893f84d5e/vllm/model_executor/layers/sparse_attn_indexer.py#L936), [allocation](https://github.com/vllm-project/vllm/blob/d51f40aeffe014e4ee1f77759b22501893f84d5e/vllm/model_executor/layers/sparse_attn_indexer.py#L314).

## Reuse is conditional, not guaranteed by matching settings

Prefill obtains gathered K/scales from WorkspaceManager; the logits are returned by DeepGEMM instead of being passed an explicit shared output buffer. The paged decode wrapper likewise returns its own logits tensor. The public wrappers here have no caller-provided logits output argument. Equal budget numbers do not establish storage aliasing. [Gather workspace](https://github.com/vllm-project/vllm/blob/d51f40aeffe014e4ee1f77759b22501893f84d5e/vllm/model_executor/layers/sparse_attn_indexer.py#L812), [producer wrappers](https://github.com/vllm-project/vllm/blob/d51f40aeffe014e4ee1f77759b22501893f84d5e/vllm/utils/deep_gemm.py#L671).

The caller processes prefill before decode and deletes each prefill chunk's logits before advancing. That supports potential reuse for non-overlapping allocations. However, ordinary eager allocations, graph-private allocations, captured addresses, and separate ubatch/lane owners must be checked before claiming one physical backing allocation. The existing profiling estimate uses max(prefill estimate, decode estimate), which is an accounting assumption about peak, not proof of actual allocator reuse. WorkspaceManager itself uses independent ubatch/lane ownership for overlap safety.

## Why not simply maximize the chunk

Larger chunks reduce repeated scheduling and launches but can reserve more memory and leave less KV capacity. Once latency improvements flatten, an even larger workspace can worsen capacity or OOM behavior without meaningful step-time benefit. Prefill may also remain bounded by gathered-K workspace, so increasing the logits budget need not reduce chunk count. A budget available on one deployment may not fit another.

The useful objective is an acceptable measured performance/memory tradeoff under the intended model and topology, not the largest allocation that succeeds on one isolated test. The current 2 GiB decode default has not yet passed representative long-output serving validation.

## Proposed validation after reviewing completed results

1. Separate policy unification from actual storage reuse. First compare current prefill 512 MiB / decode 2 GiB with prefill 1/2/4 GiB while holding decode at 2 GiB. Verify prefill actually uses the indexer rather than a dense-attention bypass.
2. For promising settings compare decode 2 versus 4 GiB. Include prefill-only, decode-only, and mixed batches; FULL decode graph and supported eager execution; TP4 and DP4/TP1/EP4.
3. Measure TTFT, decode ITL/TPOT, throughput, live allocated peak, allocator reserved memory, aggregate graph pools, and remaining KV blocks. Distinguish equal-KV timing from equal-total-GPU-memory capacity benefits.
4. For any explicit shared-buffer implementation, verify non-overlapping lifetimes, graph address stability, owner lanes/streams, mixed-batch reuse and numerical correctness. A memory snapshot or allocation trace is required before claiming a single 2 GiB backing buffer.
5. Primary serving workload must include several thousand output tokens at usual concurrency, with natural-EOS workload statistics distinguished from forced fixed-length generation. Preserve all high-concurrency unfavorable results separately.

This is a proposed test sequence, not an executed experiment or an approved shared-buffer implementation.
