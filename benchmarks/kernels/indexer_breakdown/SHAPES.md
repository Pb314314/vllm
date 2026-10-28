# Indexer shape inventory

Status: static source/config inventory on 2026-10-28 UTC; operator and model
measurements are recorded separately in `MEMORY.md` and `results/memory*.jsonl`.
No checkpoint field below was inferred from the model name.

## Versions and notation

- Historical graph-split branch: `d84086a863d2` (product code from `cff6224872`).
- Latest-main static cross-check: `9bc1c8b4fede` (not the same baseline).
- Final memory-fix PR #64138: `4e5b4f8ce9`; old prototype `553747b1df` uses a
  separate 1-GiB decode budget and is not the final PR's dispatch policy.
- `M` is configured `max_model_len`, `R` is the executed/padded decode-query
  row count, `L_r` is one row's actual visible history in index-cache units,
  and `W` is logits capacity. Uniform target verification has `R = B*(1+K)`;
  adaptive verification instead has `R = sum(q_i)` plus graph padding.
- Physical logits row stride can exceed `W`; every memory result records
  actual `stride(0)` and storage bytes. Capacity is not bytes accessed.

## Checkpoint-derived shapes

All counts exclude MTP/drafter layers and count actual indexer evaluations,
not every layer consuming the published index set. CUDA dense logits are
BF16 on SM100, irrespective of model-weight dtype; legacy cooperative and
persistent selection consume an additional FP32 logits buffer.

| Model / indexer kind | H x D | Configured index_topk | Kernel top-k | Compression / pool | Config M | Logits W | Indexer evaluations per target forward | Q/K format |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DeepSeek-V3.2-Exp | 64 x 128 | 2048 | 2048 | 1 | 163840 | M = 163840 | 61 | FP8 indexer; SM100 MX-scale adapter |
| DeepSeek-V4-Flash C4A | 64 x 128 | 512 | 512 | 4 | 1048576 | M/4 = 262144 | 21 (layers 2,4,...,42) | FP8 default or opt-in MXFP4 |
| DeepSeek-V4-Flash C128A / SWA | — | — | — | 128 / 0 | 1048576 | no score matrix | 0 | no learned top-k indexer |
| DeepSeek-V4-Pro-0813 C4A | 64 x 128 | 1024 | 1024 | 4 | 1048576 | M/4 = 262144 | 30 (layers 2,4,...,60) | FP8 default or opt-in MXFP4 |
| DeepSeek-V4.1-Flash pre-candidate dense | 32 x 128 | 512 | 512 | 2 | 1048576 | M/2 = 524288 | 3 (layers 2,8,14) | FP8 or MXFP4 |
| DeepSeek-V4.1-Flash candidate-source dense | 32 x 128 | 512 | 512 | 1 | 1048576 | M = 1048576 | 1 (layer 20) | FP8 or MXFP4 |
| DeepSeek-V4.1-Flash candidate consumers, sparse enabled | 32 x 128 | 512 | 512 | 1 | 1048576 | 2048 candidate blocks x 8 = 16384 | 4 (layers 24,28,32,36) | SM100 sparse MXFP8/MXFP4; BF16 directly to DeepSelect |
| Same V4.1 consumers, sparse disabled | 32 x 128 | 512 | 512 | 1 | 1048576 | M = 1048576, then candidate mask | 4 | dense BF16, then selected top-k backend |
| GLM5next: GLM-5.3-Flash | 32 x 128 | 2048 tokens | 512 pools | index_kpool = 4 | 1048576 | M/4 = 262144 | 45 (`indexer_types` all full) | FP8 pool cache |
| HY-v4 (source-schema defaults only) | 16 x 128 | 2048 | 2048 | 1 | 262144 | M | checkpoint-dependent; see below | FP8 cache in CUDA implementation |

HY-v4 checkpoint weights/config were not found in the mounted model cache.
Its row above is the **source schema's default, not a measured checkpoint**.
`layer_types` decides which layers use DSA; `indexer_types` / skip patterns
decide which evaluate the indexer. The schema's default indexer pattern alone
does not establish the number of sparse layers. HY-v4 serving is untested.

GLM5next selects 512 pools, expands each pool into four tokens, and can append
the incomplete tail (up to three entries). Benchmarking only `index_topk=2048`
would miss the production selector's `k=512`. Its independent CUDA copy
currently calls `as_fp32_logits` unconditionally even for an explicit
DeepSelect backend. PR #64138 changes the shared dense layer, not this copy.

## Exact local checkpoint evidence

Paths are under `/mnt/lustre/hf-models/hub/`; each ends in `config.json`:

- V3.2-Exp: `models--deepseek-ai--DeepSeek-V3.2-Exp/snapshots/194c67e12b1b0d6df0ef373ddcf215bc84027409/`.
- V4-Flash: `models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1/`.
- V4-Flash-DSpark (the dynamic model): `models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/62af8fffb2f7030cac4de2f0169f5b8d1101b646/`.
- V4-Pro-0813: `models--deepseek-ai--DeepSeek-V4-Pro-0813/snapshots/72e1d3230f6c080a530b0a1d46f8eb4602340597/`.
- V4.1-Flash: `models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/`.
- GLM-5.3-Flash: `models--zai-org--GLM-5.3-Flash/snapshots/46a050bec13002ad29ee795c16e61cf4fa3c4a9c/`.

Relevant fields: `index_n_heads`, `index_head_dim`, `index_topk`,
`num_hidden_layers`, `max_position_embeddings`, `compress_ratios`,
`index_source_layer_ids`, `candidate_source_layer_id`,
`candidate_topk_blocks`, `candidate_block_size`, `index_kpool`, `indexer_types`.

The DSpark checkpoint independently confirms the same target indexer shape:
43 target layers, H=64, D=128, k=512, C4A at layers2,4,...,42. Its config
has one next-token prediction layer and zero compression entries beyond
the target layers; target and drafter evaluation counts must stay separate.

## Source ownership and capture boundaries

Line numbers refer to historical branch `d84086a863d2`:

| Model | Dense / sparse entry | MRV2 breakable PIECEWISE boundary | Version-specific qualification |
| --- | --- | --- | --- |
| V3.2 | `models/deepseek_v32/attention.py:486` calls shared `sparse_attn_indexer` | decorated eager attention region at line 465 | max width M; sharing/skipping from index_topk pattern |
| V4 | `models/deepseek_v4/attention.py:747` calls `indexer.indexer_op` | `_sparse_indexer_and_attn`, decorator at line 732 | learned indexer exists only for C4A |
| V4.1 | `models/deepseek_v41/attention.py:1418` chooses SparseMQAIndexer or SparseAttnIndexer | decorated attention regions at lines 819 and 1022 | candidate source and earlier sources remain dense |
| GLM5next | `models/glm5next/nvidia/sparse_indexer.py:108` owns its separate kpool implementation | function decorator at line 107 | BF16->FP32 at lines 616-628; pool top-k at 633 |
| HY-v4 | `models/hy_v4/nvidia/attention.py:198` creates shared SparseAttnIndexer | decorated region at line 744 | exact active-layer count needs checkpoint layer_types |

Breakable capture is governed by `eager_break_during_capture` and actual
runtime mode (`compilation/breakable_cudagraph.py:99-116`), not solely by
`splitting_ops`. Compiled piecewise/Inductor partition is a separate mechanism;
an empty splitting list can change effective mode to NONE/FULL rather than
producing a PIECEWISE graph containing attention.

Active call ownership on this modular-model branch is also important:
V3.2 calls the decorated shared Python `sparse_attn_indexer` directly;
V4 and HY-v4 use `SparseAttnIndexer.forward_cuda`, which calls
`torch.ops.vllm.sparse_attn_indexer` (shared layer line1139). V4.1 dense
sources use that same Torch op, while sparse consumers use the Python
`SparseMQAIndexer.forward` (line127) inside the model's outer attention
eager break. GLM5next calls its decorated Python kpool function directly.
The default compile splitting list still contains `vllm::deepseek_v4_attention`
(`config/compilation.py:927`); that string alone does not establish that this
modular V4 path executes an op with that name.

| Mechanism / setting | Effect on indexer capture | Evidence / limit |
| --- | --- | --- |
| MRV2 breakable, CompilationMode.NONE, runtime PIECEWISE | outside graph, even when splitting_ops=[] | actual TP4 V4 runs; outer eager breaks |
| MRV2 breakable, runtime FULL | eager-break decorator runs inline and indexer is captured | breakable_cudagraph.py:107-110; actual TP4 V4 runs |
| Compiled VLLM_COMPILE, default splitting_ops | list includes vllm::sparse_attn_indexer | config/compilation.py:909-929,1325; inspect emitted FX op for model-specific paths |
| Compiled VLLM_COMPILE with splitting_ops=[] | PIECEWISE resolves to NONE; FULL_AND_PIECEWISE resolves to FULL | config/compilation.py:1356-1381; not equivalent to breakable mode |
| fuse_attn_quant, without Inductor graph partition | can clear split list and select FULL | config/compilation.py:1401-1418; kernel/model support still required |
| use_inductor_graph_partition | splitting ops are passed to Torch partition rules | compilation/partition_rules.py:40-76; no dynamic compiled-mode model test in this audit |

No flag is inferred to capture all indexers merely from an empty splitting
list: the resolved compilation mode, graph runtime mode, actual executed op
and breakable context must all be checked.

On `d840`, `attention_config.indexer_sparse_logits=False` by default, opt-in
on MXFP4 only. On latest main `9bc1`, the field defaults to None and the model
hook auto-enables sparse consumers when SM100/DeepGEMM/DeepSelect/layout/no-CP
requirements hold; FP8 and MXFP4 are both supported. Therefore preserve the
resolved flag and actual class of each V4.1 layer in every experiment record.

The GB300 `latency_specialized` V4.1 path has additional requirements and a
separate TP-row-sharded `_decode_rows` implementation. It is not equivalent
to the stock path used by the GB200 experiments and is not covered by them.

## Priority kernel grid derived from this inventory

- `k = 512, 1024, 2048`; GLM's 2048-token index set exercises k=512 pools.
- Dense W: 163840, 262144, 524288, 1048576; add 65536 for a 256K V4 cap.
- Sparse V4.1 W: 16384; its direct-BF16 path needs its own baseline.
- Actual valid lengths must be expressed in index-cache units. For V4 C4A,
  8192 uncompressed context tokens correspond to about 2048 valid columns.
- R includes graph padding; API concurrency is not the executed row count.
- Each dataset must distinguish capacity W, physical stride, actual row ends,
  host max_seq_len baked into a captured kernel choice, and per-row visibility.
