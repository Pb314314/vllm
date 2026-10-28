# Memory experiment environment

Recorded 2026-10-28 UTC. Commands use the shared tree's `.venv/bin/python`,
which points to `/home/bopang/vllm/.venv`. `PYTHONPATH` selects the experiment
tree, not the installed editable checkout.

| Item | Value |
| --- | --- |
| Product-code baseline | `cff6224872` via plan branch `d84086a863d2` |
| Experiment HEAD at first model launch | `a5136e6e14189cea38fe84f9b4ee3e6ad0ab11b7` |
| Python | 3.12.3, GCC 13.3.0 |
| Torch / CUDA | `2.14.0+cu132` / 13.2 |
| FlashInfer | 0.7.2 |
| GPU | NVIDIA GB200, SM100, 152 SMs; Torch reports188729MiB, NVML189471MiB |
| Driver | 580.126.09 |
| `_C_stable_libtorch.abi3.so` MD5 | `d699613784e675cfcd4efc5715f66b3b` |
| Operator hosts | `gb200-rack1-01` GPU2 and Slurm40469 `gb200-rack1-04` GPU2 |
| Model allocation | Slurm40470, `gb200-rack1-10`, four GPUs |
| Cache | `/home/bopang/cg-project/design/idx-breakdown-cache/memory` |
| External logs / snapshots | `/home/bopang/cg-project/results/indexer_breakdown_memory_20261028/` |

Device UUIDs: rack1-01 GPU2 `f7cdefd9-3d67-384f-dbc1-53bc44d93985`;
rack1-04 GPU2 `60303add-b817-42e9-2313-9a78a05cf8af`;
model rack1-10 GPU0 `d82bdabf-4dc3-8649-5109-8b02fbe4ac63`,
GPU1 `16e411e9-555f-eee8-8c75-bfbc8473e89e`,
GPU2 `7ea1a670-314b-0489-529e-0e4050fbce1b`,
GPU3 `e936324a-9d12-14ba-dddd-9929c17fe528`.
Operator results claim allocation ownership/bytes, not latency. The rack1-01
supplemental V4.1-shaped allocation probe overlapped another user's model;
the later rack1-04 GPU2 probes used a separately checked free device.
All real-model cases used the exclusive rack1-10 allocation.

The first TP2 startup on Slurm40469, `gb200-rack1-04` GPU2/3, failed before
weight loading: generated `vllm.vllm_flash_attn.cute` Python artifacts were
missing from this new checkout. This is recorded in
`results/memory_model_tp2.jsonl`; it is not a successful model measurement.
The runner stopped its own process group, and GPU2/3 were checked at 0 MiB
afterward. No other user's GPU process was stopped.

Environment correction before TP4: generated `vllm/vllm_flash_attn/cute`,
`layers`, and `ops` are symlinked to the matching main environment's generated
directories under `/home/bopang/vllm/vllm/vllm_flash_attn/`. Import of
`vllm.vllm_flash_attn.cute.utils` was verified. These are ignored generated
artifacts, not a change to tracked model or kernel code.

## Reproduction

Run the operator experiment from the experiment tree with
`CUDA_VISIBLE_DEVICES=2` and `PYTHONPATH` set to that tree:

```bash
.venv/bin/python benchmarks/kernels/indexer_breakdown/memory_operator.py \
  --rows 512,4096 --widths 65536,262144 \
  --variants auto,deep_select,chunk_auto,chunk_deep_select \
  --modes eager,full,breakable --snapshots \
  --out benchmarks/kernels/indexer_breakdown/results/memory_operator_valid.jsonl \
  --snapshot-dir /home/bopang/cg-project/results/indexer_breakdown_memory_20261028/snapshots_valid
.venv/bin/python benchmarks/kernels/indexer_breakdown/memory_analyze.py \
  benchmarks/kernels/indexer_breakdown/results/memory_operator_valid.jsonl \
  benchmarks/kernels/indexer_breakdown/results/memory_operator_summary.jsonl
```

`memory_serve.py` records the complete real-model server command, TP size,
graph mode, source revision, hook hash and case status in JSONL. Hook events
are per process; actual GPU device0 supplies tensor and allocator records.
Snapshots include allocation stacks and device allocation history. Instrumented
model startup/latency must not be used as a performance result.

The source tree tracks generated kernel artifacts separately from product
source. This environment reuses prebuilt C++/CUDA artifacts; no claim is made
that latest upstream main `9bc1c8b4fede` has been dynamically tested here.
