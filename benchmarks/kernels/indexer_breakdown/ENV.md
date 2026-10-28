# Execution environment

## Source and runtime identity

- Experiment branch: `cg/indexer-topk-breakdown`, starting plan `d84086a863`.
- After user clarification, that branch is only the plan input. Experiment
  publication uses `cg/indexer-topk-breakdown-results`; the same product
  baseline and worktree are retained.
- Historical product code: `cff6224872ad24b10c224bc4186778dcd90a1b21`.
  Only benchmark/report files change in this experiment branch.
- Final chunk-policy reference: PR #64138, `4e5b4f8ce9`, based on `b4a0830f66`.
- A separately inspected main reference is `9bc1c8b4fe`; it is not silently
  substituted for the historical product source in benchmark results.
- Shared worktree: `/home/bopang/cg-project/wt-indexer-topk-breakdown`.
- `.venv` links to the existing `/home/bopang/vllm/.venv` environment.
  Commands set `PYTHONPATH` to the experiment worktree and assert the imported
  `vllm.__file__`. Installed-package version metadata refers to the original
  editable installation and must not be used as the source revision.
- Precompiled artifacts are hardlinks from the same-source `wt-pr8` worktree
  (`cff6224872`). No CUDA sources were modified. `git diff 60a7d568f4 cff6224872
  -- csrc` is empty.
- `_C_stable_libtorch.abi3.so` MD5: `d699613784e675cfcd4efc5715f66b3b`.
- `_deepselect_C.abi3.so` MD5: `edbdb13ef0ec0a900d54cf7dd858a6cd`.
- Import check: PyTorch `2.14.0+cu132`, Triton `3.8.0`, FlashInfer `0.7.2`,
  DeepSelect extension available. Full package list: `environment-freeze.txt`.
- CUPTI was absent initially; installed `cupti-python==13.4.0` via `uv pip`.
  Its dependency resolution installed `cuda-bindings==13.4.4` and
  `nvidia-cuda-cupti==13.4.92`; torch remains `2.14.0+cu132`. Both benchmark
  workers verified actual CUPTI kernel activities; no event-timing fallback
  is accepted. The package freeze reflects the environment after this install.

## Initial probe host

- Host: `gb200-rack1-01`, Linux aarch64, NVIDIA driver `580.126.09`.
- Four NVIDIA GB200 GPUs, each 189471 MiB as reported by nvidia-smi.
- Work assignment: top-k GPU0, full pipeline GPU1, memory probes GPU2/3.
- Maximum reported SM clock: 2062 MHz; memory clock: 3996 MHz.
- GPU clock locking was attempted with `nvidia-smi -i 0 -lgc 1500,1500` and
  rejected for insufficient permission. Clocks are **not locked**; workers
  record clock/temperature observations and interleave comparisons.

## Dedicated nodes

- Slurm 40469: one exclusive GB200 node for kernel workers, requested 8 hours.
- Allocation started 09:04:02 UTC on `gb200-rack1-04`; top-k GPU0, pipeline
  GPU1, coordinator's existing default-grid smoke GPU2 (finished).
- Slurm 40470: one exclusive GB200 node for model/memory work, requested 6 hours.
- Node identities and actual start/end times are recorded by each worker.
- Initial queue estimates were about one hour; probes execute on idle GPUs
  while these allocations wait. Reports distinguish probe and dedicated-node
  measurements.

## Method and interpretation

- CUPTI timing is preferred for GPU comparisons; per-kernel activities, graph
  elapsed span, and host enqueue time are separate quantities.
- Allocation capacity is separate from valid-column memory traffic.
- Tensor live peak, allocated bytes, reserved bytes and private-pool footprint
  are reported separately. A segment is not equivalent to a logical tensor.
- BF16 ties and approximate top-k behavior are measured rather than assumed
  equivalent. Exact-reference disagreements are retained in raw results.
- No new production dispatch rule is applied in this branch.
- Historical FULL capture passes the **uncompressed** configured model length
  as the persistent selector's host `max_seq_len`: runner line 2280, indexer
  metadata line 1768, dense call line 933. Core grids use `width * compression`
  for that scalar. Actual compressed row lengths remain separate device data.

## Final integration

Measurements finished before 16:35 UTC. The kernel reservation was released
at 16:35:41 UTC after all worker GPU contexts exited. Final code checks move
portable synchronization/cache/statistic calls to `torch.accelerator`, as
required by the repository, while retaining CUDA-specific graph and allocator
snapshot APIs. Raw measurements and archived runtime source hashes are not
rewritten by this source cleanup; the native operators and arithmetic are
unchanged. Optional model-forward/serving/accuracy sweeps were not run.
