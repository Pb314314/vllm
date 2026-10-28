# Pipeline benchmark environment

Source checkout: `cg/indexer-topk-breakdown` at
`d84086a863d2d4c3aef82e3292627b3f178ff8cb`. The benchmark implements the
PR 64138 planner independently; it does not patch the product implementation.
Every JSONL result includes the benchmark file hash, source SHA, command,
Torch/CUDA versions, GPU identity, and clock/temperature samples.

Initial smoke node: `gb200-rack1-01`, physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`).
GPU clocks could not be locked because the account lacks permission. Other
workers use separate GPUs; telemetry covers the node to expose that fact.

Core scan moved to exclusive Slurm job `40469` on `gb200-rack1-04`, physical
GPU 1. GPU 0 hosts the separate top-k experiment. Commands are invoked with
`srun --jobid=40469 --overlap --nodes=1 --ntasks=1`; no other worker uses this
GPU. After that move the dev-node GPU 1 probe stopped.

Python: the worktree `.venv` points to `/home/bopang/vllm/.venv`.
Parent validation: Torch `2.14.0+cu132`, FlashInfer `0.7.2`, DeepSelect available.
Native artifacts are linked from the matching cff6224872 checkout; no C++
source changes between that checkout and the supplied artifacts.

The initial import probe found `cupti-python` missing. It was installed with
`uv pip install --python .venv/bin/python 'cupti-python>=13'`: version 13.4.0,
with `nvidia-cuda-cupti` 13.4.92 and `pyelftools` 0.33. The script explicitly
requires this package so FlashInfer cannot silently substitute another timer.

Actual CUPTI activity timing and independent CUDA-kernel profiling passed
the accepted smoke matrix: V4.1-c1, 32/65 rows, 8192 raw context tokens,
both MXFP4 and FP8, single-auto/single-DeepSelect/final-512 (12 successful
variants). The temporary larger probe on node 04 was stopped before records
after noticing and correcting an intermediate-buffer lifetime in the
benchmark: every chunk must release its logits before producing the next.
All core measurements use the corrected lifetime. The 270-variant first
core shape family on node 04 also passed all correctness and timing gates.

Caches live outside the small root filesystem:
`/home/bopang/cg-project/design/idx-breakdown-cache/pipeline`.

All timings describe this checkout's actual DeepGEMM, candidate operations,
top-k dispatcher, and allocator. They do not claim measurements of a later
main revision or a model checkpoint execution.

Synthetic-score distribution samples were generated on idle physical GPU 2
of the same exclusive job, with the GPU 1 core scan continuing. The GPU 2
process exited immediately after dumping six 128×32768 CPU BF16 tensors to
`/home/bopang/cg-project/design/idx-breakdown-cache/pipeline/score_samples`.
`pipeline_dump.py` regenerates these samples with valid per-row lengths and
source metadata. They are actual DeepGEMM outputs from synthetic finite
quantized Q/K inputs, not model-checkpoint activations. The separate top-k
worker received these for a value-distribution cross-check.

Core processes initially had CPU affinity 0–139. Topology reports GPUs 0/1
local to NUMA 0 (CPU 0–69), GPUs 2/3 local to NUMA 1 (70–139). During the
brief GPU 2 dump startup around 09:31 UTC, the V4.1-c2/R8192/1536-context
eager samples showed a transient host outlier despite stable GPU timings.
That raw observation is retained. The auxiliary large-row eager recheck is
run on local CPUs 32–39 with 30 samples, separately from the unbound core,
so the report can distinguish stable costs from host/driver interference.

## Final-PR native cross-check

A separate GPU3 batch uses the actual checkout
`/home/bopang/cg-project/wt-b4a0fix` at
`4e5b4f8ce9b067d92e7373d950727d8238e9a289` and its existing
`/home/bopang/cg-project/venv-b4a0fix` environment. The final arm directly
calls that checkout's `varlen_decode_plan` and `_varlen_decode_rows`.
Historical before/after policy paths are reproduced on these same native
operators for a controlled comparison. This remains an operator experiment,
not a full engine/model performance run.

The stable-libtorch binary differs: old MD5 `d699613784e675cfcd4efc5715f66b3b`,
new MD5 `5cd62d1bd33258d06cc310e83b591c90`. New records contain SHA256 and size
for stable-libtorch, DeepSelect and the DeepGEMM extension, the Python package
root, the selected GPU UUID and visibility mapping. New DeepGEMM strides are
recorded from actual outputs, not assumed to match the old padding.

This environment has Torch `2.14.0+cu132`, FlashInfer `0.7.2.post1`.
Installing `cupti-python==13.4.0` also updated its CUDA CUPTI library from
13.2.75 to 13.4.92 and added pyelftools0.33; Torch was unchanged. Twelve
FP8/MXFP4 matched smoke cases and a direct-production-helper probe passed
before the larger batch started. GPU3 uses CPU104–119 and separate compilation
caches under `pipeline/newnative`. Its 1 Hz power/clock log and JSONL outputs
are labeled `pipeline_newnative_*` and must not be merged into GPU1 historical
absolute timings.

`pipeline_newnative_freeze.txt` preserves the complete independent environment
after CUPTI setup. The final native batch finished core270/candidate24/actual18
records, with six expected legacy correctness rejections in the actual-bound
grid. GPU3 and its monitor exited and were released at11:41:45 UTC. Historical
GPU1 continued independently; product source in both checkouts was unchanged.

Historical GPU1 runners and monitor completed at approximately11:50 UTC.
Its preserved trace has5308 samples and1537 joined case windows; not every
early core case overlaps the later monitor. The final audit and report
distinguish rejected correctness cases, noisy original observations,
controlled repeats and derived policy rows. Full source/library provenance
is carried by the final-native records; earlier historical records use the
historical environment hashes recorded above.

Static comparison of prototype553747b1df against cff6224872 has no differences
under csrc, vllm/third_party/deep_gemm or vllm/utils/deep_gemm.py. The prototype
persistent_topk.cuh SHA256 is
65472d7da55c9c0b06609405511d9e54f6ca950b16251ee1f091eaa6643de7cb,
identical to the historical tree. Its helper's full-metadata/sliced-metadata
condition matches the reproduced pipeline scheduling.
