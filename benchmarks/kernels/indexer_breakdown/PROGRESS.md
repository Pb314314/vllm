# Indexer/top-k overnight experiment progress

## TL;DR

- 主实验完成：kernel breakdown、实际 PR native 对照、真实 TP4 模型显存。
- dense BF16→FP32 副本存在；按 Q 行分块正确且显著省临时显存。
- 512 MiB 并非总峰值上限：candidate source 约576.25 MiB。
- 最终 PR native 的 c1/R8192/8K GPU流水线约501→788微秒，有性能代价。
- 小batch是否用DeepSelect取决于行数、容量、k，不能一律切换。
- PW在图外算indexer，但本配置MRV2 profiling已计大decode；两次.90未OOM。
- 发现既有地址溢出，坏基线排除；没有修改产品实现。
- 完整结论见REPORT.md；后续metadata跨层复用的新实验另见OPTIMIZATION.md。

## Scope and publication

Execution began 2026-10-28 08:56 UTC with an approximately eight-hour timebox.
The supplied plan is PLAN.md from `d84086a863`. Historical product source is
`cff6224872`; final PR reference and separate native cross-check are
`4e5b4f8ce9`. Product code is unchanged.

The user clarified that `cg/indexer-topk-breakdown` is the plan source only.
Subsequent results are published to `cg/indexer-topk-breakdown-results`.
An initial environment checkpoint `a5136e6e14` had already been pushed before
that clarification; the plan branch is not rewritten or further updated.
The shared worktree remains `/home/bopang/cg-project/wt-indexer-topk-breakdown`.

## Completed phases

| Phase | Completed work | Evidence |
| --- | --- | --- |
| 0 | Source/native fingerprints, actual CUPTI, default persistent benchmark 30 cases, closed PR60802 review | ENV.md; results/persistent_default_smoke.json; pr60802-reference.json |
| 1 | 129 valid operator-memory cases, nine successful TP4 model launches, six capture A/B/C cases, two .90 stress runs, explicit DS ablation | MEMORY.md; results/memory*.jsonl |
| 2 | Checkpoint-derived shapes, actual indexer layer counts, model-specific PW boundaries, GLM pool k, missing HY checkpoint | SHAPES.md |
| 3 top-k | Broad row/width/k512/1024/2048, warm/cold and eager/graph grids; real checkpoint activations and controls; CV reruns and PR-native anchors | TOPK.md; ENV_topk.md; results/topk*.jsonl |
| 3 pipeline | Historical core1620, actual PR native core270 and candidate24, q/budget/full/long/candidate/actual-bound/eager/launch controls | PIPELINE.md; results/pipeline*.jsonl |
| 4 | Chinese synthesis, shape selection diagrams, measured slow regions, memory interpretation and selection correctness | REPORT.md |
| Optional Phase5 | Full model-forward dispatch/performance grid not run; no serving/accuracy claim | REPORT.md limitations |
| Follow-up | New schedule-sharing experiment and optimization design, after required measurements | OPTIMIZATION.md; results/optimization*.jsonl |

## Scientific corrections retained

- Original random raw quant-cache bytes were unsuitable correctness inputs;
  valid planar values/scales reruns supersede them. Diagnostic failures remain
  archived and are not performance evidence.
- W512K native stride is W+1024, unlike W256K/1M's W+512.
- FULL host max_seq_len is uncompressed M; actual eager raw bound and the
  compressed-bound counterfactual are separately labeled.
- Sampled selection has exact fallback; observed address wrap is a defect,
  not permitted approximation. Incorrect baselines have no winner status.
- Private pool, global capture delta, live allocation and valid-column traffic
  are separate measurements. Two equal 4104MiB segments were not reproduced.
- Per-layer GPU costs and isolated CPU wall are not model serving latency;
  adaptive stock clocks and observed power capping constrain small deltas.

## Execution and validation notes

Kernel allocation: Slurm40469, gb200-rack1-04, 09:04 UTC onward. Model-memory
allocation: Slurm40470, gb200-rack1-10, 09:16 UTC, no longer active after its
six-hour reservation. Other jobs and users' processes were not touched.

Workers append/fsync raw records with resumable keys. Coordinator checkpoint
publication did not meet the requested every-30-minute cadence during the
long run; local raw data continued to persist. Final publication includes all
completed measurements, source archives, hashes and explicit missing cases.

A final pre-commit attempt overlapped four still-appending native anchor
files. Its saved patch was retained; the saved patch's 38 displaced raw/attempt lines were
recovered by exact-line comparison without overwriting newer records. Additional
anchor trials missing after rollback are rerun with the original resumable keys.
Final
JSON parsing, key/trial and source-hash audits run after file closure.
Formatting checks are subsequently run against explicit files to avoid
stashing active result files.

## Final closure

All GPU experiments and worker writes ended before16:35 UTC. Kernel allocation
40469 was released after confirmation that no experiment GPU context remained.
Top-k final audit:67901 timed raw records,21754 complete100x3 valid groups,
zero JSON parse errors and zero duplicate timed keys. Some noisy groups remain
explicitly labeled (438 eager,35 graph) and are not stable winner evidence.
Recovered native anchors are complete:96 V3.2 groups and102 V4-Pro groups.
Attempt logs retain retry duplicates/one prior unmatched attempt per file;
see topk_recovery_audit.jsonl rather than claiming perfect attempt histories.
Two actual-PR optimization experiments and mask tuning completed and passed
checks; optional full-model forward/serving/accuracy validation remains unrun.
