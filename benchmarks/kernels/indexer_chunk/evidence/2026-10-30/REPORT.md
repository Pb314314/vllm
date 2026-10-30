# SM100 decode indexer chunking: default-selection evidence

## Decision and scope

Recommend **2048 MiB (2 GiB)** as the initial configurable BF16 decode-logits payload budget. Preserve `0` as an off switch and document `4096` as a higher-memory option for deployments prioritizing fewer chunks. This is a measured tradeoff on GB200 and the covered DeepSeek configurations, not a global optimum or a promise of zero regression.

The implementation changes the Python caller, chunk planning and profiling reservation; it does not modify DeepGEMM, switch to DeepSelect, or change top-k dispatch rules. Both cooperative and persistent BF16 support are prerequisites. The existing auto provider still selects for the row count it receives; a small tail can therefore choose cooperative without a new dispatch rule. Existing compact candidate-block selection is preserved and may use its existing FP32 selector; this is not a restored full-size FP32 logits copy.

Eligible scope: SM100 MXFP4 index cache, DCP=1, no PCP, dense/source indexer rows, auto/cooperative/persistent, top-k 512/1024/2048. Sparse candidate consumers and unsupported paths remain unchanged. This is not a general decode fix for FP8 caches or SM90/SM120.

## First-principles sizing

Let R be query rows in this step, W the configured compressed key capacity, L the effective history actually used, and B the byte budget. BF16 logits payload is `2*R*W` bytes. If it fits, retain the original path. Otherwise derive `limit = floor(B/(2*W)/16)*16` rows, form balanced chunks, and process the largest first. Each row still scores its entire effective history and selects top-k independently; there is no merge across Q chunks. The temporary logits allocation is reusable once that chunk's candidate selection and top-k finish.

Compression reduces W (and effective K length), not the number of query rows. Ordinary decode has roughly one query row per request; verification with seven drafts can have eight. With maximum model length 1,048,576 and B=2 GiB:

|Layer compression|W|Maximum rows/chunk|Requests if 8 query rows/request|
|---|---:|---:|---:|
|1|1,048,576|1,024|128|
|2|524,288|2,048|256|
|4|262,144|4,096|512|

Actual balanced chunks can be smaller. One global byte budget already produces different per-layer row limits. Separate per-layer budget knobs are not justified by the current evidence.

The budget covers **BF16 logits payload**, not whole graph/model memory. Physical row padding, per-chunk metadata, candidate block scores and selection scratch are additional. A candidate-producing C1 layer needs about 1.25 times its BF16 payload before padding/allocator details. In the tested 2 GiB C1 case its private graph pool is about 2.504 GiB. Whole-model graph pools include other layers and buffers and are larger.

Capacity W and effective length L must be distinguished: a large configured W can trigger chunking even for short L. Short effective history leaves less useful work to amortize repeated scheduling/kernel launches. Graph replay removes Python's per-chunk launches, but not the GPU schedules or extra kernel executions. All chunks are already captured into **one** FULL graph replay. Eager still executes the Python loop.

DeepGEMM's output being BF16 does not imply BF16 GEMM compute: these tests use MXFP4 Q/K and the actual DeepGEMM implementation. A useful default should preserve sufficient parallelism, but “must become compute-bound” is not an adequate acceptance criterion. We have kernel timings, not an Nsight roofline proof of compute versus memory bounds.

## Budget refinement

Final largest-first implementation; real DeepGEMM with deterministic synthetic MXFP4 Q/K, nonuniform scales and shuffled cache pages. Two reverse-order visits, 60 timing samples per visit, 0.5 s sustained warmup. Cold-L2 CUPTI spans below are separate from warm host-wall measurements. All arms use the same corrected native binary. The coarse 512 MiB–8 GiB sweep is preserved separately; it is not overwritten by this refinement.

| Budget MiB | Chunks | GPU span µs | Change vs off | Private pool GiB |
| --- | --- | --- | --- | --- |
| off | 1 | 700.27 | +0.00% | 20.010 |
| 1536 | 11 | 814.83 | +16.36% | 1.840 |
| 2048 | 8 | 771.63 | +10.19% | 2.504 |
| 2560 | 7 | 759.81 | +8.50% | 2.895 |
| 3072 | 6 | 750.63 | +7.19% | 3.363 |
| 4096 | 4 | 722.83 | +3.22% | 5.004 |

At R=4096, 2.5 GiB still gives four chunks, just like 2 GiB; spending more budget does not necessarily reduce launches. At R=8192, 3 GiB reduces eight chunks to six but still carries additional schedule/launch cost. The 4 GiB option reduces latency further but roughly doubles C1 temporary reservation. The coarse R4096/8K point measured about +70% at 512 MiB, +35% at 1 GiB, +10.6% at 2 GiB, and +3.2% at 4 GiB. This rules out 512 MiB as a good performance default for that workload.

The choice of 2 GiB prioritizes a substantial memory bound with small measured whole-model graph cost; 4 GiB is preferable when that additional memory is acceptable and eager/short-history latency matters more. There is no scalar “best” independent of deployment memory and latency requirements.

## Model forward evidence

Actual DeepSeek-V4.1-Flash weights, TP4+EP on four GB200 GPUs. Synthetic token IDs, **zero historical KV**, fixed KV allocation. Four model-shaped workloads; 150 graph samples per visit and two reverse visits. Indexer graph host capacity is fixed to the production 1M maximum; the rest of this exact-shape model-forward benchmark is not a full serving workload. Rank-maximum graph latency is reported. Peaks are maximum per-device allocated bytes; graph-private reservation is reported separately in the raw summary.

| Case | Budget MiB | Graph ms (off → chunk) | Change | Peak saved GiB/GPU | Graph pool GiB/GPU |
| --- | --- | --- | --- | --- | --- |
| q8_b256_l8k | 2048 | 40.156 → 40.175 | +0.048% | 2.501 | 6.779 → 3.277 |
| q8_b256_l8k | 4096 | 40.156 → 40.167 | +0.027% | 0.000 | 6.779 → 6.779 |
| q8_b512_l8k | 2048 | 71.544 → 71.689 | +0.204% | 7.503 | 14.736 → 4.729 |
| q8_b512_l8k | 4096 | 71.544 → 71.578 | +0.048% | 5.002 | 14.736 → 7.732 |
| q8_b1024_l4k | 2048 | 118.817 → 119.070 | +0.213% | 17.507 | 27.639 → 5.619 |
| q8_b1024_l4k | 4096 | 118.817 → 118.938 | +0.101% | 15.006 | 27.639 → 7.623 |
| q8_mixed | 2048 | 46.635 → 46.774 | +0.296% | 4.142 | 9.301 → 4.734 |
| q8_mixed | 4096 | 46.635 → 46.663 | +0.059% | 3.142 | 9.301 → 4.922 |

Case names: `q8_b256_l8k` = 256 requests × 8 query rows, 8K history; `q8_b512_l8k` = 512 × 8; `q8_b1024_l4k` = 1024 × 8, 4K history. `q8_mixed` = 128 requests at 4K + 128 at 8K + 64 at 16K, each with eight query rows (2560 rows total).

The small whole-model percentage is consistent with a limited number of affected dense/source indexer layers being only part of the full forward. It does not mean the indexer itself has zero cost; the per-indexer regressions and breakdown below remain material.

The initial mixed graph/eager sweeps measured about 5.54% eager regression at the largest 2 GiB case and 3.05% at 4 GiB, but other visits drifted materially. The controlled eager-only follow-up below is the stronger attribution. FULL graph remained substantially faster than eager on the measured model-forward points. No automatic fallback to eager is added. Operator-only actual-bound eager can have much larger relative overhead because the original operation is very short and the loop now repeats; those regressions remain in `refinement/RESULTS.md`.

The earlier job41434 had unstable/no-op control timings, including an apparent 12.9% regression in one aggregate. It is retained as preliminary evidence and excluded from default selection. Final job41450 provided stable graph comparisons with production-capacity indexer metadata, sustained comparison and reverse visits. A subsequent low-budget scan also found substantial eager baseline drift; controlled same-batch eager comparisons are recorded separately below, and the preliminary eager percentages must not be treated as universal. This does not establish PD throughput or generation throughput for the selected default.

## Lower-budget whole-model cross-check

Job41484 repeats the final implementation at 1/1.5/2/4 GiB, in forward and reverse budget order. Same real-weight, zero-history-KV model-forward scope as above.

|Workload|Budget MiB|Graph latency change|Peak saved GiB/GPU|
|---|---:|---:|---:|
|512 requests × 8, 8K history|1024|+0.275%|8.573|
|same|1536|+0.175%|8.323|
|same|2048|+0.015%|7.503|
|same|4096|-0.133%|5.002|
|1024 requests × 8, 4K history|1024|+0.434%|18.008|
|same|1536|+0.300%|18.008|
|same|2048|+0.220%|17.507|
|same|4096|+0.111%|15.006|

The small negative change is not evidence of a general speedup. At the largest workload, reducing 2 GiB to 1–1.5 GiB saves only about another 0.5 GiB of peak memory while adding launches. Together with the operator curves, this favors 2 GiB as an initial compromise. Eager measurements from this mixed graph/eager protocol were inconsistent: the 512-request unchunked visits were about 229 and 171 ms, producing an apparent roughly 25% regression. Those raw results are retained. A dedicated interleaved eager experiment prepares each batch once and restores the same state between calls; its results follow.

## Controlled eager-only follow-up

Job41510 prepares each batch once, restores the same KV state before every call, and interleaves each nonzero budget with unchunked runs. Eight warmup calls and 20 timed calls per visit; two visits per budget; four ranks reduced by maximum. State restoration, barriers and synchronization before timing are excluded. No graph capture or batch rebuild occurs between budget arms. GPU-event elapsed time includes eager CPU dispatch gaps and agrees closely with host wall time.

|Workload|Budget MiB|Two adjacent-baseline latency changes|
|---|---:|---:|
|1024 requests × 8, 4K history|1024|+6.44% / +6.42%|
|same|1536|+4.08% / +3.70%|
|same|2048|+3.73% / +3.34%|
|same|4096|+2.16% / +1.38%|
|512 requests × 8, 8K history|2048|+2.35% / +0.22%|
|same|4096|+0.81% / +1.35%|

Each comparison uses the mean of adjacent unchunked visit medians. At 1024 requests, the two unchunked brackets around 2 GiB differ by 0.65% and 0.21%, supporting a real approximately 3–4% cost. At 512 requests the second 2 GiB bracket differs by 4.20%, so 0.22% is not a precise effect estimate. The first unchunked 512-request visit also decays from 374 ms to about 176 ms within its samples, making its adjacent 1 GiB comparison unreliable; it is preserved, not silently discarded. All eight budget comparisons and raw samples are in `eager_pairs/model_41510/SUMMARY.json` and the raw receipts. The earlier roughly 25% aggregate eager regression was not reproduced as a stable effect in this controlled run; its exact cause remains unisolated.

This supports a configurable tradeoff: 2 GiB substantially bounds memory with small whole-model FULL graph cost, while 4 GiB reduces eager dispatch/scheduling cost when memory permits. It does not justify a claim of no eager regression.

## Kernel attribution

R=4096, capacity=1M, effective history=8K, C1 candidate source, top-k=512. Diagnostic warm CUPTI trace: median kernel duration times call count. Only two trace invocations per arm; use this to explain operation counts and where time is spent, not as independent high-confidence latency estimates. These sums do not replace cold-L2 end-to-end spans.

| Budget MiB | Chunks | Added schedules µs | DeepGEMM µs | Candidate reduction µs | Candidate top-k µs | Final BF16 top-k µs |
| --- | --- | --- | --- | --- | --- | --- |
| off | 1 | 0.00 | 105.65 | 77.17 | 21.04 | 151.38 |
| 2048 | 4 | 18.30 | 121.79 | 76.03 | 23.87 | 145.34 |
| 4096 | 2 | 9.73 | 110.98 | 75.07 | 20.96 | 143.17 |

The common unchunked schedule is prepared outside the timed caller; extra chunk schedules are inside it. Effective scoring work is unchanged, but splitting it increases total DeepGEMM time and adds metadata kernels. Candidate reductions and final top-k do not scale identically with chunk count. CUDA graph removes repeated CPU dispatch, not those GPU costs. No claim that all of the delta is compute utilization or bandwidth without further profiling.

## Allocator and tail policy

Largest-first balanced chunks reduce peak reservation when chunk sizes differ: R8192, 3 GiB budget falls from about 6158 MiB to 3444 MiB private pool; 6 GiB falls from 12282 MiB to 6846 MiB. Live peak was already much smaller; the original ordering retained incompatible allocations during graph capture. Identical score bits and top-k correctness pass in both orders, with no material speed change.

All six balanced-versus-full-size-tail cases completed, with 40 samples per visit and two visits. At R1031 and 2 GiB, full-size chunks improve the diagnostic cold span by 2.64% (121.81 to 118.59 µs) but increase graph pool from 1324 to 2564 MiB. At R2055 and 4 GiB, they are 1.19% slower and increase pool from 2604 to 5124 MiB. Mixed C2 R2560 improves 2.42% but increases pool from 1286 to 2054 MiB. No-op controls vary by about 1%; these small speed differences do not establish a compelling general win. Retain balanced largest-first. Full data: `tail_policy/COMPARISON.json`.

## Production service startup and shared graph pools

Jobs41502 (off) and41511 (2 GiB) use the same V4.1-Flash TP4+EP configuration, DSpark seven draft tokens, maximum model length1M, maximum sequences1024, batched tokens8192 and FULL_DECODE_ONLY capture sizes8/128/512/1024/2048/4096/8192. Both use `gpu_memory_utilization=0.85` on owned isolated GB200 nodes. This is actual serving-runner profiling and shared-pool capture, not the exact-shape forward harness.

|Startup observation, per worker|Off|2 GiB|
|---|---:|---:|
|Same profiling capture phase|33.68 GiB|11.66 GiB|
|Available KV memory after profiling|-14.12 GiB|62.61 GiB|
|Final full capture after KV allocation|not reached|10.00 GiB|
|API health|failed: no memory for cache blocks|passed|

All four B workers completed final capture and the API health/metrics check passed. No client inference was timed in this run. The unchunked arm failed before serving, so it is not a throughput comparison.

The KV headroom delta is **not** all saved tensor memory. The old disabled-chunk profiling branch still conservatively reserves a whole FP32 logits copy even on the combined BF16 runtime; the eligible chunk path replaces that estimate with bounded BF16/candidate scratch. Therefore the improvement includes both actual capture/scratch savings and corrected profiling reservation. Compare the same capture phase (33.68 versus11.66), not profiling A against final B. A later manual-KV hint in the existing worker adds final graph bytes to `peak_activation_memory`, which already includes the graph estimate; that hint is not an independent under-reservation measurement. Worker profiling code was not changed in PR2.

## Correctness and limitations

- 451 existing FP32/BF16 kernel regression tests passed on the common corrected native build (job41431).
- Nine focused final chunk-range/profiling tests passed (job41453).
- Producer valid logits bits and candidate score bits must match the unchunked reference; exact top-k values, unique/in-range positions, padding/sentinels and tail slices are checked before timing. Empty rows, mixed lengths, same-graph length shrink/grow and changed weights were exercised. The profiling estimate was required to cover measured live peak in refinement.
- Tied index identity/order is intentionally outside this PR's acceptance contract, as agreed. Whole-model hidden states and generated token sequences are **not** claimed bitwise identical. Baseline repeated execution is itself nondeterministic in the model harness; those outputs remain saved, not discarded.
- V4 real-text collocated FULL graph A/B/A: 32/32 final answers correct in all three arms; eight unique long-prefix arithmetic prompts repeated four times. A/B token-string sequence equality is 9/32; A/A is 13/32; B/A2 is 11/32. This demonstrates run-to-run variation under the serving protocol, but does not isolate the cause or prove that every A/B difference has the same cause. B's four workers witnessed replay with query chunks. The B budget is deliberately 8 MiB to force the path at 32 requests; this is functional coverage, not default-budget performance or a standard GSM8K score.
- V4.1 collocated A/B/A (job41468): 64/64 correct final answers per arm; all four B workers witnessed chunk replay. Eight unique questions repeated eight times, forced 32 MiB B budget. Token-string equality A/B=11/64, A/A=8/64, B/A2=5/64; no token-identity claim.
- V4 two-node NIXL PD A/B/A (job41481): 32/32 correct final answers per arm; all four B decode workers witnessed chunk replay at forced 8 MiB. The prefill barrier releases 32 decode requests together. Each arm transferred 7,546,079,232 bytes through NIXL with nonempty remote-block metadata; missing transfers fail the protocol. Token-string equality A/B=9/32, A/A=12/32, B/A2=14/32. This is PD functional coverage, not PD throughput evidence.
- Candidate selection without DeepSelect (job41498): the existing PyTorch fallback passed score/top-k and memory-bound checks, including same-graph length updates. The first attempt (41495) failed because the test observer only captured DeepSelect candidate scores; it was fixed in benchmark observation only, and that failed receipt is preserved.
- V4-Pro k1024 is shape coverage, not a V4-Pro end-to-end model evaluation. k2048 has native top-k correctness coverage but no claim of a full model using this MXFP4 configuration. SM90/SM120, FP8 cache, DCP/PCP and compact sparse consumers are out of scope.

## Provenance and independent dependency

The base is locally reconciled PR1 #64138 (`293880733f87a095acac787a996e68b097eac7cc`) plus cooperative #62172 (`c548ceec666c9acf3a1caf502311c14b11bd0994`). Those online heads are not a conflict-free textual merge. PR2-only patch and before/after hashes are in this review directory.

Both benchmark arms additionally use the same two-expression `size_t` address-width fix, native SHA256 `f96b1e72533bc79d154f245d0ffc62817fd99fd2987e97b8e4e63a14dbe02f92`. Without it, a pre-existing uint32 row-offset overflow invalidates the largest unchunked FP32 and BF16 baselines. The fix is archived as `ADDRESS_DEPENDENCY.patch` and must be separately reviewed/integrated; it is not a chunking performance gain. The published review branch separates the BF16 reconciliation, address-width correction and PR2 feature into three commits; the PR2-only diff excludes both prerequisites.

Related #62169 chunks whole requests on SM90/SM120 uniform decode and explicitly excludes this SM100 varlen case. This is complementary scope, but caller changes require reconciliation. Prefill chunking and fused-prefill #60618 are different paths. A future fused decode scorer/selector could avoid global logits materialization, but it would require a different kernel and exact selection/rounding validation; no unmeasured speedup is promised here.

## Publication and review status

All assigned validation runs are complete and production sources are frozen at `d51f40aeffe014e4ee1f77759b22501893f84d5e`. The evidence export includes raw numerical samples, case shapes, proofs and executed experiment scripts. Failed attempts and preliminary noisy results remain in the original local archive and are described above. No universal optimum, zero-latency-regression, generated-token identity or serving-throughput claim is made.

The feature branch depends on the reconciled BF16 PRs and separate address-width correction. The repository PR checklist and automated checks are complete. Human submitter line-by-line review and relevant-test confirmation remain required before opening the upstream PR. AI assistance was used; this report does not assert human review or testing has happened.
