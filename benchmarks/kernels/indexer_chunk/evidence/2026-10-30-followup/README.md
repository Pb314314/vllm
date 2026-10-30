# PR2 completed measurements: fixed-shape, DEP and adaptive serving

**Status:** data consolidation only. No new experiments are running. The short-output follow-ups (41655/41656) were canceled during startup at the user’s request and supply no performance results. Production head remains `d51f40aeffe014e4ee1f77759b22501893f84d5e`; no default change is made here.

## What the evidence establishes

- Query-row chunking substantially reduces measured peak allocated memory in the tested large shapes. Fixed-shape FULL graph model-forward deltas are small relative to the whole forward; they are not serving-throughput estimates.
- Actual adaptive serving has unresolved unfavorable results, including 2 GiB throughput below both unchunked baselines at high concurrency. Baseline drift is shown rather than averaged away.
- Normal-concurrency workloads with several thousand output tokens are NOT covered. The current 2 GiB default is provisional, not an established production optimum.
- Score/top-k correctness from the earlier evidence remains separate from model hidden states and generated tokens. Repeated baseline and same-graph hidden-state differences occur; this package does not establish universal model-output bit equivalence.

## Workload and timing definitions

All measurements use actual DeepSeek-V4.1-Flash weights on four GB200 GPUs, max model length 1,048,576, MXFP4 indexer cache and the same BF16 top-k prerequisites/native offset fix in every arm. TP4 uses EP; DEP is DP4 + TP1 + EP4. This package does not compare against today’s latest main without prerequisites.

Fixed-shape runs use synthetic tokens and zero historical KV, real expert routing, target model only (no drafter, sampling, adaptive policy or HTTP). KV allocation is fixed at 32 GiB/GPU. Budget visits are 0/2/4/4/2/0 GiB, 150 graph samples/visit. Each sample is the maximum of four rank times; tables average the two visit medians. The timed span excludes state restoration and the pre-forward barrier, but includes model communication. Graph capture uses production-capacity indexer bounds and otherwise exact-shape metadata; it bypasses production graph-bucket dispatch. It is not a traffic-weighted average.

Peak saved means the difference in peak allocated bytes, taking the maximum across GPUs in each visit. It includes scratch effects and is not the BF16 payload alone. Graph-private reserved memory is reported separately in JSON. Neither is the entire serving graph-pool savings or extra allocatable KV capacity.

## Fixed-shape model forward

Q is query rows per request; K is effective history including the query. TP request counts are global; DEP counts in case names are per DP rank. For `qmix`, each of 1/2/4/8 query lengths has 512 requests (7,680 total rows).

|Topology / case|Budget MiB|Baseline → chunk ms|Change|Peak saved GiB/GPU|A/A drift|Max visit sample spread|
|---|---:|---:|---:|---:|---:|---:|
|model / q1_b2048_l4k|2048|41.194 → 41.172|-0.053%|2.501|0.185%|0.252%|
|model / q1_b2048_l4k|4096|41.194 → 41.204|+0.025%|-0.000|0.185%|0.248%|
|model / q2_b2048_l4k|2048|71.845 → 71.965|+0.166%|7.503|0.051%|0.062%|
|model / q2_b2048_l4k|4096|71.845 → 71.864|+0.027%|5.002|0.051%|0.027%|
|model / q4_b2048_l4k|2048|118.934 → 119.212|+0.234%|17.507|0.037%|0.066%|
|model / q4_b2048_l4k|4096|118.934 → 119.065|+0.110%|15.006|0.037%|0.063%|
|model / qmix_b2048_l4k|2048|119.471 → 119.694|+0.187%|16.413|0.084%|0.051%|
|model / qmix_b2048_l4k|4096|119.471 → 119.496|+0.021%|14.085|0.084%|0.063%|
|model / q1_b4096_l2k|2048|72.377 → 72.482|+0.145%|7.519|0.105%|0.067%|
|model / q1_b4096_l2k|4096|72.377 → 72.362|-0.021%|5.018|0.105%|0.048%|
|model / q2_b4096_l2k|2048|119.181 → 119.470|+0.243%|17.505|0.043%|0.041%|
|model / q2_b4096_l2k|4096|119.181 → 119.333|+0.128%|15.004|0.043%|0.049%|
|dep_model / dep_b256_q8_l8k|2048|76.971 → 76.972|+0.002%|2.501|0.090%|0.355%|
|dep_model / dep_b256_q8_l8k|4096|76.971 → 77.159|+0.245%|-0.000|0.090%|59.191%|
|dep_model / dep_b512_q8_l8k|2048|131.114 → 131.171|+0.043%|7.503|0.002%|0.472%|
|dep_model / dep_b512_q8_l8k|4096|131.114 → 131.111|-0.003%|5.002|0.002%|0.727%|
|dep_model / dep_b1024_q8_l4k|2048|239.542 → 239.807|+0.110%|16.714|0.030%|0.197%|
|dep_model / dep_b1024_q8_l4k|4096|239.542 → 239.740|+0.083%|15.006|0.030%|0.291%|

Do not interpret sub-noise negative deltas as speedups. In particular, DEP 256-request/rank, 4 GiB has a 59.19% within-visit sample-median spread; its small aggregate delta is not a precise effect estimate. All samples are retained.

## DEP unequal history

Each DP rank has 512 requests × 8 query rows. History lengths by rank are balanced `[4K,4K,4K,4K]`, skew `[2K,2K,4K,8K]`, and rotated `[8K,4K,2K,2K]`. Global request count and total effective history positions are identical. Each layout is compared against its own baseline. This tests unequal K work with equal Q, NOT unequal request counts or production DP graph padding.

|Layout|Budget MiB|Baseline → chunk ms|Change|A/A drift|Max visit sample spread|
|---|---:|---:|---:|---:|---:|
|balanced|2048|131.044 → 131.079|+0.026%|0.023%|0.318%|
|balanced|4096|131.044 → 131.063|+0.015%|0.023%|0.210%|
|history_skew|2048|131.231 → 131.309|+0.059%|0.032%|33.445%|
|history_skew|4096|131.231 → 131.177|-0.041%|0.032%|0.138%|
|history_skew_rotated|2048|131.081 → 131.156|+0.057%|0.000%|0.266%|
|history_skew_rotated|4096|131.081 → 131.139|+0.044%|0.000%|0.198%|

The skew 2 GiB arm has a 33.44% within-visit spread. Its median remains recorded, but it cannot support a precise sub-percent guarantee. The rotated case is less noisy. No per-collective profiler was run, so this does not isolate EP waiting time.

## Actual adaptive serving: all three arms

A1/B/A2 run on the same node per topology: A = chunk disabled; B = 2 GiB. DSpark7 adaptive verification is enabled; synthetic random prompts, temperature 0, ignore EOS, prefix caching off. Fixed 16 GiB KV/GPU. Both arms temporarily bound only startup profiling at 2 GiB before restoring the real budget for capture/serving. This excludes extra-KV-capacity benefits. There are two request waves per measured workload after a separate short warmup.

TP4: max sequences 2048, max batched tokens 16384, captures through 16384. DEP: max sequences 1024/rank, max batched tokens 8192/rank, captures through 8192. Client concurrency is global offered concurrency, not proof of achieved simultaneous decode batch. Adaptive profiling is performed separately on each boot.

|Topology / offered concurrency|Input / output tokens|A1 tok/s|2 GiB tok/s|A2 tok/s|B vs A1 / A2|A/A drift|
|---|---:|---:|---:|---:|---:|---:|
|serve / 1024|1024 / 256|8015.0|9329.3|9424.7|+16.40% / -1.01%|17.59%|
|serve / 2048|1024 / 256|8132.1|8970.0|8801.8|+10.30% / +1.91%|8.24%|
|serve / 2048|128 / 1024|9831.0|7722.7|8974.5|-21.45% / -13.95%|9.54%|
|serve / 2048|4096 / 256|3571.1|4060.2|3838.7|+13.70% / +5.77%|7.49%|
|serve / 512|1024 / 256|6650.9|8188.1|8725.9|+23.11% / -6.16%|31.20%|
|dep_serve / 1024|1024 / 256|14360.8|14269.8|15884.8|-0.63% / -10.17%|10.61%|
|dep_serve / 2048|128 / 512|36402.6|32126.0|33452.1|-11.75% / -3.96%|8.82%|
|dep_serve / 4096|128 / 512|31376.8|26369.1|30099.1|-15.96% / -12.39%|4.25%|

A1/A2 variation invalidates a universal speedup claim. Throughput below both baselines at TP4 c2048/128-in/1024-out and DEP c2048/c4096 is an unresolved unfavorable signal, not explained away by fixed-shape results. A causal kernel attribution and stable magnitude are still missing.

|Topology / workload|Mean TPOT A1 / B / A2 ms|P99 TPOT A1 / B / A2 ms|
|---|---:|---:|
|serve / c1024_i1024|88.27 / 76.65 / 76.28|215.80 / 180.97 / 184.01|
|serve / c2048_i1024|108.28 / 98.15 / 99.93|243.00 / 220.23 / 224.01|
|serve / c2048_i128|194.53 / 245.82 / 212.54|224.11 / 271.50 / 238.81|
|serve / c2048_i4096|53.64 / 46.98 / 49.51|133.72 / 114.85 / 123.68|
|serve / c512_i1024|53.77 / 46.94 / 42.84|125.75 / 108.24 / 98.48|
|dep_serve / c1024_i1024|45.10 / 50.32 / 45.24|103.63 / 116.14 / 103.11|
|dep_serve / c2048_i128|46.63 / 53.91 / 49.15|99.81 / 114.41 / 105.38|
|dep_serve / c4096_i128|110.89 / 129.28 / 115.78|221.22 / 261.09 / 232.57|

These are request TPOT statistics, not per-token ITL quantiles. All measured client runs completed with zero reported request failures. Success is not a model-accuracy score.

### Observer limitations

The old serving observer reads a phase file each adaptive call and writes aggregated JSON every 32 calls. Its reported CPU counter does not include all flush I/O. Overhead was not isolated, so it remains a possible measurement perturbation. Phase labels also include the client’s untimed single-request probe; histogram medians/fractions therefore are NOT timed-window workload distributions. Timed throughput comes from the client’s independent timed interval. Per-rank maxima cannot be summed into simultaneous global concurrency. These limitations do not erase the unfavorable timings.

## Correctness and eager limits

All reported fixed-shape outputs are finite. Full four-rank numerical receipts are retained. Some larger TP4 and DEP cases differ across repeated unchunked baselines and/or repeated replay of the same graph; no universal final hidden-state or generated-token bit-equivalence claim is made. This does not prove all candidate differences are harmless. See CORRECTNESS_SUMMARY.json. Prior producer score bits, exact top-k score tests, and limited real-text smoke tests are documented in the earlier evidence package, not rerun here.

Generated-text string hashes and equality rates are preserved for serving; token IDs were not collected. Random-prompt throughput is not a reasoning/accuracy evaluation. New long-output representative workload correctness/performance remains unmeasured.

The new mixed graph/eager runs contain substantial eager baseline drift. Their raw eager samples are retained, but apparent large gains (including the largest DEP shape) are not accepted as causal improvements. The prior dedicated interleaved eager experiment remains separate evidence.

## Prefill/decode budget follow-up

The proposed shared budget is audited in [PREFILL_DECODE_BUDGET.md](PREFILL_DECODE_BUDGET.md). Current prefill accounting is FP32 while PR2 decode accounting is BF16; equal numbers do not prove shared storage. This remains code analysis and a proposed experiment, not a measured performance result.

## Current decision and missing evidence

- Keep PR #64806 draft. Keep the implementation and configurable 2 GiB value unchanged while reviewing the data; do not call 2 GiB optimal or production-validated.
- User priority is normal concurrency with several thousand generated tokens. No new GPU run is started by this consolidation. The canceled 256-output proposals are explicitly not validation.
- Request-count-skewed DEP with real graph padding, stable long-output A/B, and the source of adaptive-serving variation remain open.
- Do not erase extreme-concurrency regressions when normal-workload evidence is eventually added. Conversely, do not extrapolate those regressions to every deployment.

## Upstream workload references (research only)

[Merged #58466](https://github.com/vllm-project/vllm/pull/58466) uses SPEED-Bench throughput_2k, a 2048-token output cap with natural EOS, and concurrency through 256. It reports boot-to-boot adaptive cost-curve variation and a benchmark-only fixed-curve control. That is a diagnostic lead, not proof of this run’s cause. Natural EOS means the cap is not a guaranteed 2048 generated tokens.

[Merged #63886](https://github.com/vllm-project/vllm/pull/63886) tests V4.1 + DSpark on four GB200 GPUs with SPEED-Bench low_entropy/throughput_2k, natural EOS, concurrency 1–256, and AV on/off. Its model settings differ; we have not reproduced it. These references support replacing the proposed 256-output workload, not borrowing their performance conclusions.

## Reproduction and receipts

- MODEL_RAW.json.gz: complete JSON timing/correctness records and identities for four ranks in all three completed model jobs; no tensor payloads.
- SERVING_RECEIPTS.json.gz: aggregate and per-request numeric results, generated-string hashes, and driver completion records. Full generated text and per-token ITL arrays remain local and are omitted from this compact public package.
- SOURCE_MANIFEST.json: relative source paths, sizes and SHA256 of the source receipts. Large hidden-state .pt files remain local.
- EXPERIMENT_SCRIPTS.tar.gz: executed scripts/configs, identity records, commands, interpretation notes, and the canceled follow-up setup (explicitly labeled). Not a production-code patch.
- Existing source package: [initial completed evidence](https://github.com/Pb314314/vllm/tree/e3a7e2474b3f749854df211fde16febf3f0da4ad/benchmarks/kernels/indexer_chunk/evidence/2026-10-30). Older default recommendations are superseded by the qualification in this report, without overwriting their measurements.

AI assistance was used to organize and analyze this evidence. Human review remains pending; this update does not request re-review or mark the PR ready.
