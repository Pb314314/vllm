# PR checklist findings

Applied the repository `pr-checklist` skill on the final chunk patch. AI assistance was used. This is a review-ready code branch, not a claim that all publication gates or human review are complete.

## Design fit

- Scope is a caller-level SM100 MXFP4 varlen decode memory bound. No scheduler/model-runner redesign or new selection kernel. The existing per-layer compressed capacity determines row count.
- 202 production lines added, no deletions. The per-chunk helper deliberately preserves the existing decode scoring/selection sequence while leaving unsupported hardware/cache/parallel paths untouched.
- No device tensor is read back to decide chunking; only static shapes, configuration and host metadata are used. FULL graph capture/replay passed on actual serving workers. Eager and graph tradeoffs are both retained.
- Dense/source paths only; compact candidate consumers are excluded. Auto provider rules are unchanged, although its existing row-count decision can choose cooperative for a small chunk.
- Related #62169 covers SM90/SM120 uniform requests and explicitly excludes SM100 varlen. Area duplicate searches were refreshed with `gh pr list --repo vllm-project/vllm --state open --search 'indexer chunk'`. No linked issue was identified, so no issue-specific search is represented as completed.

## Validation

- Common corrected BF16/FP32 native build: 451 existing kernel tests passed.
- Final planner/profile tests: nine passed. Buildkite tethering checks passed for the extended existing suite.
- Broad 21-case sweep, final 10-case budget refinement, six tail-policy comparisons, four allocator-order comparisons and focused source/empty-row cases completed with exact score/top-k gates.
- Candidate selector without DeepSelect: initial observer omitted the PyTorch path, failed before recording a valid baseline; observer repaired and job41498 passed. This changed benchmark observation only. Failed logs are retained.
- V4 collocated, V4.1 collocated and V4 two-node NIXL PD A/B/A final-answer/protocol checks passed; B arms prove actual chunk graph replay on all four workers. These use forced small budgets and eight unique questions repeated, not standard model-quality scores or default-budget throughput tests.
- Whole-model target-forward tests use real weights, synthetic tokens and zero historical KV. Scores/selection correctness is verified independently. Generated token identity and whole-model hidden bit equality are not claimed.
- Final 1/1.5/2/4 GiB model scan completed. Graph results are stable; eager results disagree across scan protocols. Interleaved same-batch eager follow-up41510 completed: the largest workload costs +3.34–3.73% at2GiB and +1.38–2.16% at4GiB; the smaller workload still has baseline drift, explicitly retained.
- Real max-capacity service startup: unchunked A failed with negative KV-cache headroom after 33.68 GiB graph profiling. Same-config B at2GiB passed health/metrics, with11.66GiB profiling capture and10.00GiB final capture. KV headroom62.61GiB includes a profiling-reservation correction; the report separates this from actual capture savings.

## Code quality and documentation

- Final PR2 and dependency commits passed pre-commit, including type/style/docs/CI tethering. The fallback observer received an additional passing check.
- New budget is documented in the memory guide and benchmark README; zero-off and 4 GiB tuning are explained.
- The native address fix was clang-formatted; its tokens match the tested source modulo whitespace. The final env file differs from the measured snapshot only by removing obsolete “Opt-in” wording in a comment; AST identity and both hashes are recorded. The caller is byte-identical.
- The three patch files were applied in sequence to the PR1 base and reproduced all 11 final PR2 file hashes.

## PR contents and publication gates

- English PR description, detailed report, formula, execution flowchart, plot, commands and raw evidence are prepared. Keep model-forward, serving correctness, startup memory and throughput claims distinct.
- The base requires reconciled #64138 + #62172 and a separate two-expression address-width correction. Online dependency heads are not a conflict-free textual merge. The fork review branch isolates these in separate commits.
- All validation runs are complete; final evidence is exported separately from the production diff. Human review/test confirmation and upstream dependency integration remain.
- The human submitter must review every changed line and run relevant tests, as required by AGENTS.md line28. Confirmation is still pending; a new upstream PR has not been opened.
