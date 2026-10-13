# Broader upstream Metal intake: adversarial design review

Review scope: CPU-only design/process review of the staged importer and actual saved library policy. No upstream candidate was compiled, numerically tested, benchmarked, or used for model training by this reviewer. Implementation review is complete for the source hashes in `upstream-intake-independent-review.json`; actual broader-corpus counts remain separately reported. User-authorized scope includes mlx-serve, tensorfold, sushi and llama.cpp history, with eventual structured examples and execution checks.

## Release gates

1. **Do not turn reported numbers into measured verification.** Upstream commit/PR/changelog claims are legitimate source material. Preserve immutable source text, identity, attribution and the relationship to a specific edit. Label them `author_reported`; a parsed positive speedup is not a correctness receipt or an independently reproduced benchmark. `ktune_eval.py:44` already makes this distinction for the existing contest/local formats. New intake must not route ordinary claims through an official-promotion label.
2. **The model must not certify its own answer.** Generate grounded analysis and a complete edit; attach compile, numerical and timing results from the trusted harness afterwards. A completion saying `compiled=true` or `2x faster` is not verification. Synthetic explanatory analysis must be explicitly derived, with its inputs/model/prompt recorded, and cannot become an author quote or measured bottleneck.
3. **Retain the real program boundary.** Many optimization commits change host dispatch, layouts, precision, state, build settings and multiple files. Added MSL alone may neither compile nor implement the original change. Bind exact parent/head commits and source blobs; distinguish whole-commit targets from explicitly scoped file edits. Replay precisely the declared complete target. An isolated kernel task needs an explicit extraction/reconstruction receipt and dependencies; do not relabel arbitrary patch snippets as standalone programs.
4. **Quantity is counted after admission and deduplication.** Report fetched candidates, complete replayable targets, unique accepted edits, token-fit examples, evidence tiers and lineage groups separately. Commit/PR/changelog aliases, cherry-picks, plan/edit views and repeated epochs are not distinct optimization examples. If fewer than100–200 survive, report the shortfall; do not copy, split or relabel failures to reach the requested count.
5. **Protect comparisons from lineage leakage.** Maintain a persistent source/commit/patch/task/port/revert relationship registry. Related edits, their rationales, benchmark notes, generated variants and retrieval lessons must stay in the same split. Random row splitting or chronology alone does not break a shared implementation lineage. Training-only export is an honest default when independently held-out groups cannot yet be established.
6. **Execution quality is a separate promotion gate.** A source-reported edit can be useful supervised history without being part of an execution-verified set. The user's requested every-example compile/parity/benchmark standard requires a per-example trusted task/harness and actual receipts. Missing build context, unsupported hardware or absent reference implementation means `not_run`/`blocked`, not pass. No universal standalone-Metal compilation shortcut can close that gap.

## Example contract

Each example should bind repository URL, license/source metadata, complete parent/head IDs, original source artifact hashes and paths, full target patch hash, exact parent context hashes, all changed paths and declared target scope. Record aliases and lineage IDs independently of the training row ID. Preserve raw author text and prior archive admission decisions.

The structured task may contain operation, numerical contract, baseline code and dependencies, dispatch interface, shapes/layout/dtypes, target hardware and an optimization objective. Each field needs a source or `unknown`. Pre-change measurements may appear as task context when their provenance is available. Never invent a profiler result to fill a bottleneck field. The choice of changed files or hunks is answer-derived and must remain disclosed even when all code shown is from the parent.

The answer can contain author-grounded reasoning, architecture/dispatch changes and the complete functional edit. If rationale was generated later, its type is `derived_analysis`, not historical fact. Do not silently replace missing rationale with claimed expert reasoning. Keep source-reported metrics and subsequent trusted verification in sidecars/tool results, so generation quality is not graded by self-reported success.

Use independent evidence dimensions rather than one overloaded `verified` boolean:

| Dimension | Examples of states | What a positive state establishes |
| --- | --- | --- |
| Source identity | missing / pinned | Specific archived artifact and commit identities |
| Edit replay | not_run / failed / passed | Declared patch transforms the exact parent into expected source |
| Performance claim | absent / author_reported / contradictory | What the author reported, with units and scope |
| Build | not_run / failed / passed | Exact source/config built under the recorded toolchain |
| Numerical check | not_run / failed / passed | The recorded reference, inputs and tolerances passed |
| Performance rerun | not_run / failed / slower / unresolved / faster | The recorded workload/hardware/timing protocol outcome |
| Dataset placement | train / development / protected / quarantine | Eligibility under this dataset's declared split policy |

A source-only or author-reported example is not a failed optimization. A known incorrect, reverted-for-correctness or non-improving result should remain in the negative ledger, or an explicitly formulated diagnosis/repair task, never silently become a positive optimization target.

## Metrics and timing

Preserve the original metric type, direction, units, baseline/candidate identity, hardware, shape/workload, batch/context length and whether it is kernel time, throughput, end-to-end latency or a composite. Unknown fields stay unknown. A title like “30% faster” without a bound workload is a source claim, not a recomputable measurement. A10% latency reduction implies1/0.9 speedup, not1.1; do not mix tokens/s with seconds/token or geomeans with ratios of arithmetic means.

For execution promotion, freeze trusted references, numerical tolerances and measured workloads before candidate generation. Require shape/interface checks, finite values and dtype-appropriate absolute/relative tolerances; explicitly decide NaN/Inf behavior. Cover tails, nonmultiples, dispatch branches, layouts and boundary sizes applicable to the task. A candidate must not change the reference, test inputs, timing code or tolerance to pass.

Best-of-N selection must preserve every attempted candidate, invalid patch, compile error, correctness failure, timeout, unchanged/slower result and its generation/tool budget. Use equal budgets for base and fine-tuned models. Select using a development workload, then confirm the chosen candidate on separate hidden inputs and independent timing runs with synchronized/interleaved baseline/candidate measurements. Report correct-and-faster successes over **all** predeclared tasks, not just compilable winners. TuningN itself against the test set contaminates the comparison.

## Old library rules and new admission

The saved `kernel-sft/ktune-tool-snapshot/configs/library-sources.json` has35 examples and231 excluded records, including210 quarantined. Its `full_edit_training_admitted=false` flags and retrieval-only policy describe a prior experiment. The user's new history-training authorization allows a new scoped admission decision while preserving that earlier provenance. Do not treat the prior retrieval restriction as a universal ban.

Specific held-out IDs, source/commit/implementation overlaps and exposed benchmark tasks still matter. Keep them protected where possible. If an explicit new dataset policy intentionally trains on a previously held-out family, retire the corresponding old holdout claim and document that decision. Generic shared syntax alone is not proof of leakage, but unresolved overlap cannot be advertised as independence.

The current importer groups repository/lineage/commit/diff relationships in `ktune_data.py:407`; this is conservative and useful but does not discover every cross-repository port. `ktune_data.py:496` checks exact full-parent source for its existing formats. `ktune_data.py:700` currently trains prompt -> complete patch only; it does not supply the richer structured rationale or actual per-example execution receipts requested here. `ktune_data.py:218` explicitly labels hunk-selected context as patch reconstruction. Preserve that honesty when adding new schemas.

## CPU acceptance cases for the implementation owner

- One commit represented by a PR, announcement and two archived records produces one canonical target plus aliases; count remains1.
- Cherry-picked equivalent edits across repositories remain linked even if commit IDs differ; either deduplicate with recorded proof or count them as correlated views, not independent families.
- Same displayed score but changed candidate commit, benchmark scope, units, hardware or parent becomes a different/unbound claim; no automatic receipt promotion.
- Negative, zero, NaN, infinity, booleans and malformed metric values never become positive timing evidence. Missing timings may remain legitimate nonnumeric source claims.
- A commit touching kernel and dispatch cannot emit an allegedly complete kernel-only commit. Explicit scoped targets list omitted paths and cannot inherit whole-program build/benchmark claims without proof.
- Missing old/new files, ambiguous merge parent, changed Git blobs, truncated diff, new/deleted/renamed/binary sections and unsupported dependency changes fail closed or enter an explicit unsupported ledger.
- Parent prompt must not contain added target code, post-change benchmark notes or rationale unless the declared task specifically conditions on them. Test target-selected hunk disclosure separately; it is not an untouched optimization-discovery task.
- A historical “pass” record with a later bound correctness revert remains negative/conflicted for positive-optimization admission; unknown subsequent status is not fabricated.
- Old retrieval-only flags can receive a new authorized training decision without erasing the old flag; protected exact task/commit lineage cannot silently change splits.
- A held-out edit's port, rationale, generated variant or retrieval lesson cannot enter training unnoticed. Split decisions remain deterministic under source order changes and aliases.
- Oversized context or target is rejected whole with measured token count, never truncated to inflate intake count. Full supported context is distinct from the longest actually tested training length.
- Model-generated verification claims, missing or stale harness identity, altered tolerances/reference, wrong candidate hash, failed/timed-out execution or omitted candidate attempts cannot pass execution verification.
- Source text containing tool-like instructions, path traversal or shell fragments remains unexecuted data. Git queries use explicit arguments and immutable revisions; extraction never runs repository hooks, build scripts or filters by accident.
- All exported train/ledger/provenance files are atomically published, checksummed and bound to the project before training; interrupted preparation remains untrainable until recovered.

## Scale and efficacy limits

The three-row standard pilot exercises three distinct edits four times each, not12 unique examples. Those prefixes together retain about676MiB. At similar size,100–200 examples would imply roughly22–44GiB of prefix state, while the current cache is2GiB; longer rows can be substantially larger. This estimate is not a measured throughput prediction. Measure corpus lengths and representative cold-prefix/suffix costs before scheduling broad training. Do not reuse three-case warm-cache throughput as evidence of100-case speed.

Balance by real operation family and evidence scope after deduplication; do not inflate families by renaming aliases or making artificial variants. A first100–200 source-reported dataset and an eventual300–800 execution-checked dataset are distinct milestones. Report how many meet each standard. Training NLL reductions, patch syntax and memorized historical edits are diagnostic measures; held-out apply/build/correctness/performance outcomes are the efficacy measures.

## First implementation review feedback

Read-only review of the initial `upstream_data.py` identified target pollution: the completion included Git blob IDs, SHA-256 source hashes and a patch hash alongside full source. Those opaque identifiers reintroduce the arbitrary-header learning already observed in the three-case pilot. The owner removed them from rendered source/answer targets while retaining exact hashes in provenance. I also requested that unchanged dependency files remain context-only rather than copied into the supervised answer.

An initial `reported_improvement` outcome needed no bound supporting claim. The owner chose to preserve that field as an explicit curator-declared classification, without claiming semantic verification of source support. Exact quote containment proves a quote occurs in the archived evidence; it does not prove a curator's paraphrase is entailed or that reported numbers are valid. These limits must be visible in the final schema/report.

The proposed `reviewed_units` option addresses very large changed files by binding explicit before/after UTF-8 byte ranges and hashes for reviewer-declared complete functions/kernels/types/declarations. Its range checks require monotone nonoverlapping units and identical unchanged gaps/tails across every changed file. This establishes exact reconstruction coverage, but not language-level completeness or dependency closure. The latter remain reviewer assertions until a trusted task harness builds and executes the resulting program. Required negative tests include omitted files/edits, overlapping/reordered ranges, invalid or split UTF-8 offsets, missing side hashes, and new/deleted-file empty sides.

Final source and CLI review passed **70 CPU tests** (51 importer + 19 CLI) independently in 20.41 seconds. Four additional real DeepSeek tokenizer cases matched the installed CompletionsDataset token IDs and completion masks exactly without importing MLX. The tests cover full and reviewed-unit reconstruction, UTF-8 boundaries, missing/overlapping edits, new/deleted sides, protected and negative aliases, source tampering, token/EOS integrity, no truncation, and standard SFT preparation compatibility. Final source/documentation/test hashes are in `upstream-intake-independent-review.json`. No remaining concrete execution blocker was found for this source-reconstruction intake scope.

The importer does not run a compiler. The separate `ktune compile-metal` command checks a narrow stdlib-only translation-unit pair and reports AIR compilation separately; it establishes neither linkage, GPU execution, numerical parity nor speed. Its receipt cannot promote upstream examples to `execution_verified`. Actual broader-corpus counts and any later execution receipts remain independent evidence. No expanded-corpus execution-quality pass is asserted here.


## One-command integration follow-up

The additional `finetune-upstream` command reuses standard SFT preparation, mandatory provenance binding, training, stopped/completed status and optional training-fit checks. Independent review passed **75 CPU tests** (51 importer + 24 CLI) in 19.28 seconds, plus Ruff. The import must have eligible complete examples, preparation must mark provenance required, and binding must succeed before training. Prepare-only performs no updates; a stopped run is not relabeled completed or given a fit result. The new CLI/documentation hashes are recorded separately in `upstream-finetune-independent-review.json`; the importer source remains unchanged. This review does not certify real broad-corpus training or generated-kernel quality.
