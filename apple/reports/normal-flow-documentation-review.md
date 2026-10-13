# Normal-flow documentation and effectiveness review

October 12, 2026, during the standard three-change run. This review changed documentation and the separately authorized post-run CPU auditor. It did not run a model, alter the active project or modify pinned runtime-v3.

The current [normal-flow recipe](standard-three-change-run.md) records the launched source/configuration and leaves final metrics pending. The [one-row result](last-block-training-result-and-recipe.md) remains a distinct experiment. The new [auditor receipt](standard-fit-auditor-hardening.json) binds its later version/SHA, 42 passing CPU tests and a successful audit of the actual prepared case mappings; it is not a claim that three-row fitting has finished.

| Finding | Resolution or remaining gate |
| --- | --- |
| README and guides still described the first all40-adapter failure as the current overall fitting result | Added current-result pointers and labeled the failed trial/test count historical. The original loss/strength failures remain visible. |
| A proposed three-case recipe used reserve12 and current source, while the actual launch uses reserve8 and runtime-v3 | Main result recipe now uses protocol v2, the pinned runtime and the launch's explicit reserve8. Old review bodies retain their cutoff with a later-results pointer. |
| Imported/selected/trained change counts were easy to confuse | Distinguish24 eligible full-parent edits,3 selected changes in the active run, and1 distinct change in the completed eight-step diagnostic. Four exposures per row are repeated training, not extra independent examples. |
| BF16 descent could be presented as proof that activation quantization caused the old failure | Report own-mode baseline0.5061723590 and fresh post0.5014906038; the historical control has not been replayed under the new source. Ordinary BF16 casts, indexer quantizers and routing remain. No matched causal claim. |
| Generic success or `tool_verification_complete` could imply complete port/model quality | Scope the auditor flag to this three-case saved-optimizer, adapter-reload and warm/fresh-cache path. Fitting, reference-model correctness and generated-kernel usefulness remain separate claims. |
| CPU report accepted case labels without checking their source-to-token mapping | Fixed only the current standalone auditor/tests: protocol/raw hash, raw bytes, retained source row, case ID, lineage and exact token/mask hash must match. Swapped/foreign labels and altered source/provenance are rejected. |

## Remaining quality and usefulness weaknesses

1. **Three related training cases are not a test set.** They share a historical source lineage, and parent files were selected using the target edit. The prompts lack an independently chosen task/workload contract and surrounding build repository. No uncertainty claim or generalization claim is supported by this sample.
2. **Fitting may mostly improve edit packaging.** In the completed one-row experiment, comments and headers accounted for almost all NLL reduction; added-code top-1 stayed112/114. The standard audit therefore reports every row, both weighted/macro averages, code/comment/header losses and corrected/regressed added-code tokens. An aggregate gain cannot replace that disclosure.
3. **Fresh scoring verifies reload behavior within this implementation.** Component VJP tests, frozen-prefix parity and optimizer audits do not replace independent full-model logits/gradient reference coverage for the same packed model and declared numerical mode.
4. **Actual cache parity is still an outcome to measure.** Small F32/BF16 tests passed, but the final real probe must show three hits and no new misses and match the fresh cache-disabled adapter outputs. Absent/incomplete evidence stays unverified.
5. **Autonomous kernel usefulness has not been tested.** Teacher forcing supplies previous correct answer tokens. No new patch has been shown to apply, compile, pass independent numerical checks and produce repeated paired timing gains on the intended M5/NAX target. Training on M1 does not retarget the historical optimization.
6. **Inference and long-example throughput remain limiting.** Generation recomputes the full prefix for each token. The24-example set's median length17,806.5 exceeds the completed5,809-token operating point substantially; the32,768 ceiling and80GiB guard do not establish corpus-wide feasibility. The current3-case run is intended to extend measured training coverage to5,822 tokens, with results still pending at this review cutoff.

These are concrete limits on interpreting the current outputs. They do not require discarding the completed fitting result or delaying the fixed standard-run audit. The primary objective for this run remains the predeclared paired training-fit and implementation checks, preserving any regressing rows and failed evidence.
