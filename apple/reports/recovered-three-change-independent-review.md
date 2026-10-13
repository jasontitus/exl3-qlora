# Independent review of recovered three-example fitting results

CPU-only review completed after the adapter retry exited successfully. No model execution, GPU workload, engine edit or matched-control evidence change was performed. The machine-readable receipt is `recovered-three-change-independent-review.json`.

The recovered `fit.json` reassembled exactly using the unchanged pinned runtime-v3 validators. All154 source bindings in the final summary and34 recovery-input hashes matched. Baseline PID49558 and adapter PID64418 were distinct; the adapter launch receipt bound the actual retry. The completed80GiB baseline was reused, and only the adapter child used72GiB; reserve8GiB and all math/cache/data/checkpoint settings remained fixed. Both failed starts remain recorded. The final12-step checkpoint was selected before fresh scoring.

All three complete training rows, totaling2,892 supervised tokens, passed the exact per-row supervised-logit comparison between warm training-mode scoring and fresh uncached adapter evaluation. The three examples share one lineage and were each presented four times. There are zero held-out examples.

I independently recomputed every row/category from the raw per-token NLLs and predictions using the bound lexical category arrays. The final summary and category totals match:

| Measure | Baseline | Adapter | Interpretation |
| --- | ---: | ---: | --- |
| Token-weighted answer NLL | 0.3555532213 | 0.3262765797 | 8.2341% reduction on trained answers |
| Macro-average row NLL | 0.3718079316 | 0.3265421289 | 12.1745% reduction; every row improves overall |
| All-token top1 | 2670/2892 | 2679/2892 | Teacher-forced predictions |
| Added-code NLL | 0.0622445520 | 0.0590366394 | Improvement concentrated in row2 |
| Added-code top1 | 1501/1527 | 1502/1527 | One corrected token, no added-code argmax regression |
| EOS top1 | 1/3 | 0/3 | A regression despite lower aggregate EOS NLL |

Added-code NLL **worsened** slightly on row0 (+0.0004029397) and row1 (+0.0004738208), while row2 improved (-0.0035412402). The one added-code top1 correction is row2 answer-token249: baseline predicted `const`, adapter predicted the target `if`. Correct preceding answer tokens were supplied; this is not an autonomously generated or tested kernel.

Row1's final EOS target lost top1; the new argmax is `diff`. This does not establish that free generation will fail to terminate, but it is a concrete teacher-forced termination regression and must remain visible. Removed-code NLL also rose slightly across the three rows.

Diff headers account for47.2664% of total NLL decrease, comments44.8457%, and added code5.7855%. Thus92.1121% of the measured gain is headers/comments. These are coarse lexical assignments with boundary-token rules, not semantic correctness categories. Aggregate fitting success must not be presented as all-category improvement or useful kernel quality.

No concrete receipt-integrity blocker was found for the stated three-case training/checkpoint/reload/cache comparison. The evidence does not certify generalization, autonomous patch generation, compilation, numerical parity of a generated kernel, speedup, or unique diagnosis of the earlier all-layer regression. Checksums establish local evidence consistency, not independent execution attestation or a full frozen-weight payload audit.
