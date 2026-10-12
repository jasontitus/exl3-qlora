# Final documentation and evidence review

Review date: 2026-10-12. Scope: documentation, saved verification receipts and
CPU-only command parsing. No source edits, model loading, GPU work, shell recipe
execution or generated-kernel execution. All three tested inference-only
strengths (+0.1, -0.1, adaptive +10) worsened NLL. The bounded series is complete,
with no adapter promoted as an improvement; see the [aggregate](ktune-strength-diagnostics.json).
New native/head/sparse VJP
component checks passed and the current suite has 518 passing tests plus 27
subtests. This report does not choose a new adapter or revise the original
first-update failure; see the [diagnosis update](first-step-diagnosis-update.md).

## What a new user should conclude

| Claim | Current evidence and limit |
| --- | --- |
| A full example can train | One 5,809-token example completed an update in 31.0 minutes; peak MLX 73.95 GiB. This is the largest complete training example tested here. |
| The 80 GiB profile | A process budget, with cache allocations inside it. The preset requires 32 GiB reserve and 112 GiB startup headroom; the measured full run explicitly used 12 GiB reserve and a 92 GiB startup gate. |
| 32,768 tokens | An implementation ceiling. Neither an 80 GiB budget nor successful CPU tokenization establishes training feasibility at that length. |
| The first update learned the answer | It did not pass this check: NLL rose from 0.5012578592 to 0.5053724668. The adapter changed logits and its saved state is valid; those observations do not imply improvement. |
| `completed` / exit 0 | Computation completed. The current CLI does not convert worse `loss_delta` into a failed process status. Read the NLL delta and quality scope separately. |
| Kernel quality | Unassessed. Teacher-forced likelihood supplies the answer and does not generate, compile, test or time a new kernel. Practical long-patch generation also needs compatible efficient KV-cached inference. |

The [combined verification receipt](ktune-full-parent-verification.json) confirms
fresh-base agreement with the pre-update loss within 2.24e-8, changed adapter
logits, and `training_target_nll_improved=false`. This is strong reporting
consistency, not independent full-model numerical parity. The
[original fit receipt](ktune-full-parent-first-update-fit.json) and
[checkpoint audit](ktune-full-parent-checkpoint-audit.json) remain unchanged.

## Usability fixes made in this pass

- The kernel guide now explains the operational meaning of `completed`, the sign
  of `loss_delta`, and why neither is a kernel-quality verdict. Its primary
  recipe reproduces the observed trial; continuation commands do not promise
  that another update fixes fitting.
- The beginner guide now identifies 5,809 as the tested complete length, shows
  the failed fitting outcome, and routes long kernel-patch work to the scoped
  ktune recipe. Its two-update/generation walkthrough is explicitly for short
  text, not a practical long-patch inference path.
- The quality guide leads with current long-context evidence and labels the old
  `violet` result as evidence from the earlier arithmetic/source revision.
- The high-memory review links the packaged final-block probe receipt, rather
  than a sibling workspace directory unavailable in the installed `apple` tree.

## CPU verification coverage

The [check receipt](documentation-checks.json) covers all 27 Markdown documents,
including this review: **195 local links**, **30 shell/Python
snippets**, and **28 CLI examples** parsed with the actual isolated argparse
functions, including the engine JSON-config path. Shell snippets passed `bash
-n`; Python snippets passed AST parsing. No application workload was executed.
Parsing establishes syntax, option names and required arguments; it does not
prove model feasibility, remote availability, placeholder paths or successful
execution of the recipes.

There are no unresolved link targets on this machine. Four README source links
point outside the partial staging tree into the repository's upstream sources;
they exist in the actual full checkout. Sixteen historical review links point
to existing absolute source/workspace artifacts, including line-number links.
Those older reports are local audit records, not standalone portable documents.
All current guide-to-report links and the probe receipt are within the staged
`apple` tree. Markdown section anchors were also checked.

## Practical time and corpus coverage

The complete update's 1,862.1 seconds correspond to approximately 3.12 serialized
example tokens/s or 0.44 supervised tokens/s. Fresh full-context scoring has taken
523–570 seconds per condition. The 24 format/evidence-eligible parent examples
have a median of 17,806.5 tokens, over three times the tested length. With the
5,809-token update already reaching 73.95 GiB against its 80 GiB cap, the importer
must not be read as certifying that all 24 examples will fit. The kernel guide
now explicitly distinguishes import readiness from RAM feasibility. Generation
still recomputes the complete prefix per output token and has no validated
practical long-patch throughput.

## Remaining user-facing limits

None of the three tested derived strengths fixed fitting. The completed
bounded series preserves that failure and must not be presented as independent
quality evaluation or new optimizer updates. The covered component checks
passed, but do not constitute a full-model gradient oracle. A derived inference adapter is not a resumed Adam checkpoint.
A smaller learning-rate or strength remedy must not be recommended before it is
supported by results. Higher-level full-model reference parity, longer-context
training, a continued-update check on this revision, practical generation, and
independent generated-kernel correctness/performance remain separate gates.
