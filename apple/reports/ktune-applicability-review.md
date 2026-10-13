# Kernel-training applicability review

Later-results pointer (2026-10-12): this body preserves its review cutoff. The
[completed one-row diagnostic and BF16 own-mode result](last-block-training-result-and-recipe.md)
and [active normal three-change recipe](standard-three-change-run.md) supersede
pending-result wording below. The launched normal run uses protocol v2,
runtime-v3, an explicit 8 GiB reserve and all 12 checkpoints retained. Its fresh
three-row fitting/cache-parity results are still pending.

Date: 2026-10-12. Scope: read-only review of the staged importer, project CLI,
training-fit scoring and actual prepared full-parent pilot. The reviewer also
implemented the importer; this is an adversarial effectiveness pass, not an
independent second-author code audit. The separate ktune review and backend
review cover their respective implementations. No model, GPU, candidate build
or benchmark was run for this review. Only documentation was changed.

## Assessment

The current corpus/CLI can support a reproducible **conditional historical-edit
fitting pilot**. It cannot establish that the model became better at discovering
useful kernel optimizations. The scope labels are appropriately conservative:
`target_selected_complete_parent_files`, `training_only`, and
`quality_certified=false`. Keep those labels in the final pilot result even if
answer likelihood improves or the historical patch can be reproduced.

The first complete update passed execution/checkpoint checks but **failed its
fresh training-fit check**: base NLL 0.5012578592 became 0.5053724668 with the
adapter (+0.0041146076; lower is better). The changed logits are evidence of an
adapter effect, not improvement. The original trial is preserved; separate
inference strengths +0.1, -0.1 and adaptive +10 also worsened NLL. This bounded
series is complete; no tested adapter was promoted as an improvement. The
[aggregate](ktune-strength-diagnostics.json) preserves all scores and derivations.
Native/head and sparse-attention component checks pass within
their declared scopes; they do not establish full-model gradient correctness or
fix the failed fitting result. See the [diagnosis update](first-step-diagnosis-update.md)
and [first-fit receipt](ktune-full-parent-first-update-fit.json).

Practical inference is also unfinished: the current generation path reruns the
entire 40-layer prefix for each new token. It is unsuitable for routine generation
of long kernel patches at this context length. A usable training-to-generated-
kernel workflow needs an adapter-compatible inference integration with efficient
KV caching as well as the separate correctness/performance harness.

The actual selected record is
`official:qwen38-125b-a6b-mlx-v1:1a621870-4993-4c43-9472-6f7d8f745140`.
It has 5,809 serialized tokens, with 4,998 prompt tokens and 811 supervised answer
tokens. Its complete diff hash is
`b70bc4342d40df45ae315cd5d49ec3cfa78483804eaf8c00db38592e658ac4bf`.
The importer verified this diff against parent
`b37385f8c93ab5f2d787fbf468fd6a347c71cb92` and head
`0850f7ffee84ca61d0fe439782ec59f5e5101d30`, and preserved complete touched parent
files. The prepared dataset has one training row and zero validation/test rows.

## Findings and concrete consequences

1. **Training hardware differs from the kernel's execution target.** The archived
   receipt reports `engine_device=m5` and
   `baseline_box=m5-max-128gb-1-qwen38-125b-a6b-mlx`. The parent Swift code explicitly
   requires `supportsNAX`, a GPU architecture generation check, and macOS 26.2 or
   newer before entering the changed path. The M1 Ultra can train on the source
   text, but an M1 fallback run cannot validate this M5-specific optimization.
   Future execution must prove dispatch reached the intended path on compatible
   hardware. The guide now distinguishes training host and kernel target.

2. **The desired decision is under-specified.** The generic prompt requests an
   Apple Silicon optimization and shows target-selected complete files. Source
   guards constrain shapes/dtypes, but the actual workload distribution, hardware
   target and baseline profile are absent. The target explains a 40-core GPU,
   1024-row workload and occupancy benefit. Learning this answer cannot show
   that the model selected the transformation from sufficient pre-change evidence.
   A prospective task needs those constraints independently frozen before any
   candidate or historical answer is shown. This is a corpus-design limitation,
   not something a larger sequence limit fixes.

3. **Whole-answer NLL is dominated by material other than the change.** A direct
   character/line audit of the preserved 2,341-character target found 10 metadata
   lines (593 characters), 17 copied-context lines (706), 7 added-comment lines
   (559), 3 removed-code lines (229), and 4 added-code lines (254). Metadata,
   context and added comments constitute 79.4% of characters; added executable
   code is 10.9%. These are character counts, not token-loss weights. They show
   why lower mean answer NLL is compatible with better formatting/copying and an
   incorrect coordinated edit. Do not make code-quality claims from NLL alone.

4. **The fit CLI evaluates a subset.** The default is one immutable training
   row, with available/evaluated counts and selected-row identity in the receipt.
   This is correct for the one-row pilot, but the default on a full 52-row import
   still checks only its first row. After this finding, the CLI gained explicit
   `--fit-examples N`; later `ktune verify` uses that recorded setting. The guide
   now documents it. A dataset-wide fit statement requires N to cover all rows;
   even that remains training fit rather than generalization.

5. **The selected data are success-biased and target-conditioned.** Complete
   parent files avoid truncation and added-line prompt leakage, but their file
   selection still comes from the target. Official successes are selected using
   historical outcomes; most failed/correct-but-slower attempts live only in the
   ledger. This is useful positive SFT but gives no coverage guarantee or learned
   stopping/rejection policy. Neither random row splitting nor reusing the mixed
   ktune lesson store creates a clean evaluation set.

6. **Archive speedups are not incremental gains of this patch.** The pilot's
   archived 2.04x decode / 1.53x prefill ratios compare the accumulated candidate
   against a fixed baseline. They must not be attributed to this tile change.
   The receipt audit already discloses this. A new candidate's useful improvement
   must be measured against the specific frozen parent/current baseline under a
   predeclared workload.

7. **Backend usability remains a separate acceptance gate.** At inspection, the
   first full-parent training attempt had status `interrupted`, no completed
   checkpoint, and an exit-75 receipt: 32.56 GiB observed against a 32 GiB process
   budget. This is a recorded failed feasibility attempt, not proof the 5,809-token
   recipe completed. A later 32 GiB attempt completed all forward layers and
   failed during the first backward block at 32.34 GiB, also before an update.
   The new project, `full-parent-pilot-development`, has now completed one
   update with an 80 GiB cap, explicit 12 GiB reserve, 8 GiB packed cache, decoded
   caching off and explicit chunk 8192. Training took 1,862.1 seconds and peaked
   at 73.95 GiB MLX. The parent's [CPU audit](ktune-full-parent-checkpoint-audit.json)
   found 160 finite adapter tensors (3,153,920 parameters), all 80 LoRA B tensors
   nonzero from zero initialization, 322 finite optimizer tensors, matching file
   checksums and Adam step 1. Fresh scoring then found worse answer NLL:
   0.5012578592 base versus 0.5053724668 adapter. This first update passes the
   operational gate but fails the fitting gate. The 27.66 GiB observed-process
   reading was recorded at the end, not the peak.
   **5,809 tokens is the largest complete training example tested.** The earlier
   synthetic final-block probe passed at a 58.76 GiB peak with a different,
   explicit 16 GiB reserve. No 32K memory or full-model numerical parity claim
   follows from either result or the configured ceiling.

8. **Likelihood verification does not provide a practical generated-output
   workflow.** `workflow.generate` calls `hidden_states` on the entire growing
   prefix for every new token; there is no persistent decode KV cache. The later
   5,809-token baseline completed its 5,808-position forward in approximately
   835.3 seconds before failing in backward. The successful high-memory pilot's
   forward took about 584.2 seconds with different cache/chunk settings. Neither
   is measured generation latency or a controlled full-update speed comparison;
   full-prefix repetition makes an 811-token historical patch—or
   another completion of hundreds of tokens—impractical in the current engine.
   `ktune verify` deliberately scores the known answer with teacher forcing:
   one backbone forward per selected example for each base/adapter condition,
   followed by masked token NLL. It cannot prove that the adapter generates a
   usable patch. Efficient, adapter-compatible KV-cached inference is a remaining
   integration requirement. Compatibility with `mlx-serve` has not been established;
   naming it as a possible future integration does not close this gate.

9. **Operational success is currently slow and covers the short end of the
   accepted corpus.** The complete 5,809-token update took 1,862.1 seconds,
   approximately **3.12 serialized example tokens/s** or **0.44 supervised
   answer tokens/s** (811 supervised tokens). These rates divide one complete
   example by its whole update time; they are not generation throughput.
   Fresh full-context teacher-forced scores took approximately **523–570 seconds
   per condition** in the observed runs. A routine base/adapter check therefore
   costs roughly another 18 minutes for this example, and it still does not
   generate a patch.

   The 24 full-parent records admitted by the format/evidence audit have a
   median length of **17,806.5 tokens**, over three times the tested 5,809, and
   the maximum admitted length is 25,486. The tested update already peaked at
   **73.95 GiB against an 80 GiB cap**. These observations do not establish that
   the remaining examples fit; memory does not have a validated extrapolation
   here. Import eligibility and `ready_for_sft=true` are not RAM-feasibility
   certification. Arbitrary long-patch generation remains a separate unvalidated
   practical-use path because it recomputes the complete prefix for every token.

## Minimal next gates

| Gate | Acceptance evidence | What it does not prove |
| --- | --- | --- |
| Complete-example import | Exact parent/head/diff bindings, no truncation, prompt+target+EOS counts, protected lineage policy, reproducible selected IDs | Task independence or improvement |
| Long-context update | Finite loss/gradients, intended adapters changed, frozen base unchanged, memory within guard, complete update checkpoint and fresh reload/resume | Independent full-model parity or generated code quality |
| Practical adapter inference | Verified adapter/model semantics in an efficient KV-cached decoder, measured long-context completion time and complete output receipts | Kernel correctness or improvement |
| Conditional edit output | Same complete prompt and generation budget for base/adapter, full raw token/EOS/truncation evidence, strict application/replay to pinned parent | Correctness or speed |
| Runnable task | Independently frozen hardware/workload/build/dispatch task card, protected correctness harness, compatible device and actual fast-path execution | Benefit from training |
| Correct optimization | Build plus numerical/state correctness over declared cases; controlled paired timing against the exact parent; practical gain above noise with separate confirmation | Generalization across kernel families |
| Training benefit | Paired base/adapter results on independent task families with identical candidate/tool/revision budgets, all failures retained | Universal speedup or support for untested hardware |

A later generation demonstration can remain a training-case replay, but should
follow a practical inference integration and be labeled explicitly. The present
teacher-forced fit check is the bounded way to inspect learning without launching
hundreds of full-prefix forwards. Reproducing a recorded solution would still not
establish performance on a new optimization problem.

## Lightweight tests before more expensive model runs

- Keep the existing CPU parser, hash, lineage, length, SSH-bound and edit-replay
  tests. Add a task-card validator that rejects missing target hardware, shape/
  dtype/layout distributions, numerical tolerance, parent revision, allowed paths,
  benchmark command and baseline identity before launching an optimization task.
- Add target applicability receipts: the benchmark must state which dispatch
  branch ran. Unsupported-device fallback should report `not_applicable`, never
  a successful optimized kernel.
- Exercise grading with controlled bad candidates: an unchanged/comment-only
  patch, a compilable but slower patch, an invalid/numerically wrong patch with
  an attractive timing, and a patch that edits tests or timing. None can satisfy
  the complete correctness-plus-performance gate. Include failures in the task
  denominator.
- Add separate format/application and behavioral results. Whole-patch exact
  match is a replay diagnostic; alternative correct faster implementations must
  remain valid outcomes. A strict full-output budget/EOS check must reject
  incomplete candidates even if their first lines match.
- If inspecting likelihood changes, optionally report added executable-token
  likelihood or predeclared correct-vs-broken patch preferences alongside full
  answer NLL. This is a diagnostic only; it cannot replace execution. Avoid
  selecting favorable examples or checkpoints using the final benchmark.
- Use a small predeclared development panel first, including at least one
  different lineage and one no-improvement/negative control. Do not call a tiny
  panel statistically representative. Keep a later final set and any derived
  retrieval content untouched during development.

## Documentation/report accuracy

The current guide correctly distinguishes context ceiling from measured memory
and states that import/fit are not quality certification. It now states the
M5/NAX execution target and first-row fit scope explicitly; the existing exporter
section was preserved.

`metal-training-assessment.md` describes the earlier tool before history import
and with a 512-position limit; an opening historical banner was added in this
review. `long-context-design-review.md` has since recorded the corrected focused
suite after its initial 23-pass/2-failure result. The separate
`bounded-moe-review.md` records the subsequent independent code review and the
parent-reported 432-test plus 27-subtest full-suite result. Their methodological
cautions remain useful. Old short-context quality receipts also predate the documented
FP8 exponent fix and must not be silently reused as numerical proof for the new
engine semantics.

Evidence inspected: staged `ktune_data.py`, `ktune_cli.py`, `sft.py`,
`KERNEL_TUNING.md`; the actual project and immutable imported records under
`artifacts/ktune-20261012/full-parent-pilot`; associated memory-abort receipts and
the recorded official evidence. No new trained-output quality measurement was
performed in this review.
