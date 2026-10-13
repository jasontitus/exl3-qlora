# Frozen-prefix experiment: process review and decision rules

2026-10-12. Source and saved evidence audit plus CPU checks only. No new GPU result
is asserted by this review. The original failed first update and strength probes
remain unchanged.

## Findings

1. **No identified target-shift, freezing, or optimizer-sign defect.**
   `training.py:27` selects hidden positions `loss_start-1..end-1` to predict the
   original supervised target positions `loss_start..end`, and evaluation uses
   the same mask. The previous full-model pre-update NLL and fresh base NLL agree
   within 2.24e-8. Frozen native linears call `freeze()` (`native.py:29`), while
   only adapter A/B names enter the optimizer. Saved step1 moments and B values
   already agreed with the configured Adam rule. This supports these specific
   contracts; it does not prove the composed 40-layer surrogate gradient.

2. **The adapters already have FP32 parameters.** `native.py:58` initializes A
   and B with MLX's default FP32 arrays. Inherited MLX-LM LoRA computes its
   residual with these FP32 weights, casts the residual to the activation dtype,
   then adds the base. Simply advertising FP32 adapter parameters or moments is
   not a new remedy. A separately recorded variant that adds the FP32 residual
   to a promoted base and then casts the sum can eliminate one rounding point
   while preserving the BF16 output and zero-B base. Returning FP32 output would
   change downstream base arithmetic even at zero B and confound that test.

3. **Low starting loss does not establish that the optimization is learned.**
   The CPU tokenizer audit verifies that all 811 targets are exactly the raw
   completion's 810 tokens plus EOS. There are 208 diff-header, 208 unchanged
   context, 105 removed-code, 175 added-comment, 114 added-code and one EOS token.
   Thus 64.2% are headers/context/removals; only 14.1% are classified as added
   code, and those include copied syntax. An ordered prompt-answer token
   alignment matches 356 tokens. This makes copying a plausible explanation for
   low mean NLL, but per-token losses are needed to establish its contribution.
   The machine-readable audit and target categories are
   `metal-fit-data-audit.json`; the reusable CPU tool is
   `scripts/audit_metal_fit_data.py`. These categories are diagnostic additions,
   not a replacement for the full answer loss or an execution-quality metric.

4. **The previous strength series does not isolate a learning-rate cause.**
   Every tested positive and negative scale worsened the original row's NLL.
   MLX Adam's default `bias_correction=False` makes its first sign-like update
   approach 3.1623 times the specified learning rate. That is a known optimizer
   choice, not a sign error. Changing it requires an explicit fresh optimizer
   contract; selecting a favorable scale after seeing losses cannot retroactively
   turn the original update into a successful run.

## The discriminating experiment

`scripts/diagnose_frozen_prefix.py` restricts trainable adapters to block39's
existing wq_a/wo_b A/B tensors. It captures h/pre and all four shared attention
state leaves before block39 during an ordinary full-model evaluation, preserving
the complete sequence. It then replays the actual final block, final collapse,
norm and vocabulary head and requires exact base NLL/logit-digest parity before
publishing the cache. It materializes and checks exact serialized tensor equality.

This is valid final-block SFT because blocks0–38 are frozen. The prefix cache
contains their actual full-context values; it is not truncation or a surrogate
target. Block39 has no indexer, compressor or Engram source. Its ordinary attention
and MoE forward still run every step; in particular, final-block MoE routes are
not fixed when adapter outputs change. The original full-model step1's block39 A
tensors can be imported while B is zeroed, avoiding the otherwise different A
initialization produced by a smaller adapter-layer selection.

The script records the unchanged full answer loss, per-token diagnostics when
requested, the first raw gradient tensors, actual Adam state and adapter
checkpoints after every update, and optional final uncached evaluation. The cache
and exact optimizer continuation bind data bytes, script/engine/runtime identity,
model metadata and weight-file stat fingerprints, rank/alpha, decoder chunk and
layer selection. Weight stats are not full payload hashes or cryptographic
authentication. Corrupted/incomplete caches and missing parity fail closed.

The requested primary row1 in `artifacts/loss-fix/three-metal-changes.jsonl` was
CPU-checked to equal the original prepared row exactly: 5,809 serialized tokens
and 811 supervised targets. Default Adam bias correction remains explicitly false;
`--bias-correction` opts into a distinct recorded optimizer configuration. Runs
with several selected rows cycle one actual example per update and evaluate all
selected rows. Extra diagnostic rows are not automatically independent held-out
examples.

## Interpretation gates

- Before any quality claim, require exact cached/uncached base parity, finite
  raw gradients and optimizer state, and a saved real update. A cache pass alone
  is not a learning pass.
- If the same-row first gradient for block39 disagrees with the original
  full-model step1's reconstructed gradient under the same A/base/engine, compare
  masked hidden/head cotangents and recomputation next. Restricting upstream
  trainability should not change this local derivative mathematically. Floating
  reduction differences must be measured rather than assumed to explain it.
- If final-block updates lower full-mask NLL and a fresh uncached evaluation
  reproduces them, report a useful final-block training configuration. It does
  not establish the root cause of the all-layer regression. Multiple updates and
  changed adapter scope differ from the previous one-step strength experiment.
- Report added-code/comment and copied-context losses alongside the primary
  full-mask loss. Improvements only in scaffolding are weak evidence of learning
  the edit. Do not select a subset after seeing its scores to redefine success.
- Do not promote a trained adapter that remains worse than the zero-B base.
  Preserve all attempted steps and configurations, including negative results.
  Any adaptively selected best training checkpoint remains a development result.
- Genuine Metal optimization requires generated edits that apply to a pinned
  parent and pass the matching trusted build/correctness/performance harness.
  Teacher-forced token improvement or exact replay of a seen patch does not prove
  generalization or speedup. Lineage-separated unseen tasks remain necessary.

## Verification performed here

Ten CPU-only cache/row tests pass, including source/model/data mismatch, incomplete
publication, missing rows/hashes, result mismatch, payload tampering and symlink
rejection. Ruff passes. The row1 describe-only path confirms the real dimensions.
GPU capture/training is operated sequentially by the parent; no success result is
invented here. Only new scripts/tests/reports were written; engine source and
historical project evidence were not edited by this review.

## Addendum: first-gradient comparison

After the parent produced the first real cached-prefix update, the CPU-only
`frozen-prefix-first-gradient-comparison.json` audit compared its saved raw
gradients with original full-model step1 gradients reconstructed from Adam's
first moment divided by 0.1. The original gradient norm was below the clipping
limit, so its recorded gradients were not clipped. Original checkpoint and cache
checksums were verified, original A was identical, B was zero, and the complete
5,809-token context and 811 targets were identical.

Both B gradients pass the predeclared relative-L2 threshold 1e-5 and cosine
threshold 0.99999999: relative L2 is 4.107e-8 for wo_b and 4.242e-8 for wq_a;
minimum cosine is 0.999999999999997. Maximum absolute difference is 1.455e-11.
Both A gradients are exactly zero. These differences are consistent with the
rounding retained in saved FP32 Adam moments. The cached/uncached base NLL was
exactly 0.5012578592107731, matching the original fresh base evaluation.

This is strong evidence that restricting trainability and caching the frozen
prefix preserved the final-block derivative. It is not a test of the original
upstream-layer derivatives, proof of the all-layer regression's cause, or evidence
of generated kernel quality. Post-update fitting results and uncached final
verification remain separate parent-run evidence.
