# First complete-example update: fit regression diagnosis

Date: 2026-10-12. Read-only source/receipt inspection and CPU-only NumPy analysis of saved safetensors. No GPU execution and no source or adapter modifications were performed.

The completed 5,809-token example run proves that the full context can execute forward, backward, update, save, and fresh-process evaluation. Its first update does **not** pass the fit-improvement gate: NLL rose from 0.5012578592107731 to 0.5053724668388994 on its 811 supervised tokens, a change of +0.0041146076281263. The original failed fit receipt should remain immutable. A successful later diagnostic scale or training run must have its own receipt.

The pre-update training NLL was 0.5012578368186951, only 2.24e-8 from fresh base evaluation. This strongly supports consistent serialization, target masking, and forward loss accounting. It does not prove backward correctness.

## Actual checkpoint optimizer audit

The CPU-only result is `reports/first-step-optimizer-diagnostic.json`, bound to the saved adapter and optimizer SHA256 hashes. Using first-step Adam moments, the gradient was independently reconstructed as `g = m / 0.1`:

- Reconstructed gradient norm: **0.03829979971080936**, matching logged **0.0382997989654541**.
- Maximum absolute second-moment error against `v = 0.001*g*g`: **2.07e-17**.
- Every one of **1,024,000** saved B coordinates is nonzero and moves opposite its corresponding computed gradient. Wrong-sign coordinates: **zero**.
- All A first/second moments are zero, as expected with zero B initialization.
- Saved B agrees with the installed uncorrected Adam rule, `-lr*m/(sqrt(v)+1e-8)`, to maximum absolute error **6.38e-12**.
- The computed gradient predicts a first-order loss change `g·ΔB = -0.0008529185677381717`; the measured quantized forward changes in the opposite direction by approximately 4.82 times that magnitude.

An important scale detail: installed MLX 0.32.3 `Adam` defaults to `bias_correction=False` (`mlx/optimizers/optimizers.py:498,545`). The workflow constructs Adam without overriding it. Consequently the first update approaches **3.1623 × learning_rate × sign(g)** when epsilon is small relative to the second moment. Most B coordinates are approximately 3.1e-5 in magnitude at the configured learning rate 1e-5. The total B displacement has L2 norm approximately 0.03078 across all 80 matrices. This is intentional behavior of the selected MLX optimizer, not an identified sign bug. Any change to bias correction would need an explicit optimizer/engine contract change and a fresh run.

The predicted contributions are -0.0002605217 from wq_a adapters and -0.0005923968 from wo_b adapters. These provide useful subgroup directions for later isolation.

## Full-size path inspection

For the actual 5,808 input positions:

- Attention uses 64 heads, dimension 512, a 128-position causal window, and query tiles of 16. The input length is divisible by the tile size. Sparse VJP shapes and scaling do not introduce a separate long-length branch at this size.
- Ratio-two compressed KV has 2,904 entries, and ratio-one KV has 5,808. Top-512 selection is active. Candidate block pruning is **inactive**: 5,808 entries occupy 726 blocks, fewer than the 2,048 kept blocks. Bugs specific to pruning above 16K therefore cannot explain this observation.
- Main-query adapters also change the indexer's query input at index-source layers. Forward can swap selected positions; the declared backward intentionally excludes the discrete selection derivative.
- Frozen MoE selection can similarly change with adapted hidden states, while its backward differentiates the selected experts and continuous routing weights only.
- Native LoRA uses FP32 adapter arrays, but casts its residual to the activation dtype and adds it to the BF16 base result. Forty layers contain BF16 rounding, FP8 activation rounding, and FP4 compressed-KV rounding. Their declared derivatives act as identity surrogates through the casts/activation quantizers, which is not the exact derivative of a quantized staircase.
- The actual checkpoint head is BF16 `[129280,5120]`. The trunk attention projections are MXFP8 U32 matrices; compressor matrices are BF16. The frozen head's VJP therefore uses dense BF16 matmul, while the trunk additionally depends on MLX's MXFP8 input VJP.

No concrete actual-dimension indexing, transpose, target-alignment, or optimizer-sign defect was found. The custom attention VJP includes the key and value contributions of shared KV; the MoE VJP includes continuous router weights; and block recomputation passes differentiable shared KV across layer boundaries. These findings narrow the investigation but do not certify the complete gradient.

## Working hypotheses and decisive next checks

Quantization/route sensitivity and an overly large first effective step are plausible explanations, because the predicted improvement is small relative to the measured discrepancy and the update touches more than one million coordinates. They remain hypotheses; the failed fit must not be dismissed as expected noise or counted as successful learning.

1. **Same-data strength diagnostic:** keep A fixed and multiply saved B by scales such as `+0.1`, `+0.316`, and `-0.1`, retaining the already measured zero/one endpoints. This scales the actual first-step direction because initial B was zero. Use separate immutable diagnostic outputs and preserve the original checkpoint. The negative scale is diagnostic only. A lower positive scale that improves NLL establishes a usable direction for this example, but not general quality. A negative scale alone improving NLL does not prove a sign bug: a staircase objective need not locally follow its STE. Parent owns this GPU experiment.
2. **If ambiguous, measure route changes:** compare selected top-512 IDs and selected MoE experts between zero B and a chosen scale, per layer. Count changed quantized elements or changed scale groups separately. This distinguishes observed routing/rounding changes from speculation. wq_a-only versus wo_b-only scales provide additional isolation; neither subgroup fully avoids downstream route changes.
3. **Close actual-dtype coverage gaps:** the existing native MXFP8 gradient fixture is float32 with a 128×128 weight. Compare actual BF16-input wide native attention matrices against independently decoded weight transposes, and the actual BF16 head-plus-CE input VJP against an independently expressed dense reference for a few supervised rows. Preserve BF16 boundary rounding and report cosine/error norms, rather than demanding inappropriate float64 bitwise equality.
4. **Attention VJP at real dimensions:** use a bounded captured query tile with 64 heads, 512 features, 128 window entries, and 512 selected compressed entries. Compare the custom VJP against a separately expressed dense autodiff attention expression with frozen selections, nonuniform cotangents, and repeated-key accumulation. Compare components before accumulating all query tiles, then verify tile accumulation. This targets a full-size numerical kernel path without rerunning 40 layers.
5. **Surrogate directional replay if needed:** freeze reference selection IDs and quantization residuals, use an explicitly smooth float32 local replay, and compare its directional finite difference with `g·Δ`. Ordinary finite differences of the original quantized forward cannot validate an identity STE. A replay must also handle BF16 boundary rounding deliberately; merely bypassing FP8 while leaving all BF16 casts is not a smooth oracle and must not be advertised as one.

Repeated tiny component tests and a successful training call cannot replace the fresh fit gate. Conversely, one adverse post-update NLL is insufficient evidence of a custom-VJP defect without these discriminating checks.

## Addendum: completed component probes — 2026-10-12 (America/Los_Angeles)

The following completed receipts were inspected after the initial diagnosis:

- `native-vjp-actual-shapes.json`: all five actual-weight MXFP8 trunk projections and the BF16 head/CE case pass. The maximum native input-gradient relative L2 error is 4.8824e-5 and minimum cosine is 0.999999998808. The head input gradient exactly matches the independently decoded NumPy reference in this fixture.
- `native-vjp-fullm-sampled.json`: all five projections execute 5,808 rows and pass all-row finiteness checks. Eight fixed rows per projection match the independent oracle, with maximum relative L2 error 3.8268e-5. This covers full-M execution but does not numerically inspect all output rows.
- `sparse-attention-real-dim-vjp.json`: all four cases pass with H64, D512, window128, K512, query tiles16, lengths145/160, and F32/BF16. Maximum whole-tensor gradient relative L2 error is approximately 2.6368e-4. The dense expression is independent of the custom VJP but shares MLX's arithmetic/autodiff backend. Synthetic repeated selections stress accumulation; they do not exercise real top-k selection or the compressor.

These results materially reduce the likelihood of a local native-projection/head or attention transpose/dtype/VJP defect in the tested paths. They do not establish correctness of the composed 40-layer surrogate gradient, identify which discrete operation caused the fit regression, or establish parity with the native full model. Passing component probes cannot turn a measured fit regression into an improvement.

The completed strength measurements are:

| B multiplier | Training-row NLL | Delta from base |
| --- | ---: | ---: |
| 0, base | 0.5012578592 | 0 |
| +1, original update | 0.5053724668 | +0.0041146076 |
| +0.1, derived diagnostic | 0.5090817250 | +0.0078238658 |
| -0.1, derived diagnostic | 0.5019030712 | +0.0006452120 |

The nonmonotonic pattern is compatible with a nonsmooth quantized/routed forward. It is not proof of that explanation, and it does not support saying that a smaller learning rate fixes the problem. The negative direction also regressed, so it supplies no evidence of a globally reversed optimizer sign.

The parent recorded `larger-strength-plan.json` before starting the +10 probe. It accurately calls this an adaptive training-row sensitivity diagnostic, binds prior results and implementation identity, and keeps `quality_certified=false`. The derived directory has only adapter weights/configuration and a derivation receipt; it is not an optimizer checkpoint. The +10 result was pending when these recommendations were recorded:

- **If +10 lowers NLL:** report the exact gain as an adaptively selected scaled-adapter training-fit result. Preserve the original optimizer checkpoint's regression. Do not present the derived weights as a completed Adam update, a general-purpose learning-rate recommendation, or held-out quality. The result supports that the learned direction can improve this one example at a different strength. Any later use for training must explicitly warm-start from those weights with fresh optimizer state; it must not claim exact resume from the unscaled checkpoint. Finish integration with the ordinary fit/held-out gates still intact.
- **If +10 does not lower NLL:** end this bounded diagnostic series without promoting any tested adapter over the base. Report that full-context training mechanics, checkpointing, and the tested component gradients worked, while this one-step training-fit attempt failed and useful full-context adaptation remains unverified. Finish the experimental tool integration with that limitation visible; do not weaken its gates or change engine arithmetic merely to obtain a positive receipt.

No additional GPU work or source changes are recommended during this bounded series unless a concrete defect appears. A future efficacy experiment should be a separately recorded training run and use multiple independent examples for quality assessment; this single training example cannot support generalization, functional kernel correctness, or speedup claims.


## Final bounded-series outcome — 2026-10-12

The adaptive +10 probe finished: NLL **0.5074643705753422**, delta
**+0.006206511364569067** versus base, in **566.3 seconds**. It also failed the
same-answer improvement check. Thus the original +1 optimizer step and all three
tested derived strengths (+0.1, -0.1, +10) are worse than the base on this row.
**The bounded series is complete; no tested adapter was promoted as an improvement.**
The earlier conditional recommendations above are preserved as their recorded
pre-result decision rule; the no-improvement branch applies.

The [final aggregate](ktune-strength-diagnostics.json) contains all five conditions,
raw scores, plan/checkpoint/derived-weight hashes, exact unchanged-A/scaled-B
checks and unchanged engine identity. The original optimizer checkpoint and
failed-fit receipt remain unchanged. The derived directories are inference-only
artifacts, not new Adam updates or resumable optimizer checkpoints.

The 80 GiB configuration's complete update, saved-state checks and the covered
component VJPs remain verified. The current suite passes 518 tests plus 27
subtests. These results do not establish improved full-context fitting,
correctness of the composed full-model gradient, useful generated kernels, or
held-out quality. No scalar-strength or learning-rate remedy is established by
this bounded experiment. The cause remains unresolved; a future efficacy study
requires a separately recorded experiment rather than relabeling this trial.
