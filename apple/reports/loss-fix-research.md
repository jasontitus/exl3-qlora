# Fitting failure: evidence, localization tests, and complete Metal edit data

Later-results pointer (2026-10-12): this body preserves its review cutoff. The
[completed one-row diagnostic and BF16 own-mode result](last-block-training-result-and-recipe.md)
and [active normal three-change recipe](standard-three-change-run.md) supersede
pending-result wording below. The launched normal run uses protocol v2,
runtime-v3, an explicit 8 GiB reserve and all 12 checkpoints retained. Its fresh
three-row fitting/cache-parity results are still pending.

2026-10-12. Read-only inspection of repository revision `b7668879`, existing
receipts and current primary sources. No model computation or engine edits were
performed for this report. Experiments below are proposals, not measured fixes.

The completed run trained **one historical edit example**, selected from 24
eligible examples. It contains two unified-diff hunks across two Swift files
(one embeds Metal), with 11 added and three deleted lines. It changes a prefill
matrix tile from 64 to 32 rows and the matching dispatch condition. Counting
files, hunks, generated variants, or repeated optimizer steps as additional
independent kernel optimizations would be misleading.

## What is established, and what remains a hypothesis

The original update completed in 1,862 seconds, peaked at 73.95 GiB of MLX
memory, and produced finite saved adapter/optimizer state. Its 811-token answer
NLL increased from 0.5012578592 to 0.5053724668. Derived inference-only strengths
+0.1, -0.1 and +10 also worsened NLL. The nonmonotonic series does **not** support
declaring a smaller learning rate or reversed update to be a fix. The recorded
optimizer reconstruction matches the saved first Adam update, and component
VJP checks pass within their declared coverage. These narrow checks do not
prove the entire model's surrogate gradient is a descent direction. See the
[immutable strength series](ktune-strength-diagnostics.json),
[optimizer audit](first-step-optimizer-diagnostic.json), and
[component coverage](first-step-diagnosis-update.md).

| Candidate explanation | Evidence now | Discriminating test |
| --- | --- | --- |
| Wrong optimizer sign or missing update | Checked first-step sign/state equations match; 80/80 B matrices changed from zero | Preserve the audit. Do not spend another full update merely checking the same sign. |
| BF16 adapter addition loses small changes | The local class inherits MLX-LM's forward, whose adapter result is cast to input dtype before addition. This is a real rounding boundary; its responsibility for this failure is **unmeasured**. | On identical captured actual inputs, measure nonzero adapter residuals that leave the sum unchanged; compare BF16 and FP32 accumulation with a declared return dtype. Recasting the sum immediately to BF16 does not remove the rounding boundary. |
| FP8/FP4 activation rounding and identity STE mismatch | `attention.quantize_activation` uses exact rounded forward values and an identity VJP. A full-model descent guarantee is absent. | With a fixed captured block input and cotangent/target, compare actual loss change and predicted gradient direction, then repeat with only the relevant quantizer disabled in a diagnostic path. Record changed quantization codes and scales. |
| Changed indexer/MoE decisions | Index selection is discrete and frozen for differentiation; later forward passes may choose different routes. No route-change measurement exists for the failed update. | Record route/index agreement at matching boundaries for base and updated adapters. A fixed-route diagnostic can isolate this effect, but cannot serve as deployment-quality evaluation. |
| Full backward/state handling defect outside checked components | Native projection, head CE, and fixed-selection sparse VJP coverage passed; full-model finite-difference agreement was not established | Use a real captured final block plus actual answer-head loss and adapters to test end-to-end parameter directions and state restoration, then compare the same case through the full model. |
| Inadequate data/target coverage | Only one edit and two attention adapter targets per layer were trained | Relevant to eventual usefulness, but it does not explain away failure to improve even the chosen training row. Localize fitting first. |

**Upstream arithmetic matters.** MLX-LM 0.32.0's LoRA implementation returns the
base result plus the scaled adapter residual cast to the input dtype. Our
FP32 A/B parameters already preserve optimizer updates; making them FP32 again
cannot remove forward BF16 rounding. This implementation match means the cast
is not independently evidence of a porting defect. [Pinned MLX-LM LoRA source](https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/tuner/lora.py)

QLoRA distinguishes low-bit **weight storage** from BF16 matrix computation.
It does not establish that this DeepSeek graph's extra FP4/FP8 activation
quantizers, sparse selection and mixed native/EXL3 paths share its optimization
behavior. The original paper's all-linear adapter result is specific to its
reported experiments. [QLoRA paper, sections 3–4 and appendix B](https://arxiv.org/pdf/2305.14314)

STE research calls the resulting direction a coarse gradient, rather than the
true derivative of the rounded objective. Its positive-alignment results use
restricted network/input assumptions and a suitable estimator; poor choices
can be unstable. That supports a targeted experiment here, not a diagnosis by
analogy. [Yin et al., 2019](https://arxiv.org/abs/1903.05662)

PyTorch's QAT reference explicitly simulates quantization in the forward pass
and uses a surrogate backward rule. Its published warmup disabled fake
quantization initially during a much longer training regime. That is evidence
for separating numerical modes experimentally, not evidence that removing
quantization only during our training will improve the unchanged inference
model. Any changed arithmetic must be recorded and tested in the same mode at
evaluation. [PyTorch QAT explanation and experiments](https://pytorch.org/blog/quantization-aware-training/)

Mixed-precision training's FP32 master-weight and loss-scaling techniques
address parameter-update/gradient information loss in half precision. They do
not demonstrate that BF16 adapter-sum rounding caused this run's failure.
Finite FP32 adapter gradients and matching optimizer state narrow that
hypothesis here. [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

## Bounded experiments worth running next

1. **Localize using the actual final block and answer positions.** Capture the
   real frozen prefix state once for the trained row, including hidden streams,
   pre-combination state and shared KV/index state. Bind it to source, model,
   tokenizer and input hashes. Test final-block adapter parameter directions
   against the same final norm/head/answer loss. Synthetic activations are
   useful for arithmetic tests but cannot replace this actual-data check.
   Reuse the captured state only while the whole preceding prefix remains
   frozen. This turns repeated localization into final-block work rather than
   repeated 40-layer passes.
2. **Change one boundary at a time.** Compare the existing mode, FP32 adapter
   accumulation, and the relevant activation-quantization intervention on that
   fixed local problem. Include zero-adapter output equivalence, finite/nonzero
   parameter gradients, changed-value counts and direction-versus-loss results.
   Keep the actual forward objective as the final gate; a smooth diagnostic
   objective improving is insufficient. No universal numeric tolerance is
   inferred from the papers: declare tolerances and measure repeated-base noise
   before interpreting tiny deltas.
3. **Use last-layer-only training as localization, with a new project.** It
   removes long chains of adapted layers from the causal question. MLX-LM
   defaults to the last 16 layers and supports selecting all layers, so a
   last-N interface is conventional; that default is not a quality guarantee.
   Our existing run targets `wq_a` and `wo_b` in all 40 blocks, not all linear
   modules. PEFT's QLoRA guidance targets all linear layers. Layer count and
   target-module count must therefore be recorded separately. A final-layer
   success supports an adapter fitting path; it does not fix or validate the
   earlier all-40 setting. [MLX-LM defaults](https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/lora.py),
   [last-N implementation](https://github.com/ml-explore/mlx-lm/blob/v0.32.0/mlx_lm/tuner/utils.py),
   [PEFT QLoRA target guidance](https://huggingface.co/docs/peft/en/developer_guides/lora#qlora-style-training)
4. **Confirm through the complete model before scaling data.** Save/reload the
   new candidate, recompute base and adapted answer NLL with the complete
   context, and retain per-row values. Require improvement beyond measured
   repeated-base variation; preserve failures and every tested condition.
   Then use the three-example probe below. Keep the 80 GiB process guard,
   measured reserve policy, packed-only cache and serialized microbatches.
   Do not pack examples into a 15K-token sequence. A claimed prefix-memory or
   backward-time saving needs measured evidence; selecting fewer layers alone
   does not document that the engine has dropped every unnecessary tape.

## A concrete three-change fitting probe

The [selection manifest](multichange-fit-selection.json) records exact IDs,
source hashes, full target hashes and lengths. All records come from
`full-parent-official-audit/train.jsonl`; no prompt or target is truncated.

| Submission suffix | Complete serialized tokens | Supervised tokens, including EOS | What it teaches |
| --- | ---: | ---: | --- |
| `c0e42c7e-268c-454f-a9cb-e705f952b819` | 3,606 | 204 | One SIMD ownership constant changes from two rows to one. Useful minimum fit diagnostic; mostly copied diff structure. |
| `1a621870-4993-4c43-9472-6f7d8f745140` | 5,809 | 811 | The already-trained prefill tile edit, with matching launch and specialization changes. |
| `8a690716-8068-4da8-9532-26047c96aa9c` | 5,822 | 1,877 | Adds a complete two-row GDN Metal recurrence and selects it for prefill. A more substantial code target than a constant change. |

One pass over these rows is 15,237 serialized and 2,892 supervised tokens. The
last row is 13 tokens longer than the largest completed training example; it
still needs a guarded measurement. The remaining 24-example set has median
17,806.5 and maximum 25,486 tokens. Its importer eligibility does not establish
80 GiB execution feasibility.

Predeclare the update/sample budget and score **all three** rows before and
after. Report every paired answer-NLL delta, the token-weighted mean, and the
unweighted mean across examples. A decrease in the weighted mean accompanied
by a regression on the 204-token row is a mixed result, not “three changes
learned.” Check exact same prompt/target IDs, answer masks and EOS handling.
Completion-only loss is a standard SFT objective, but loss reduction only
shows increased likelihood of the known answers. [TRL SFT objective](https://huggingface.co/docs/trl/sft_trainer#train-on-completion-only)

This selection intentionally contains **one conservative source lineage**.
There are no validation/test examples in its current import. Repeated
submission history shares code, benchmark workload and prior improvements;
randomly assigning these three rows to train/test would not create an
independent test. Complete parent files were selected using the historical
target, and prompts explicitly omit the surrounding repository, build
contract and performance workload. They are useful historical edit-imitation
data, not standalone optimization tasks. Full source binding improves
provenance but does not remove this selection bias.

Historical receipts cover one benchmark case and 64 checked steps, on an M5
target. Their timing ratios are against a fixed historical baseline, so the
reported cumulative speedup cannot be attributed to each isolated diff.
Training on an M1 Ultra does not make an M5-oriented optimization effective on
M1. For a future independent test, freeze hardware/workload/build contracts
before candidate selection, keep a separate unexposed task lineage, generate
from the prompt only, then apply/build, run numerical tests and perform paired
timings on the intended target. KernelBench's primary metric likewise requires
both functional correctness and a speedup threshold; syntax or NLL alone is
insufficient. [KernelBench paper](https://arxiv.org/abs/2502.10517),
[official evaluation methodology](https://github.com/ScalingIntelligence/KernelBench#-evaluation)

The current full-prefix generation path remains impractical for long patches.
This report therefore proposes a measured fitting diagnostic, not a claim
that the tool already demonstrates faster newly generated Metal kernels.
