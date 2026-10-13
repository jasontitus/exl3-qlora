# Metal tuning: completed fit and current verification boundary

The normal SFT flow completed a real DeepSeek V4.1 Flash EXL3 3.0-bpw fine-tune on **three distinct complete historical Metal edits**, four exposures per edit, and **all three improved after fresh adapter reload**. Full-answer token-weighted NLL fell **0.35555322 → 0.32627658 (8.23%)**. The equal-example mean fell **0.37180793 → 0.32654213 (12.17%)**. This is a training-fit result, not held-out kernel quality.

The [audited result](three-changes-standard-fit-result.json), [independent review](recovered-three-change-independent-review.json), and [normal-flow recipe](standard-three-change-run.md) bind exact source, tokens, masks, checkpoint, process and resource receipts. All three fresh uncached supervised-logit digests exactly match the final warm training-mode probe. Every probe row actually hit the frozen-prefix cache.

| Complete historical change | Serialized tokens | Supervised targets | Base NLL | Trained NLL |
| --- | ---: | ---: | ---: | ---: |
| SIMD ownership: two rows to one | 3,606 | 204 | 0.317424 | 0.241231 |
| Prefill tile 64 to 32 and matching dispatch | 5,809 | 811 | 0.501258 | 0.461151 |
| Two-row GDN recurrence and prefill selection | 5,822 | 1,877 | 0.296742 | 0.277244 |

There were **12 updates**, **11,568 supervised target presentations**, and **78,848 trainable parameters** in final block 39's `wq_a`/`wo_b` LoRA adapters. Seed 8, rank 4, alpha 8, Adam LR 1e-4, chunk 8192 and checkpoint activation arithmetic were fixed before launch. No example was truncated. This is three of the 24 eligible records in the earlier narrow import; repeated epochs, files and hunks are not additional independent examples.

## What improved in the answers

The [category totals](three-change-category-totals.json) show why this is only a fitting milestone. Across 1,527 added-code targets, NLL fell **0.06224455 → 0.05903664**, while teacher-forced top-1 correctness rose **1,501 → 1,502**. The one corrected position changed ` const` to the reference ` if` in the bounds check `if (dv_idx >= …)`. No added-code position changed from correct to incorrect. These predictions receive the correct preceding answer tokens; they are not an independently generated patch.

Added-code NLL slightly worsened on the first two examples and improved on the third. **92.1% of the total NLL-sum decrease came from comments and diff headers**, versus 5.8% from added code. EOS top-1 correctness regressed from 1/3 to 0/3 even though pooled EOS NLL fell. The audit preserves these regressions. No new patch has been applied, compiled, checked numerically or timed as a result of this adapter.

## Memory, time and repeatability

Training plus the final probe took **30.14 minutes** and peaked at **53.08 GiB MLX allocation**, under the declared 80 GiB process cap and 8 GiB reserve. The three frozen-prefix entries retained about 645 MiB inside a 2 GiB cache. Initial example updates took 280–579 seconds; repeated updates took 19–48 seconds. These observations combine this adapter scope and cache; they are not a controlled speed benchmark or a throughput estimate for hundreds of examples.

Fresh baseline scoring used an 80 GiB cap. The first adapter evaluator failed its startup check because 86.48 GiB was available and 88 GiB was required. A predeclared resource-only retry used **72 GiB with the same 8 GiB reserve**, identical arithmetic/data/cache settings and the same selected checkpoint. The completed baseline was retained with its original receipt. A separate mistyped-path invocation failed before model loading and is also preserved. The [recovery provenance](../artifacts/loss-fix/three-changes-standard-prefix/fit.json) includes both failures and the actual successful retry.

## Use the saved result

The [exported adapter](../artifacts/loss-fix/metal-tuning-adapter-step12/bundle.json) is byte-identical to selected step 12. It is an adapter for this EXL3 engine, not a PEFT or `mlx_lm` adapter. Exact resume retains the optimizer in the project; the export contains inference weights only.

```bash
cd ~/experiments/exl3-qlora/apple
./artifacts/loss-fix/metal-tune status

# Deliberate continuation: 16 total updates, not 16 additional updates.
./artifacts/loss-fix/metal-tune resume --steps 16 --no-compare
```

The convenience entry point pins runtime-v3 and the saved project. Later source improvements do not silently migrate that project. The selected adapter requires checkpoint activation mode and chunk 8192. Long-patch generation currently recomputes the full prefix per token; an efficient adapter-compatible generation path and execution harness remain necessary for a practical kernel-quality benchmark.

## Loss regression diagnosis and fixes

The original all-40-layer one-step run genuinely worsened loss. Saved Adam updates and the independent composed-gradient checks do not show a reversed or missing gradient. Actual final-block observations show strong BF16 addition absorption and changed expert routing; the surrogate-gradient prediction did not describe the first realized step. The separate BF16 activation experiment decreased its own objective, but changes the zero-adapter model.

The same-runtime all-40-layer control has now completed under the [prospective resource amendment](matched-checkpoint-control-resource-plan.json). One checkpoint-mode update increased NLL **0.501257837 → 0.503300196**; the BF16-mode treatment decreased its own NLL **0.506172359 → 0.501490604**. The [final audit](matched-checkpoint-control-76g-audit.json) and [independent review](matched-control-final-independent-review.md) confirm this regression-versus-descent contrast on the same previously inspected 811 targets. The control was planned after the treatment result was known. Training caps were 76 versus 80 GiB, with the same 8 GiB reserve; this is not an equal-resource timing comparison.

The control reproduced the historical baseline but not the exact historical update: 78 of 80 B matrices differ, and the old post-update NLL was 0.505372467. A [source review](checkpoint-reproducibility-source-review.md) found no intended checkpoint-mode formula change. Repeated-index GPU atomic accumulation is a possible source of nondeterminism, not an established explanation. The two activation modes also have different zero-adapter objectives. These results do not identify a unique root cause or certify BF16 mode as a general fix. The working three-example recipe above retains checkpoint activation arithmetic and combines selected adapter scope, a different learning rate and multiple updates; it does not isolate one universal remedy.

Adversarial review also found and repaired checkpoint-loader validation defects. The loader now validates a canonical fresh Adam schema, finite state, exact scalar counters, learning rate, RNG structure and arithmetic compatibility before mutation. **30 real MLX/workflow integration tests passed**, including 16 new failure-preservation and exact-resume cases; see [integration receipt](loader-real-mlx-integration.xml) and [independent CPU review](loader-hardening-independent-review.md). These repairs protect resume integrity; the inspected historical loss-regression checkpoint was valid.

## Broader, structured training data

The original three records provide complete parent files and historical patches, but do not contain independently verified profiling analyses or fresh executable-kernel tests. The broader intake is now implemented for pinned mlx-serve, TensorFold, Sushi and llama.cpp Git histories and archived performance evidence, with `prepare-upstream` and `finetune-upstream` commands. Its [adversarial design review](upstream-intake-adversarial-review.md) requires grounded rationale, explicit unknowns, complete source units/dependencies, cross-fork deduplication and verifier-produced evidence.

The current history inventory is a candidate pool, not a ready 100–200-example training set. Whole-source-file prompts can exceed a million tokens in monolithic files. Complete function/unit extraction and context-specific compilation are needed before claiming practical coverage. Historical speedup claims remain useful source evidence, but only an executed, bound compilation/correctness/timing receipt can establish a verified result. Newly generated kernels will need the same gates on the target hardware, with all attempted candidates retained in best-of-N evaluation.

The actual bounded 100-commit audit reconstructed 93 changes. Five complete-file
records fit the context ceiling; 88 were too long and seven lacked supported
source. These five include non-kernel changes and unresolved source-reported
correctness evidence, so they are not a curated positive kernel benchmark.
A reviewed llama.cpp convolution example is 3,394 complete tokens and both
isolated kernels compile. TensorFold's complete EXLR8 projection translation unit
compiles before/after under Metal 3.1 and 4.0. These compiler checks are separate
from source claims, and no new structured example has yet been trained. See the
[actual inventory and counts](upstream-corpus-inventory.md),
[upstream tool contract](../docs/UPSTREAM_INTAKE.md), and
[one-command recipe](../KERNEL_TUNING.md#structured-upstream-history-flow).

The actual `finetune-upstream --prepare-only` command also completed for the compact llama.cpp example: one full 3,394-token record, 1,521 supervised targets, no truncation, and complete bound provenance. The [preparation receipt](upstream-llama-real-preparation.json) records zero optimizer updates. Omitting `--prepare-only` invokes the standard trainer; the prepared project can also be resumed through the normal CLI.

## Executed source-kernel qualification

Two narrowly scoped GPU harnesses now pass on this M1 Ultra. Their tested kernels come from upstream history, not model generations.

| Source pair | Executed checks | What remains unverified |
| --- | --- | --- |
| TensorFold EXLR8 projection, `51426b8e` | All 65,536 codebook states and 162 projection cases: finite bitwise before/after parity, 131,328 projection output elements and unchanged tail canaries | Independent absolute projection oracle, prompt/routed/rotation paths, exceptional values, timing, full commit |
| llama.cpp transposed 1D convolution, `560445bf` | 16 fixed cases across FP32/FP16 weights: 29,498 outputs per variant match both the baseline and an independent exact dyadic CPU oracle; finite results, unchanged inputs and prefix/row-gap/tail canaries | General rounding-sensitive distributions, exceptional values, surrounding ggml runtime, timing, full commit |

The [TensorFold GPU receipt](../artifacts/loss-fix/tensorfold-exlr8-parity-build-v3/parity.json) and [independent check](tensorfold-projection-gpu-independent-check.json), and the [llama.cpp GPU receipt](../artifacts/loss-fix/llama-conv-parity-v2/gpu-parity.json) and [independent CPU reconstruction](llama-conv-gpu-independent-check.json), bind exact source, compiler inputs, ordered libraries, runner and declared cases. They do not establish the historical reported speedups. They also do not automatically promote imported records to `execution_verified`: performance measurement and a general receipt-admission path remain unfinished. The [qualification recipe](../KERNEL_TUNING.md#source-kernel-qualification) documents the scope and commands.

The integrated implementation passed **1,040 tests plus 27 subtests** ([JUnit](loss-fix-integrated-pytest.xml)). Independent reviews repaired source/include validation, artifact binding, zero-byte compiler outputs, checkpoint state validation and provenance handling. The remaining product gaps are a sufficiently large curated dataset, held-out generated-kernel evaluation, and a compiler/correctness/timing best-of-N loop. Falling answer loss alone does not fill those gaps.

The tested source is installed at `~/experiments/exl3-qlora/apple` and preserved in the runtime-v4 snapshot. The [installation receipt](structured-upstream-integration.json) checks all 69 source/script/lock snapshot files and both projects' integrity. The completed three-example project and existing Studio stay on runtime-v3; the newly prepared upstream project matches runtime-v4. No old project was silently migrated.
