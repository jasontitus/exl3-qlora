# Measured Metal-patch fitting result and next recipe

**Historical report.** The three-change run and same-runtime activation-mode control have since completed. See the [current consolidated results](metal-tuning-result.md) for fresh-reload metrics, the completed control, structured upstream intake and scoped GPU qualification. Statements below about active or pending work describe this report's earlier cutoff.

2026-10-12, updated after the eight-step diagnostic and the separate BF16 one-step experiment; the standard three-change run is active. The tool now has a **measured fitting improvement on the original complete Metal-patch example**: answer NLL fell from **0.5012578592 to 0.4735794807**, a **5.52% reduction**. This used the existing checkpoint activation arithmetic; the experimental BF16 intervention has not established a cause or a fix.

**One distinct kernel change was trained**, repeated for eight optimizer updates. It is the historical prefill tile change spanning two Swift files and two diff hunks, with 11 added and three removed lines. Eight updates do not mean eight different kernels. The standard run is now training on three selected changes from the same historical lineage. Its complete 12-step and fresh-scoring results are pending; the selected rows are not an independent test set. The archive has 24 eligible complete-parent examples, not 24 examples already trained or proved to fit in 80 GiB.

Sources: [original diagnostic](../artifacts/loss-fix/last-block-original-row/diagnostic.json), [token summary](../artifacts/loss-fix/last-block-original-row-fit-summary.json), and [independent CPU audit](last-block-eight-step-audit.json).

## What the completed run demonstrates

The complete example contains 5,809 serialized tokens, including 811 supervised answer tokens. Only final block39's `wq_a` and `wo_b` adapters trained: four FP32 tensors, **78,848 parameters**, rank4/alpha8. Blocks0–38 were frozen and their actual full-context states were captured once. Attention and expert routing in the final block continued to run normally after every update.

The prefix capture matched the full model's zero-adapter loss and supervised logits exactly. After update8, running the entire backbone again without the prefix cache reproduced the final loss, perplexity and supervised-logit SHA exactly. This final check was a **same-process uncached forward**, not a fresh-process adapter reload. All eight saved checkpoints passed checksum, finite-adapter, finite-optimizer and step/cursor audits; fresh-process scoring remains a gate for the standard three-change run.

| Step | Full-answer NLL |
|---:|---:|
| Zero-adapter baseline | 0.5012578592 |
| 1 | 0.5016792692 |
| 2 | 0.5012030379 |
| 3 | 0.4988910968 |
| 4 | 0.4959397518 |
| 5 | 0.4959021619 |
| 6 | 0.4890443195 |
| 7 | 0.4816420118 |
| 8 | 0.4735794807 |

The first step still worsened the objective. Later steps improved it under the predeclared eight-step budget. This establishes useful final-block fitting on this row. Compared with the original failed all40-layer trial, both adapter scope and learning rate changed (1e-5 to 1e-4), and eight updates replaced one. It does not isolate why the original all-layer first update failed. The separate BF16-mode result below also measures training descent without a matched current-source control.

The complete capture, eight updates with per-step scoring, and final full forward took **26.1 minutes**, with **52.86 GiB peak MLX allocation** under an 80 GiB process guard and explicit 12 GiB system reserve. After capture, updates plus their suffix score took roughly 57–63 seconds each. The saved prefix payload is 257,846,706 bytes. These are measurements of this one context and configuration, not a throughput guarantee or 32,768-token capacity claim.

## What improved, and what remains unproved

All metrics below use teacher forcing: the correct preceding answer tokens are supplied. They do not measure autonomous patch generation.

| Target category | Tokens | Baseline NLL | Trained NLL | Correct top1 tokens, before → after |
|---|---:|---:|---:|---:|
| Complete answer | 811 | 0.501258 | 0.473579 | 714 → 715 |
| Added code | 114 | 0.061606 | 0.059625 | 112 → 112 |
| Added comments | 175 | 1.545566 | 1.446374 | 111 → 112 |
| Diff headers | 208 | 0.593122 | 0.568587 | 180 → 180 |
| Removed code | 105 | 0.000593 | 0.000582 | 105 → 105 |
| Unchanged context | 208 | 0.021527 | 0.022974 | 205 → 205 |
| EOS | 1 | 1.113771 | 1.054298 | 1 → 1 |

About **77.3% of the total NLL decrease came from comments**, **22.7% from headers**, and **1.0% from added code**; unchanged-context regression offsets part of those gains. No added-code token switched from wrong to correct or from correct to wrong. The categories are coarse text labels, not semantic measures of optimization. The model already assigned high probability to much of the code before training.

No newly generated patch has yet demonstrated successful application, compilation, correctness or speedup. The historical optimization target is M5/NAX; the M1 Ultra is the training host. A useful kernel model still needs efficient adapter-compatible generation and the matching target-hardware execution harness. Current generation recomputes the full prefix per token, making a several-hundred-token patch impractical at this context. No claim of improved unseen-task quality follows from this fit.

## Observed first-step arithmetic — evidence, not an isolated cause

The [completed arithmetic observation](../artifacts/loss-fix/rounding-observation-step1-70g/arithmetic.json) replayed the original zero-B and first-update adapters through final block39 using its verified frozen prefix. Original layer calls supplied the returned arrays; adding observers preserved both scores and supervised-logit digests exactly. Both conditions also matched the saved experiment scores. The [independent CPU audit](rounding-observation-audit.json) checked the bound source/checkpoint receipts and independently recomputed the raw router comparison and gradient/update dot products.

For the first update, **89.19% of nonzero `wq_a` residual elements and 95.99% of nonzero `wo_b` residual elements produced no output-bit change when added to their local BF16 base output**. None of these residual elements rounded to zero during the residual cast itself; the measured effect is absorption in the addition. FP32 residual/base L2 ratios were 0.0004285 and 0.00009474 respectively. These are local element counts over all 5,808 input positions, not percentages of answer tokens or useful learning signal. The element arrays were not retained, so the audit checks their reported arithmetic consistency rather than independently reconstructing every count.

Normal MoE routing changed the selected expert set at **152/5,808 positions (2.62%)**: 126 prompt positions and 26 of the 811 supervised answer-prediction positions. Ordered expert slots changed at 389 positions, which includes reorderings of an unchanged set. The observer never froze or replaced routing. In particular, `wo_b`'s local base branch also changes after upstream `wq_a`; its absorption count does not isolate that upstream effect.

The saved initial surrogate gradient dotted with the actual first parameter update predicts an NLL change of **−0.000117741**, while the measured change is **+0.000421410**. The 0.000539151 discrepancy confirms that this first-order prediction did not describe the realized first step. It does not identify which combination of nonlinear response, discrete route changes, activation quantization or BF16 addition caused the discrepancy. This final-block observation cannot explain the original all40-adapter failure by itself, and later checkpoint-mode updates already improved the same row.

The separate all40-layer BF16 activation intervention **completed one optimizer update and fresh post-update scoring**. Its own complete uncached pre-update objective was **0.5061723590** and its post-update NLL was **0.5014906038**, a decrease of **0.0046817552 (0.925%)** on the same 811 targets. The [audit](bf16-all-layers-one-step-audit.json) verifies the complete IDs/mask, finite/checksummed state, matching original A initialization and 80 changed B matrices. Its baseline is the pre-update training objective; it is not a separate fresh-process zero-B evaluation.

This is measured descent in the experimental mode. It is **not yet a matched causal explanation of the historical checkpoint-mode failure**: the old control has not been replayed under the new source. The two modes have different zero-adapter objectives, so comparing their final absolute losses directly would mix mode effects with learning. BF16 mode still retains packed weights, ordinary BF16 casts, FrozenIndexer quantizers and normal routing; it is not smooth or quantization-free. The [protocol](activation-causal-plan-v2.json) and [launch receipt](../artifacts/loss-fix/bf16-all-layers-launch.json) preserve the distinction. The explicit reserve changed from 12 to 8 GiB after startup headroom measurement, while the process ceiling stayed 80 GiB.

This distinction follows established practice: likelihood depends on conditioning and tokenization, so comparisons must preserve exact targets/context and use each changed model's own baseline ([Transformers perplexity documentation](https://huggingface.co/docs/transformers/v4.43.0/en/perplexity)). Straight-through gradients are surrogates; the relevant theory does not guarantee that every step descends in this graph ([Yin et al.,2019](https://arxiv.org/abs/1903.05662)). These sources motivate the checks; they do not establish this experiment's cause.

## Reproduce the measured diagnostic

Use the preserved runtime and prefix cache. This starts again from the cached zero-B initialization; it is not continuation from step8. Every output directory must be new. Run serially after other GPU work ends.

```bash
cd /Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple
export EXL3_PYTHON="$HOME/experiments/exl3-qlora/apple/.venv/bin/python"
export EXL3_MODEL="$HOME/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"

PYTHONPATH="$PWD/artifacts/loss-fix/runtime-v1/src" "$EXL3_PYTHON" \
  artifacts/loss-fix/runtime-v1/scripts/diagnose_frozen_prefix.py \
  --model "$EXL3_MODEL" --data artifacts/loss-fix/three-metal-changes.jsonl \
  --rows 1 --prefix-cache artifacts/loss-fix/last-block-original-row/prefix-cache \
  --output artifacts/loss-fix/last-block-replay-new \
  --steps 8 --learning-rate 0.0001 --rank 4 --alpha 8 --seed 8 \
  --chunk-size 8192 --memory-gib 80 --reserve-gib 12 \
  --packed-cache-mib 8192 --decoded-cache-mib 0 --allocator-cache-mib 256 \
  --verify-uncached-final --include-token-metrics
```

The cache binds source/runtime, model metadata and payload stat fingerprints, token data and initial adapter checksums. Keep its original runtime; do not point this command at current source or a different activation mode. The 12 GiB reserve is the explicit setting of this measured run, not a replacement for the development profile's 32 GiB default. Startup headroom checks still apply.

## Standard three-change run — launched, final results pending

[Protocol v2](three-change-standard-protocol-v2.json) records the changed sampler and initialization before this run, plus the explicit 8 GiB reserve. The active project uses the immutable [runtime-v3 snapshot](../artifacts/loss-fix/runtime-v3/snapshot.json). See the [normal-flow recipe and pending result table](standard-three-change-run.md). The three full examples contain **3,606 / 5,809 / 5,822 tokens**, with **204 / 811 / 1,877 answer targets**. Twelve standard updates mean four exposures to each change, in the fixed seed8 order `[2,1,0,0,2,1,1,0,2,0,2,1]`, totaling 11,568 supervised token presentations. This is a fresh selected-block39 initialization, not continuation of the successful diagnostic or an exact reproduction of its imported A matrices.

The normal importer consumes the raw records, then writes the prepared token rows. A [CPU audit](three-standard-token-audit.json) using the exact tokenizer/dataset/helper bodies verified that those rows retain the original order, complete IDs and loss masks and produce the same train-file SHA. Normal `--prepare-only` repeated that check successfully in the final frozen runtime before this training launch; the project has exactly three training rows, no valid/test rows and zero truncation. The CPU audit bypassed only MLX-LM package import-time Metal initialization; it did not load a model.

```bash
export EXL3_RUNTIME="$PWD/artifacts/loss-fix/runtime-v3"
export PYTHONPATH="$EXL3_RUNTIME/src"
"$EXL3_PYTHON" -m exl3_mlx.cli finetune \
  --model "$EXL3_MODEL" --data artifacts/loss-fix/three-metal-changes-raw.jsonl \
  --output artifacts/loss-fix/three-changes-standard-prefix --demo \
  --steps 12 --max-length 5822 --learning-rate 0.0001 \
  --gradient-accumulation-steps 1 --rank 4 --lora-alpha 8 \
  --adapter-layers 39 --activation-mode checkpoint --chunk-size 8192 \
  --profile development --memory-gib 80 --reserve-gib 8 \
  --packed-cache-mib 8192 --decoded-cache-mib 0 --prefix-cache-mib 2048 \
  --allocator-cache-mib 256 --save-steps 1 --save-total-limit 12 \
  --comparison-examples 3 --final-fit-examples 3 --prepare-only --no-compare

"$EXL3_PYTHON" - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('artifacts/loss-fix')
project = root / 'three-changes-standard-prefix'
expected = root / 'three-metal-changes.jsonl'
prepared = project / 'data/train.jsonl'
assert hashlib.sha256(prepared.read_bytes()).hexdigest() == hashlib.sha256(expected.read_bytes()).hexdigest()
manifest = json.loads((project / 'project.json').read_text())
assert manifest['data']['counts'] == {'train': 3, 'valid': 0, 'test': 0}
assert manifest['config']['seed'] == 8
assert manifest['config']['adapter_layers'] == [39]
assert manifest['config']['activation_mode'] == 'checkpoint'
assert manifest['config']['final_fit_examples'] == 3
print('Exact full tokens/masks and fixed experiment settings verified.')
PY

"$EXL3_PYTHON" -m exl3_mlx.cli resume \
  artifacts/loss-fix/three-changes-standard-prefix --steps 12 --no-compare

# Fresh-process base and adapter training-fit scores on all three rows.
"$EXL3_PYTHON" -m exl3_mlx.cli ktune verify \
  artifacts/loss-fix/three-changes-standard-prefix
```

The standard SFT seed defaults to 8 and is checked explicitly before training. `--demo` is required because all three records share one lineage; train data is never copied into a validation set. The 2 GiB in-memory prefix cache is optional, bounded inside the process limit, and does not change which blocks train. Its full real-model standard-run usefulness remains to be measured here. The shown project already exists and is active: these commands record its configuration, not a request to start another concurrent run. For a new experiment choose a new output directory. Keep runtime-v3 for preparation, continuation and verification; the 8 GiB reserve is this launch’s explicit headroom setting, not the profile default.

Acceptance requires fresh base and adapter scores for **every** original row, full-mask token-weighted and macro means, all per-row deltas, token-category diagnostics, finite saved updates and matching input/provenance receipts. Preserve and disclose any regressing row. The paired project fit command now requests per-token metrics for both fresh child evaluations, supplying aggregate and per-row evidence in one pass per condition. Apply the fixed token-category labels to those complete per-token arrays. Aggregate improvement alone must not be labeled all-case success. Held-out kernel quality still requires independent tasks and actual generated-patch execution on the target hardware.

The final probe is enabled explicitly with `--final-fit-examples 3`. It scores the same three complete rows in the still-warm training model after saving step12, preserving training mode and performing no optimizer update. The subsequent two fit children explicitly disable the prefix cache. A matching score alone is insufficient to claim warm-cache verification: all three final-probe forwards must have registered cache hits, with no new misses, and the bound row targets/logit digests and per-token scores must match the fresh adapter evaluation. Missing or partial evidence stays labeled incomplete.

After those receipts exist, the standalone CPU audit produces the per-row and code-category report without loading the model:

```bash
"$EXL3_PYTHON" scripts/summarize_standard_fit.py \
  --project artifacts/loss-fix/three-changes-standard-prefix \
  --protocol reports/three-change-standard-protocol-v2.json \
  --categories artifacts/loss-fix/token-categories-0.json \
               artifacts/loss-fix/token-categories-1.json \
               artifacts/loss-fix/token-categories-2.json \
  --runtime "$EXL3_RUNTIME" --decode-context \
  --output reports/three-change-standard-fit.json
```

Use the exact frozen runtime root in `--runtime`; the command rejects source or checkpoint mismatches. The report audits every retained checkpoint and reconstructs the deterministic sampler from its verified cursor and seed, also checking each step's supervised target count. This is evidence of four exposures to each of three historical changes, not three independent optimization tasks.
