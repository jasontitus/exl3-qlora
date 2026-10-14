# Audit paired evaluation results

Use the CPU-only `exl3-mlx-audit` command to check that a base-model evaluation
and an adapter evaluation scored exactly the same complete examples and loss
masks, and that their per-token metrics agree with their reported totals.
It accepts any nonempty number of examples; no GPU or model is loaded.

This is an advanced companion to the normal `exl3-mlx compare PROJECT` flow.
It does not run evaluations, create category labels, authenticate a model launch,
or measure generated-code quality.

```sh
uv run exl3-mlx-audit \
  --prepared heldout.prepared.jsonl \
  --categories heldout.categories.json \
  --base base-token-metrics.json \
  --adapter adapter-token-metrics.json \
  --output paired-audit.json
```

The output path must be new and its parent directory must exist. Invalid or
inconsistent input fails without publishing a successful audit. Input hashes
bind the bytes examined; changing inputs during the audit is rejected.

## Required inputs

- Prepared JSONL: ordered rows containing `input_ids` and `loss_start`, as used by
  the evaluation. Preserve the full rows because the selected-example hash covers
  them, not only their supervised suffixes.
- Categories JSON: `tokenizer_size`, `prediction_vocab_size`, a `rows` list, and
  aggregate `counts` by label. Each category row supplies `index`, `loss_start`,
  `input_ids_sha256`, `supervised_tokens`, and one `categories` label for every
  supervised token. Labels have three slash-separated components, such as
  `metal/added/code_or_other`. Labels must come from a separately reviewed
  classification procedure; this command checks consistency, not meaning.
- Base and adapter JSON: workflow evaluation receipts produced with
  `--token-metrics`, including `evaluation_context.selected_examples_sha256`,
  sequence capacity, example/token totals, and per-row `target_ids`,
  `predicted_ids`, `token_nll`, `correct_tokens`, and `loss`.

Hashes use SHA-256. The full selected-example digest uses sorted-key compact
JSON; per-row ID digests use compact JSON arrays. Match the model's actual output
vocabulary and tokenizer size rather than inferring them from observed targets.

## Interpret the result

`aggregate.all` reports token-weighted NLL and corrected/regressed teacher-forced
token counts. Category and family aggregates help distinguish added-code gains
from comment or metadata gains. `macro_example_nll` separately averages per-item
NLL so a few long examples do not hide the behavior of shorter examples. Lower
adapter NLL is better prediction of these supplied answers, not proof of better
new code, fewer runtime errors, or faster kernels.

A successful result remains `metrics_consistent_pending_provenance_audit` with
`quality_certified: false`. Verify the exact base, runtime, adapter/checkpoint,
completed processes and dataset split separately. Then assess generated patches
with compilation, numerical-correctness and performance tests on held-out tasks.

## Normal project comparisons

`exl3-mlx compare PROJECT` now automatically audits the paired token receipts
before publishing `comparison.json` and `report.html`. This also applies to the
comparison following standard training. The audit uses exactly the selected
prepared rows (the saved `comparison_examples` limit), without a fixed example
count. It saves the selected rows, category snapshot, and `token-audit.json` in
the comparison attempt and records their hashes.

Default categories describe observed token IDs: supervised tokens and the local
tokenizer's declared EOS IDs. They do not infer code, reasoning, or correctness
from text. Generation stop IDs are not automatically treated as EOS. The report
contains token-weighted NLL, separately named macro example NLL, teacher-forced
correct predictions, corrected tokens, and regressed tokens. The comparison
status uses token-weighted NLL. These are not generated-patch success, numerical
correctness, inference speed, or quality certification.

For externally reviewed categories, use
`exl3-mlx compare PROJECT --categories categories.json`, or
`trainer.evaluate(categories="categories.json")`. The category schema is the
same as the standalone auditor. Rows must match the exact selected sequences,
order, masks and target counts; vocabulary dimensions must match the project's
bound local model configuration and tokenizer. The source is captured once,
preserved in the attempt and checked for changes before publication. Supplied
category labels remain user assertions, not semantic certification.

Existing project provenance checks still run. A malformed receipt, changed
input, or category mismatch fails without replacing a previous comparison.
Failed attempts remain available for diagnosis. The audit checks receipt
consistency; it does not independently attest process or checkpoint provenance.
