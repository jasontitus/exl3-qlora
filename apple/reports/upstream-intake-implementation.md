# Upstream intake implementation and observed bounds

Implemented in current staging only; immutable runtime-v2/v3, active training,
matched-control plans and existing datasets were not edited. No GPU job ran.

The new `ktune prepare-upstream` command imports offline pinned Git commits into
structured historical reconstruction examples. Full parent/head blobs, complete
raw diff, all evidence and exclusions remain bound sidecars. Answers contain
actual changed source (or explicitly reviewed units), without separately rendered
Git hashes or duplicated patches. Unchanged declared dependencies are prompt-only.

`reviewed_units` requires exact UTF-8 byte ranges/hashes, a named review and declared
dependency scope. Nonoverlapping replacements plus unchanged-gap/tail comparison
must reproduce every full changed file. This proves coverage and byte identity;
it does not certify semantic completeness or executable dependency closure.
Opt-in `historical_source` task fields are explicitly retrospective/target-derived.

The narrow CPU tokenizer uses the same masked chat-template and prefix contract as
MLX-LM CompletionsDataset, without the MLX package initializer. CPU tests cover
independent installed-class serialization, EOS/special-token validation, source
mutation, malformed evidence, protection propagation, duplicate and negative
aliases, byte ranges/UTF-8/new-deleted source, whole-edit replay, metadata-stripped
standard SFT preparation, and compatible provenance-envelope hashes. The initial
combined suite passed 69; adding the independent compiler CLI dispatch brings the
current set to 70 (19 final CLI checks passed separately). Ruff is clean.

Real whole-file audit: eight named upstream commits all reconstructed, eight unique
normalized patches, only two complete structured examples fit 32768 tokens (28115
and 23493); six remain excluded intact. The exploratory export predates only the
final report-envelope/EOS hardening. Final reviewed llama convolution import fits
3394 tokens (1873 prompt, 1521 target), preserves complete kernel/type/host context,
and reconstructs all changed bytes. It is not compile- or execution-verified.

The strict `execution_verified` admission gate currently admits zero. Actual
compiler, numerical, workload-specific paired timing and independently confirmed
best-of-N selection require separately bound verifier receipts. Source-written
success flags cannot promote records. The separately implemented `compile-metal`
command dispatches to the parent's narrow compiler gate and has no effect on this
importer's execution status. The larger 100-candidate inventory is a collection
attempt, not a claim of 100 accepted or quality-verified training examples.

See `docs/UPSTREAM_INTAKE.md` for the manifest/API contract and
`reports/upstream-intake-author-receipt.json` for source hashes and exact artifacts.

## Subsequent standard-flow integration

Added `ktune finetune-upstream` after parent request. It shares the existing
standard SFT flags/lifecycle, requires `prepare(provenance_required=True)` and
successful evidence binding before updates, preserves `--prepare-only`, and only
performs requested fresh fit checks after completed training. The final CLI suite
passes24 CPU tests, including stopped/prepared/completed flows and binding failures;
the current importer+CLI total is75 unique tests. No model training was launched
while implementing this command. The core importer was unchanged during the
parallel100-candidate audit.
