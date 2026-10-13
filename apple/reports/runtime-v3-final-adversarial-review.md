# Runtime-v3 adversarial review — 2026-10-12

No blocking defect was found in the active standard three-change training and
fresh verification path. The installed `apple/src/exl3_mlx` files and frozen
runtime-v3 both match all 39 source-file SHA256 entries in the active project's
implementation manifest; their dependency-lock hashes match too. The project's
three prepared rows have SHA256
`ca6fd8c939be9de5f7715e79d5f1ffc9354d251d1092aa9a3cc77d5d5ce80cb3`,
sequence length5821 (5822 serialized tokens), and the planned block39/checkpoint
activation/rank4/alpha8/seed8/12-step settings. The active process was not modified
or interrupted. This review used source reads and bounded CPU fixtures only.

The full-suite receipt `loss-fix-v3-pytest.xml` records zero failures/errors:
800 XML test entries, corresponding to the parent's 773 tests plus27 subtests.
This review does not claim to have rerun that GPU suite. Some cache and evaluator
code was authored by this reviewer; separate process/quality agents reviewed
those changes, so this report is not an independent authorship claim for them.

## Nonblocking findings and follow-ups

### P2 — Standalone adapter load does not enforce recorded numerical semantics

`src/exl3_mlx/train.py:107` writes `engine_semantics` into adapter_config.json,
but `load_adapter` at lines130–148 only compares base metadata and the model's
rank/alpha/layer/activation-mode configuration before validating tensor values.
An adapter from different numerical engine semantics can therefore be loaded by
the standalone workflow even if its recorded semantics are explicitly different.

Confirmed with the exact installed `load_adapter` function body extracted into
a CPU fixture: replacing the verified-config/tensor reads with finite CPU arrays
and setting `engine_semantics="known-incompatible-older-arithmetic"` still reached
model.load_weights. This tests control flow, not Metal numerics or safetensors
parsing. Existing checksum validation does not close this semantic-version gap.

This does not affect the active fresh run: its standard project rejects changed
source/runtime identity (`sft.py:319`) and validates the checkpoint contract's
current semantics (`sft.py:877`). Recommendation after the pinned experiment:
reject an explicit mismatch and require an explicit documented migration for
legacy standalone configurations lacking the field. Preserve the current frozen
runtime for old exact continuation; do not silently relabel an old adapter.

### P2 — Resume validates optimizer structure, not moment values, before mutation

`src/exl3_mlx/training.py:299–318` initializes the destination optimizer, checks
saved shapes/dtypes/RNG structure and step equality, then loads adapter tensors
and assigns the saved optimizer state. It does not reject NaN/Inf moments,
negative second moments, or a nonfinite saved learning rate first.

Confirmed by executing the unmodified installed `load_checkpoint` function body
with CPU stubs for MLX and file payload loading. A receipt with matching hashes,
shape/dtype-correct arrays, a NaN first moment and negative second moment was
accepted, and adapter loading was reached. This is a malformed-but-checksummed
state case; it is not evidence that the active optimizer produced such values.
Checksums establish payload integrity, not numerical validity.

The fresh active run does not use resume. Its final standalone audit separately
checks finite saved tensors/moments, and optimizer_step refuses nonfinite updated
adapters before saving a checkpoint. Recommendation: validate all numeric state,
nonnegative second moments, legal scalar step/cursor and learning-rate contract
before committing any destination model/optimizer/RNG mutation. Add a rejection
test that preserves all destination values. This was already a defensive gap,
not introduced by the activation or prefix-cache changes.

### P2 — Standalone summary trusts protocol case labels without per-row binding

At review time, `scripts/summarize_standard_fit.py:464` required three unique
protocol case IDs, and lines577–578 attached them to prepared rows, but did not
compare them against the hash-bound retained training provenance. A CPU synthetic
native-project fixture accepted protocol IDs `wrong-one`, `wrong-two`, and
`wrong-three` with all prepared data, scores and checkpoint receipts unchanged.
The reported row labels then used those names.

This does not change numerical fit scores. The actual project's retained
`data/provenance.jsonl` entries were checked manually: the correct cases
c0e42c7e/1a621870/8a690716 occur in the intended order, and dataset.json's raw
source hash agrees with the protocol's raw_sha256. The issue was sent to the
quality-review agent for a standalone audit-script fix and swapped/foreign-case
regression tests. It requires no engine change or active-run restart. Until
fixed, do not describe the script alone as independently verifying case labels.

## Supported-path checks

* Frozen-prefix caching stops strictly before the earliest attention or expert
  adapter (`deepseek.py:252–266`). A zero B matrix is still trainable, and the
  all40-layer mode disables this cache. The suffix's attention/MoE routes execute
  normally on every call.
* The exact full token batch/shape, training/eval flag and recomputation flag are
  part of each cache key. Cached h, pre-mix and all four attention-state leaves
  are materialized privately, excluded from model/optimizer/checkpoint parameter
  traversal, and admitted only within the LRU byte bound. Oversized entries
  bypass without evicting useful entries.
* The corrected storage signature survives MLX's temporary frozen-array wrappers
  during gradient tracing. It retains original view owners to prevent allocation
  reuse at the same address (`prefix_cache.py:20–57`). Source-file stat checks and
  prefix trainability/structure checks run before hits and insertion. The API
  explicitly prohibits in-place frozen storage/private-behavior/cached-array
  mutation; these guards are not a content hash of all model weights.
* Activation mode is explicit per module and serialized, never scoped by a
  temporary global. Checkpoint mode retains its previous quantizer path. The
  experimental bf16 mode bypasses only differentiable-path explicit FP8/FP4
  activation roundtrips; all FrozenIndexer native-input and FP4E8M0 quantizers,
  packed weights, ordinary dtype casts and dynamic routing remain. Loading or
  resuming a different activation mode is rejected before parameter mutation.
* The final probe runs after the final immutable checkpoint is saved, performs
  scoring only, and preserves training mode to reuse actual warm entries
  (`workflow.py:541`; `training.py:148`). Both evaluator paths restore the entry
  flag on scoring/check failures. The probe records original per-row targets,
  predictions, NLL and supervised-logit hashes, selected-data/source/checkpoint
  identity and before/after cache counters.
* Standard fresh fit evaluation explicitly sets prefix_cache_mib=0, starts
  separate base and adapter processes, and binds command/data/source/selection
  receipts. Comparison requires per-row logit SHA equality and matching targets,
  predictions and token loss within1e-6. Warm-cache certification additionally
  requires one hit per probe row with no new misses (`fit_probe.py:127`). A probe
  belonging to a different selected checkpoint is unavailable, not a pass.
* Adapter files are checked for expected keys/shapes/dtypes and finite values
  before model.load_weights. Project export retains the activation mode and
  recalculates the adapter checksum. Exact project resume binds dataset,
  arithmetic settings, runtime and source; a resource-only prefix-cache budget
  does not alter that numerical contract.

## Interpretation and remaining gates

The one-row all40-layer experimental bf16 run has independently audited own-mode
baseline0.5061723589897156 and fresh-process post-update loss0.5014906038202744,
a decrease0.0046817551694411375. All80 B matrices changed from zero; the saved A
and zero first-step A moments matched the expected initialization. The audit
correctly sets matched_causal_claim=false: an older checkpoint-mode control is
historical until replayed under matched current source. This demonstrates one
successful same-mode training update, not a certified cause of the original
checkpoint-mode regression, a universal fix, or full-model native parity.

The earlier final-block eight-step experiment's final uncached check was a
same-process full-backbone forward. It must not be presented as a fresh-process
reload. The active standard project addresses that distinction with separate
fresh base/adapter processes and a bound final warm-cache probe. Its outcome is
pending; no training-success or real reusable-cache-parity claim is made here.

Report token-weighted aggregate, equal-case macro, every per-case NLL and the
predeclared token categories. A gain dominated by comments/header tokens does
not establish better code. Teacher-forced predictions use the correct previous
answer tokens, unlike generation. Three changes from one connected historical
lineage, with target-selected parent files, are training-fit evidence only.
No generated edit has thereby demonstrated application, compilation, numerical
correctness, faster kernels, or generalization. The standard summarizer's
`tool_verification_complete` should be read narrowly as verification of this
fixed optimizer/cache/reload path, not every supported tool/model behavior.
