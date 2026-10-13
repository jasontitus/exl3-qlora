# Selected adapter scope and frozen-prefix diagnostic review

2026-10-12. CPU/read-only review; no model load or GPU computation. The reviewer
implemented the standard scope facade, so that portion is a self-review with
independent input from the engine reviewer, not a claim of independent authorship.

## Standard SFT layer selection

The API resolves `adapter_layers=None` to all 40 blocks. Explicit selections
must be nonempty unique integer IDs in 0..39 and are copied/sorted into the
saved configuration. Both standard and ktune `finetune` expose the selection.
Only `wq_a` and `wo_b` receive adapters; expert adaptation is not implied.

Training forwards the exact selection to the backend. Project identity compares
it as an immutable training setting, checkpoint discovery compares it against
the saved adapter configuration, and export preserves that configuration.
Resume accepts resource changes but rejects layer overrides. Evaluation and
generation reconstruct the adapted model from the verified checkpoint's
configuration. The zero-adapter baseline continues to use the ordinary base
path; this is not a new baseline arithmetic mode.

Compatibility is deliberately narrow: old standard-project configurations with
an omitted layer setting, and explicit JSON `null`, mean all 40. This does not
bypass the separate implementation/runtime check. Historical checkpoints from
this engine serialize a concrete layer list through `model.adapter_config`;
a malformed or foreign checkpoint with a different list is not silently
migrated. The implementation added a regression for source-identity rejection
even when an older all-layer configuration is recognized.

Independent review caught `Number('')` turning a trailing comma in Studio into
block 0. Empty split fields are now removed before conversion. `39,` selects
only 39; `,38 39,` selects 38 and 39; comma-only input yields an empty selection
that server validation rejects. Pending new-project edits survive polling, the
saved scope remains visible, and the resume action does not submit the editable
selection. An explicit API attempt to override it on resume is rejected.

The focused CPU suite passed **115 tests**, including invalid scope inputs,
default semantics, CLI precedence, checkpoint mismatch rejection, resume,
export, Studio API behavior and the actual browser JavaScript under a Node DOM
mock. [JUnit receipt](adapter-layers-cpu-tests.xml). Ruff passed for the changed
Python files. This suite tests interfaces and integrity, not fitting efficacy.

## Frozen-prefix diagnostic

The inspected script is `scripts/diagnose_frozen_prefix.py`. Capturing the
complete hidden streams, pre-combination state, shared KV/index state and empty
Engram input immediately before block 39 is mathematically appropriate when
blocks 0..38 have no trainable adapters. Block 39 attention and MoE routing then
execute on every replay; the script does not freeze that block's routing. Its
architecture preflight rejects a final block with an indexer, compressor source
or Engram, preventing an unsupported shortcut through those features.

Positive controls in the inspected implementation include zero-B initialization,
exactly four trainable block-39 adapter tensors, full answer masks, source/model/
data binding, checksummed captured arrays, baseline uncached/cached logit and NLL
equality, finite gradients/updated parameters, saved Adam state and explicit
additional-step semantics. The optional final uncached comparison is needed
before claiming the resulting adapter also improves the complete model path.

Two receipt concerns were sent to the script owner before capture:

- The diagnostic script itself must be part of cache/resume implementation
  identity. Recording its hash only in the report does not prevent a changed
  script from accepting old cached states. Bind its hash or an equally strict
  implementation contract before capture, then freeze it.
- Row indices must be mapped to the intended case IDs before running. `--rows 1`
  is not a portable way to identify the earlier trained edit. Bind the prepared
  dataset hash and a case-ID-to-row mapping from provenance; label every score
  with that mapping. The [three-example selection](multichange-fit-selection.json)
  contains one conservative source lineage and is training-fit-only.

No additional concrete mathematical blocker was found in this static pass.
It did not establish numerical parity itself. A successful last-block probe
would show a narrower fitting path; it would not identify BF16 rounding as the
root cause, prove all-40-layer gradient correctness, or establish newly generated
kernel correctness or speed. Comparing one changed arithmetic boundary at a
time remains necessary for causal attribution. See the
[research and proposed localization tests](loss-fix-research.md).
