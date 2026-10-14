# Preparing a reviewed exact-edit corpus

`exl3-mlx ktune prepare-reviewed` prepares a portable, explicitly selected corpus using the same local tokenizer and ordinary SFT project flow. It does not fetch repositories, execute supplied commands/scripts, load model tensors, or start training.

```sh
exl3-mlx ktune prepare-reviewed --source bundle/corpus.json \
  --model /path/to/local/model --output /path/to/new/dataset \
  --project /path/to/new/project
```

The optional project is prepared with provenance required, its actual prepared rows are compared with the reviewed token IDs/masks, and only then is provenance bound. Omit `--project` to publish the dataset alone. API: `exl3_mlx.reviewed_corpus.prepare_reviewed_corpus(source, model, output, project=None)`.

## Input contract

The manifest has `format: exl3-reviewed-agent-corpus-v1`, `max_length` (complete serialized tokens), `tokenizer` (the exact local `tokenizer_identity` mapping), `model_config_sha256`, `policy`, ordered `entries`, `frozen_splits`, and `manifest_sha256`. The last field hashes UTF-8 compact, sorted-key JSON of all other fields (`ensure_ascii=False`, no nonfinite numbers). All file references are relative `{ "path": "...", "sha256": "..." }` objects; symlinks, traversal and missing/hash-mismatched inputs fail.

Each entry declares a unique `id`, `kind: exact_edit`, explicit `split: train|valid|test`, four `inputs` references (`source`, `raw`, `prepared`, `replay`), and a `review` reference. A review has `format: exact-edit-review-v1`, `status: approved`, a named `reviewer`, and `cases` containing the exact entry ID and identical four input references. Every case in each selected review must be selected. Reviewer names are assertions, not authenticated signatures.

Source records bind repository, head commit, all relevant `commits`, optional `lineage_keys`, and complete changed/context `files`. Each file has `role`, `before`, `after`; each non-null image has text `content`, SHA-256, Git `blob` SHA-1, and immutable mode `100644` or `100755`. Host edits belong in the same complete change. Source membership in the claimed historical Git tree is a review responsibility; this command verifies captured blobs, not a fresh repository checkout.

Raw rows use the package's native agent-edit tool schema, with exactly one final `exec_command` edit. The command is parsed as the package's fixed patch grammar and replayed with package-owned Git apply; arbitrary supplied shell is never executed. Prepared rows must equal ordinary local normalization/tokenization exactly, including prompt, IDs and loss mask. Prior replay receipts must match freshly replayed complete source bytes. Files without final newlines are supported. Empty-file additions and file mode changes are explicitly refused in this stage; no empty-file edit is silently omitted.

The policy contains all five arrays: `protected_commits`, `protected_patch_ids`, `protected_lineages`, `heldout_repositories`, and `lineage_aliases` (arrays of related lineage names). Exact normalized patch, source payload and repository/commit duplicates fail. Explicit lineage connections are transitive and cannot cross splits. This is finite protection, not exhaustive semantic clone detection.

`frozen_splits` must bind both raw and prepared files for valid/test, including empty files; train can also be frozen. Their exact bytes, ordering and rows are preserved. Incompatible key-order-dependent tokenization is refused rather than normalized silently. The local template must be explicit in `chat_template.jinja` or tokenizer configuration; no fallback template is invented. Adding/removing a separate template changes identity.

## Publication and limits

Every declared row is admitted or the entire dataset import fails. No truncation, silent exclusions, splitting, or automatic duplicate winner is allowed. Inputs are captured once, checksummed and checked again before publishing a fresh output directory atomically without replacement. Inputs are bounded to 64 MiB each and 512 MiB total captured bundle bytes.

Dataset publication and optional project creation are separate phases. If project preparation or the final exact token comparison fails **after** dataset publication, the validated dataset remains published. A partial project can remain with provenance pending; training is not started. Inspect the failure before preparing a fresh project. Atomic publication does not promise an all-or-nothing transaction across both directories.

Outputs include raw/prepared split pairs, complete per-row provenance, length audit, admission/overlap receipts and `ktune-import.json`. `ready_for_sft` means this preparation contract passed; it does not certify compilation, numerical parity, speedup, source authenticity or downstream model quality.

This first stage accepts exact edits only. It explicitly rejects a mixed corpus containing reasoning/diagnostic rows, including the current experimental 229-row corpus with 43 diagnostics; it never drops them. A subsequent generic admission contract must bind diagnostic evidence and source lineage, distinguish observations from inferred rationale, reproduce full target masks, and share the same frozen-split and review protections. That integration is pending.

Executable, self-contained examples of the complete manifest and approval schema are in `apple/tests/test_reviewed_corpus.py`; they use a tiny strictly local tokenizer and arbitrary train/valid/test sizes.
