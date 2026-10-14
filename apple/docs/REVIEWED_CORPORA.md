# Preparing a reviewed corpus

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

Version `exl3-reviewed-agent-corpus-v1` remains exact-edit only and explicitly rejects mixed corpora. Version `exl3-reviewed-agent-corpus-v2` adds diagnostics under the contract below. Existing experimental corpora are not automatically approved or migrated; the current 229-row corpus needs explicit migration/review of its 43 diagnostic evidence bindings before this importer can accept it. No row is silently dropped.

Executable, self-contained examples of the complete manifest and approval schema are in `apple/tests/test_reviewed_corpus.py`; they use a tiny strictly local tokenizer and arbitrary train/valid/test sizes.

## Version 2: mixed diagnostic and exact-edit corpora

Use the same CLI/API and set manifest `format` to `exl3-reviewed-agent-corpus-v2`. Exact-edit entries retain the v1 contract. A diagnostic entry has `kind: diagnostic` and four input references: `source`, `raw`, `prepared`, `evidence`. Its approval receipt uses `format: diagnostic-review-v1`, with the same exact per-case four-payload binding. An exact-edit approval cannot approve a diagnostic implicitly.

Diagnostic raw data is a user and assistant text pair, optionally preceded by one explicit system message, without tools. A system message requires `system_mode: task_instruction_only` (reviewer assertion that it contains generic instructions, not source facts) or `supplied_observations` with bound `system_spans` and a supplied-observation prompt mode. System text is preserved verbatim; it is never stripped to make a row fit. The source retains the complete before/after source images and shared commit/lineage protections. No tool replay is claimed for a diagnostic.

Evidence uses `format: diagnostic-evidence-v1`, `task_type` (`diagnose_bottleneck`, `explain_tradeoff`, `reject_unsafe_change`, or `choose_measurement`), `prompt_mode` (`before_only` or `supplied_observations`), `references`, `prompt_spans`, and `claims`.

Every reference has a unique `id`, `kind`, a captured `file` path/SHA binding, half-open UTF-8 byte `start`/`end`, and the selected range's `sha256`. Ranges cannot cut a UTF-8 character. Supported kinds:

- `before_source` / `after_source`: also require `source_path`; the complete evidence file must equal that exact source image before any range is selected.
- `author_report`: requires `origin` with a primary-source HTTP(S) `url` and explicit author/document `identity`. A bound report is an author assertion, not a local reproduction.
- `local_cpu` / `local_gpu`: additionally bind an `execution` JSON file with `executed: true`, nonempty `argv`, input/output file bindings, nonempty environment metadata, matching `kind` and compatible `backend`. CPU assertions require `cpu`; GPU assertions require `metal`, `cuda`, `rocm`, or `other_gpu`. The evidence file must be one of the bound outputs. These are imported execution assertions: the importer does not rerun or authenticate them.

Prompt spans and claims use the same byte-range/hash fields, plus a nonempty list of reference IDs in `refs`. Every reference must be used. `before_only` prompt spans can reference only `before_source`; supplied observations must be labelled `supplied_observations`. This validates declared relationships; semantic review must check whether the prose actually respects that information boundary.

Each claim also has `class` (`source_observation`, `inferred`, `author_reported`, `local_cpu`, `local_gpu`) and explicit nonempty `limits`. Classes must agree with referenced evidence kinds; inferred claims remain labelled inference. Claims must cover every nonwhitespace assistant target byte without overlaps. Missing, unknown or unbound references fail; the importer does not invent grounding for gaps.

A manifest `multi_views` array explicitly lists every same-source overlap group by entry ID, for example `[["edit-1", "diagnostic-1"]]`. Duplicate raw/prepared views are rejected. Distinct views of the same change must be in one split, pass all protection checks, and be explicitly selected; declared groups must match actual commit/patch/source overlap. The output reports these exposures. No automatic caps, extra copies or inferred selection are applied.

Version 2 preserves captured input files as flat content-addressed `.bin` artifacts with a `captured-inputs.json` index. The ordinary project provenance binding copies these too, so evidence survives moving away from the original bundle. Frozen split bytes/order and exact local token IDs/masks remain mandatory.

These gates establish **binding consistency**, not semantic truth, complete factual support, absence of prompt leakage, primary-source authenticity, numerical correctness, or performance improvement. Those remain explicit independent review and execution requirements. The tests include deliberately labelled imported assertions to verify this distinction; acceptance must not be described as hardware qualification.

Diagnostic source inspection supports regular UTF-8 text modes `100644` and `100755`, including new executable-text files and transitions between those modes. It records the full before/after content-and-mode transition hash separately from text patch identity; mode differences do not evade shared text/commit lineage protection. It never executes executable-text files or claims mode replay. Exact-edit rendering rules are unchanged. Binary/symlink source modes, deleted files, empty-file additions and pure mode-only diagnostics remain explicit refusals in this increment.

## Independent semantic review before approval

Passing the importer checks does not settle whether an answer teaches a sound
optimization decision. Review each claim against the captured passage, and
record required corrections before issuing `diagnostic-review-v1` approval:

- Separate observed code behavior, author measurements, local reproductions and
  recommendations. A suggested experiment is an inference even when motivated
  by a measured result. Bind reported timings to the report, not just the source
  code that implemented the change.
- Preserve qualifications: partial register-count recovery is not full recovery;
  a numerical difference reported inside a rounding floor is not an established
  quality regression. “Remains enabled independently” does not establish that a
  component was independently tested.
- Keep benchmark configuration and measurement meaning exact. A grouped or
  ungrouped implementation/test does not establish the grouping used for a
  separately reported benchmark. Peak memory is not the same measurement as
  temporary workspace. Leave missing hardware, precision, workload and grouping
  fields unknown rather than filling them from nearby code.
- Distinguish an algebraic repair from an executed numerical result. A formula
  can avoid the stated overflow while still failing near the dtype maximum;
  retain that limitation. An author showing the old failure does not itself
  provide an after-change test result. Bind claims about widened declarations
  to both before and after source when describing the transition.
- Check the information available in the prompt. Symptoms learned from a later
  report require an explicit supplied-observation boundary and evidence binding.
  They cannot be justified by labelling the prompt `before_only`.
- Date limitations of collection or rendering tools. A historical inability to
  replay a change does not establish that the current importer cannot handle it.
- Check every factual clause in a mixed claim span; split the span when one part
  is source-supported and another is a recommendation or hypothesis.

Corrected prompt or answer text creates a new dataset revision with newly
prepared token IDs and masks. Preserve the original data and receipts, and keep
new approval pending until the revised evidence is reviewed. Do not replace rows
inside a running campaign. Report any substantive defects found in its frozen
training set alongside that campaign's results; an unchanged held-out set does
not make those defects disappear.

### Portable execution receipts

The diagnostic importer validates an optional `portability` object on local CPU
or GPU execution assertions. It captures the original receipt by relative path
and SHA-256, then checks that the portable receipt changed only input/output file
paths. The mapping must cover exactly the original bindings, preserving hashes,
roles, order and all other metadata. Duplicate or unused mappings, nested
portability, missing originals and altered originals are rejected. Historical
absolute paths are metadata only and are never opened or executed.

This is a reproducible provenance consistency check, not authentication of who
ran an experiment or certification of numerical correctness or performance.

### Match answers to the agent-visible evidence

A provenance-backed statement is not automatically an appropriate target answer.
For a baseline-only diagnosis task, the answer may propose a repair, explain its
mechanism and specify an experiment. It must not assert that the proposed repair
already produced an exact speedup, numerical parity or memory reduction when
those observations are absent from the prompt. Keep historical outcomes in
provenance, or create a separate report-conditioned task with the observations
explicitly supplied. Likewise, do not assert unrelated repairs from omitted code.
Review prompt/answer compatibility separately from source truth and hash checks;
the importer cannot determine this semantic relationship mechanically.

## Explicit documentation scope in version 2

An exact edit may omit independently reviewed documentation changes while retaining
all captured source. This requires a source `edit_scope` object with format
`reviewed-file-edit-scope-v1`, coverage `complete_selected_file_edits`, sorted unique
`selected_paths`, and `omitted_paths`. Selected and omitted paths must partition
all changed files exactly. Every omission declares `classification: documentation`,
exact `before_sha256`/`after_sha256`, and a nonempty review reason. This increment
accepts only existing `.md` files with unchanged regular modes. A filename is not
proof of documentation semantics: the independent review must establish that no
runtime, build, test or host edit is omitted. Context records must have identical
before/after images and modes, so changed files cannot hide under a context role.

The final action must replay the complete selected files exactly. Historical
`omitted_documentation` receipts, when present, must agree on paths and hashes.
Version 1 continues to refuse scoped edits. Output explicitly records
`full_commit_replay: false` for a scoped edit, along with both full-source and
selected-edit identities. Both identities participate in duplicate, protected
source and cross-split checks: changing unrelated documentation cannot disguise
an identical executable edit. All omitted source bytes remain in provenance;
scoped replay does not establish full-commit compilation or benchmark results.
