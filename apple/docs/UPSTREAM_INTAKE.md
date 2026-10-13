# Pinned upstream-commit intake

`exl3-mlx ktune prepare-upstream --source manifest.json --model MODEL --output NEW_DATA`
produces standard prompt/completion training JSONL and full source/evidence sidecars.
This is historical reconstruction, with target-selected complete changed files.
The importer itself never compiles or executes candidate code. The separate
`ktune compile-metal` command compiles a narrow stdlib-only Metal translation-unit
pair; it does not establish host linkage, numerical parity, runtime performance,
or execution_verified admission. A correctness runner, timing harness, best-of-N
generator, and prospective optimization benchmark are separate gates.

The input is UTF-8 JSON with `format: exl3-mlx-upstream-commits-v1`,
`repositories: {id: {path: local_git_mirror, url: https_repository_url}}`,
`candidates: [...]`, optional `evidence: {...}`, optional `protected_keys: [...]`,
and `manifest_sha256`. Compute that SHA-256 from the remaining object serialized
with Python `json.dumps(sort_keys=True, ensure_ascii=False, separators=(',', ':'),
allow_nan=False).encode()`.

Each candidate has unique `id`, a `repository` ID, full lowercase `commit` SHA,
and optional direct `parent` SHA. Merges require an explicit selected parent;
root commits are unsupported. Optional `source_group` and `related_commits`
connect ports, PRs, revert families, and known related campaigns. `protected`
quarantines its whole connected group, including when its own source fails
reconstruction. `outcome` is `source_only` (default), `reported_improvement`,
`negative`, or `contradictory`; the latter two are retained but never positive
training targets, and their normalized patch equivalents are also excluded.

The context policy is `complete_changed_files`: every UTF-8 regular file in the
whole commit is retained before and after, including host dispatch, additions,
deletions, and changed documentation. Optional `dependency_paths` add complete
files at both pinned revisions. Dependency closure stays explicitly unknown;
there is no inference that these files compile in isolation. Binary files,
symlinks/submodules, unsafe paths, unavailable offline objects, >128 files,
>8 MiB per file, or >32 MiB before/after source exclude the entire candidate.
The importer disables external diff, text conversion, hooks, replacement Git
objects, lazy network fetch and Git transport. Materialize chosen blobs before
running it. It never executes commands from source, commit messages or evidence.

`task_kind`, if supplied, is an explicit reviewed declaration with
`value: kernel|host_dispatch|integration`, `reviewed_by`, and `evidence` citations.
It is not inferred from a filename. Missing type stays unknown. The complete Git
patch remains in provenance; the answer uses complete after files rather than
also repeating that entire patch. No target is cropped to fit token context.

Optional annotations are `task`, `operation`, `hardware`, `bottleneck`,
`constraints`, `rationale`, `architecture`, and `dispatch`. Each has `text`,
`basis: source|inferred`, and `evidence: [{source: ID, quote: exact_substring}]`.
Evidence entries contain `path` (relative to manifest), `sha256`, original HTTP(S)
`url`, and `phase: pre_change|post_change|unknown`. `commit_message` is a built-in
post-change source; absent rationale uses its complete text. A citation proves
which bytes were used, not the truth of a paraphrase or performance claim.

Only source-grounded annotations with entirely pre-change evidence enter prompt
Task/Operation/Hardware/Bottleneck/Constraints fields. Other values stay unknown
there and are retained in the answer/provenance with phase and inference labels.
This avoids silently turning post-result measurements into prospective task
specifications. Complete source is data, not trusted execution instructions.

Git object identity, before/after files, exact diff, and stable Git patch IDs are
checked. Source reconstruction is distinct from compilation, numerical parity,
and runtime measurement: those stages are always `not_run`. The strict
`--minimum-verification execution_verified` gate currently admits **zero** rows,
even if a manifest claims execution passed. A future harness must bind exact
parent/head/source, dependencies/build, toolchain, workload and hardware, raw
results and every candidate attempt. Best-of-N selection needs all failures and
an independent confirmation benchmark after selection; model-authored status or
source claims cannot establish those gates.

Counts are actual candidates, reconstructed records, unique normalized patches,
complete token-fit training rows and exclusions. A stable patch-ID deduplicates
aliases/cherry-picks conservatively (including whitespace variants). All related
repository/commit/declared-group/normalized-patch families remain one train-only
lineage. `protected_keys` accepts the exact `repository:<canonical URL>`,
`declared_group:<ID>`, `stable_patch:<ID>`, and `commit:<SHA>` identities. Existing
held-out protections must be explicitly carried into a new manifest; no automatic
blanket contamination-free claim is made. There is no random row holdout or
manufactured target example count.

## Reviewed source units and retrospective task conditioning

A candidate can explicitly set `context_policy: reviewed_units` with
`reviewed_units: {reviewed_by, dependency_review, units: [...]}`. Each unit has
`path`, `kind` (`complete_function`, `complete_kernel`, `complete_type`,
`complete_declaration`, or `whole_file`), and both `before`/`after` bindings
`{start, end, sha256}`. Indices are zero-based half-open **UTF-8 byte offsets**;
bind missing sides of added/deleted files to `[0,0)` and the SHA-256 of empty bytes.
Every range is checked against pinned full source. Units may not overlap or
reorder; their replacements and all unchanged gaps/tails must reconstruct each
full head file byte-for-byte. Missing changed files or uncovered edits fail.
`whole_file` declarations must include all bytes. Unchanged declared dependencies
remain prompt context, never unnecessary completion-copy targets.

This demonstrates complete edit coverage, not semantic completeness of a
function or its dependency closure. The named reviewer must actually review those
claims. No arbitrary automatic extraction, regex isolation, or standalone-build
claim is made. The entire before/after blobs and raw commit patch remain in
provenance even for source-unit examples. Rendered targets preserve exact code
bytes while omitting metadata-only blob/hash identifiers.

Default `task_conditioning: pre_change_only` uses only bound pre-change source
annotations in the prompt. Opt-in `historical_source` also permits source-grounded
post-change operation/hardware/bottleneck/task/constraint metadata, prominently
labeled a **retrospectively authored, target-derived task specification**. Inferred
annotations remain out of the prompt. Rationale/architecture/dispatch remain answer
fields. This is imitation/reconstruction training, not prospective optimization
discovery. Quote containment is mechanically verified; the semantic accuracy of
curator paraphrases and `outcome` classifications is explicitly curator asserted.

Tokenization uses a CPU-only local Hugging Face fast tokenizer and the same masked
prompt/completion chat-template prefix contract as MLX-LM. No model or Metal device
is needed to audit data. Exact source code strings are never rewritten to remove
hash-like literals or comments; only separately rendered Git metadata is omitted.

The distinct `exl3-mlx-upstream-import-v1` report uses the existing
`ktune-import.json` evidence-envelope filename. Its returned `manifest_sha256`
and `artifacts_sha256` can be passed unchanged to
`SFTTrainer.bind_provenance(dataset_directory, report)` after preparing a project
with `provenance_required=True`; no optimizer update is started by this importer.
Use the directory as the SFT source to preserve its explicit train-only split.
Standard preparation retains case/evidence metadata in provenance and strips it
from text tokenization. Preparation remains separate from actual training and all
execution-quality gates.

## One-command standard SFT project

`exl3-mlx ktune finetune-upstream --source manifest.json --model MODEL --output NEW_PROJECT --prepare-only`
imports, prepares a standard SFT project with mandatory provenance, and binds all
source/evidence before any update. Omit `--prepare-only` to run the normal trainer;
the same resource, adapter-layer, learning-rate, rank, activation-mode, stop/resume
project and fit-check machinery is used as archive fine-tuning. The default is two
updates. `--verify-fit` requests the normal fresh base/adapter training-likelihood
check; this remains train-only evidence, not generated-kernel or performance quality.
No separate Python binding step is needed. The imported dataset is preserved next
to the project as `NEW_PROJECT-upstream-data`. `--minimum-verification
execution_verified` still admits no examples until a bound execution verifier is
implemented, and a failed evidence binding prevents training.
