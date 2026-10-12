# ktune import and evaluation review

Date: 2026-10-12. Scope: read-only inspection of the actual
`jason-m5-macbook:/Volumes/Scratch/model-work/ktune` archive and harness; review of
the staged importer, CLI and training-fit comparison; CPU-only receipt checks.
No candidate build, GPU benchmark or model training was performed by this reviewer.
This report does not declare the long-context backend or the complete tool passed.

## What the evidence establishes

The previous standard-SFT pilot establishes operational updates, resume and export.
Its single synthetic held-out answer likelihood improved while the capped generated
answer remained wrong (`standard-sft-verification.json`). It establishes no Metal
optimization improvement.

The inspected ktune archive contains 3,030 official Qwen submissions, including
66 accepted records and 65 harvested promoted diffs. Every accepted receipt reports
one correctness case and 64 checked steps. These are useful historical results,
but their tested workload is narrow. The local experiment index is a summary,
not a substitute for a complete measured edit and its receipts.

`ktune_eval.py` audits the actual official and local schemas without importing MLX.
It recomputes official pair means/composite scores and the local four-pair gate,
requires finite positive timings and noncontradictory correctness/identity fields,
and keeps local screening distinct from official promotion. It expressly does not
authenticate an authority, verify source payloads, rerun tests or certify quality.

The complete official dump has SHA-256
`e77ed5504276c602a0ba462abd75af775b1162ae21eda51e480e4fb9ea674b96`.
The strict receipt audit admits 53 of its 66 accepted records:

- One accepted record was superseded and has no promoted source reference.
- Six promoted references differ from the measured/submitted commit. Rebase or
  merge equivalence could explain this; the available receipt alone does not prove it.
- Seven early receipts have different candidate and baseline golden hashes.
  One also has a commit mismatch. These need benchmark equivalence evidence,
  not a presumption that the kernels were wrong.

There are 13 excluded accepted records after accounting for the overlap. The
remaining 2,964 records are not promotions and fail promotion eligibility by design.
Do not call these counts 2,977 corrupted records. Even the 53 passing audits do not
bind a downloaded `.diff` to Git source content; parent-file mode adds that check.

The actual local `mlxfast-qwen38-ktune-v210/0001/result.json` was also checked. Its
aggregates are internally consistent, but its four-pair screen fails: two wins,
negative confidence lower bound, and excessive movement in the unrelated axis.
The validator retains this valid negative result rather than promoting it.

## Concrete implementation findings

The following were sent to the owning agents with CPU reproductions where applicable.
Statuses below reflect the subsequent source review and targeted CPU tests.

| Finding | Consequence | Required response | Review status |
| --- | --- | --- | --- |
| Unified-diff parser retains old file state across `diff --git` without new `---`/`+++` headers | A malformed second file is admitted and attributed to the first file; mixed mode-only sections are silently omitted from prompt context | Reset section state; require headers and complete supported text hunks for every section; reject unsupported mixtures | Fixed; inspected and regression tested |
| `ktune_cli.py` sets outer status to `trained` after any normal trainer return | A stopped or zero-update run can be labelled trained and trigger a fit check with no checkpoint | Preserve actual project status and skip automatic fit checking when incomplete | Fixed; inspected and regression tested |
| `ktune-source.json` is written outside the immutable project hash manifest | Provenance can change after preparation without resume validation noticing | Bind importer manifest, sidecars and source mapping into project provenance | Fixed; copied evidence and source mapping checked on project open/resume |
| Training-fit comparison does not prominently state its subset size or check matching counts | A first-row fit result can be read as a dataset-wide result | Include available/evaluated rows and supervised-token counts; reject mismatched comparisons | Fixed; inspected and regression tested |
| Snapshot source is called `authenticated_snapshot` although checks use self-contained hashes | Name overstates the guarantee | Use checksum/integrity wording; retain explicit no-independent-authentication limitation | Fixed: `checksummed_snapshot` |
| Interrupted/failed evidence binding leaves a published project without the final provenance hash map | Ordinary resume can bypass the unfinished evidence step; a retry encounters the partial directory | Persist a requirement in initial project preparation, publish a durable pending marker before copying, block training/resume until completion, and recover owned partial stages | Fixed; initial prepare gap, interrupted copies, final-publication interruption, changed reports/files, and staging symlink/escape covered by CPU regression tests |

Parser reproduction used a valid `f.metal` hunk followed by a `g.metal` section
containing `@@ -1 +1 @@`, `-old2`, `+new2` but no file headers. The preimage function
returned both hunks under `f.metal` instead of rejecting the malformed target.

Positive aspects: examples are explicitly training-only; target-selected hunk
locations are disclosed as target-derived context; outcome notes and timings stay
out of prompts; complete targets are retained without truncation; evidence tiers,
negative attempts and exclusion reasons remain in sidecars; pinned parent-file
mode checks the complete Git diff against the archive. No held-out quality or
speedup claim follows from `verify_training_fit`.

Evidence recovery uses
`SFTTrainer.from_project(project, allow_pending_provenance=True).bind_provenance(source, report)`
with the original import report. The repair option only opens the project for
binding; training still fails until completion. A retry rebuilds its recorded
private stage or verifies every file in an already published evidence directory.
It cannot substitute a different report or silently accept a mutated bundle.
These checks address interrupted writes and accidental mutation; the unsigned
project manifest is not protection against a local actor intentionally rewriting
both content and its recorded hashes. Project source/runtime identity checks
still apply during recovery.

## Remaining design and effectiveness limitations

1. **Retrieval leakage.** The actual `docs/harness.md` says the lesson builder reads
   all tracked sources with no provenance, lineage or split filtering. That store
   is unsuitable for a held-out comparison. Disable retrieval or build a separate
   training-only store; quarantine held-out campaign history and derived lessons too.
2. **Protected lineage coverage.** The snapshot's excluded development records have
   IDs and artifact references, but lack sufficient code/commit hashes to detect
   all overlap with other imports. The importer cannot infer complete protection
   from that metadata. The conservative track/repository grouping helps, but does
   not establish an independently held-out benchmark.
3. **Historical local gates.** `docs/process.md` admits its statistical gate was not
   enforced for eight of ten written-up Qwen candidates. Notes and local positive
   summaries are not verified promotion labels. Strict local receipt recomputation
   is useful; it does not turn those results into independent confirmation.
4. **Editable task context.** Hunk reconstruction contains target-selected locations;
   full parent files still lack dispatch/build/workload contracts. Both are honest
   editing exercises, not proof that a model can discover useful optimizations.
5. **Protocol boundary.** The existing ktune edit boundary requires complete fenced
   SEARCH/REPLACE blocks with unique parent matches and no new files. Most official
   targets are unified diffs. The new `ktune_patch.py` exporter now validates and
   applies a generated unified diff in a disposable Git index using pinned source
   blobs, then emits full-file blocks with exact replay equality. It requires
   whole-file authorized ranges in the receiving task. New/deleted/renamed/mode/
   binary/EOF-newline edits, unsupported filenames, delimiter collisions and
   output over the existing 250,000-byte harness cap fail closed. It performs no
   compilation or benchmark. The adjacent upstream `metaltune.searchreplace`
   already provides verified edit conversion; using its compact unique-hunk
   target format directly is a useful future corpus improvement.
6. **Context support is separate.** Larger configured limits are not numerical proof
   of longer-context attention, training, resume or generation. Those tests are
   owned by the backend implementation and must be reported separately.

## Execution-based acceptance plan

Freeze prospective task parents, editable ranges, compiler/runtime versions,
reference implementations, workloads and numerical tolerances before obtaining
model candidates. Keep the evaluation test and timing implementation outside the
editable surface. Store hashes of every generated answer, applied source tree,
build artifact, reference, configuration and result receipt.

Use the existing challenge-freeze/verify and strict edit-replay mechanisms. A
candidate must apply, compile and preserve interfaces before timing. Numerical
tests should cover fresh random inputs, tails/nonmultiples, supported dtypes and
layouts, extreme values where specified, and shape-dependent fast paths. A result
that is incorrect, nonfinite, timed out, or changed the measured workload fails
regardless of apparent speed.

Benchmark base and adapted models on the same task set, generation settings,
candidate count, tool access and revision budget. Separate one-shot performance
from fixed-budget iterative optimization. Begin without the mixed lesson store;
then compare with an identical training-only retrieval policy. Use independent
confirmation timing after selecting a candidate, synchronized/interleaved runs,
warmup separation and actual target hardware. Record kernel time, end-to-end cost
and memory constraints as applicable. A fixed-reference speedup is not an
incremental speedup over the preceding champion.

Primary measure: fraction of predeclared tasks producing a correct, practically
faster candidate within the fixed budget. Report compile/correctness rates,
timeouts, unchanged/slower cases and regressions across all tasks. Do not remove
failed tasks from the denominator. Group uncertainty analysis by independent
kernel family; tiny related examples are a feasibility pilot, not broad quality.

## Verification receipts

- `tests/test_ktune_eval.py`: 51 CPU-only cases passed. They include wrong-but-fast
  results, NaN/infinity/bool timings, summary tampering, missing/duplicate pairs,
  source/golden drift, superseded records, noisily positive and no-op results,
  local-vs-official distinction, and malformed records.
- Ruff passed for the new validator and its tests.
- `tests/test_ktune_patch.py`: 24 real CPU Git cases passed, including exact replay,
  multiple files, dirty checkout isolation, unchanged repository bytes, disabled
  external filters, unsupported changes/paths, wrong preimages, symlinks/gitlinks,
  delimiter collisions, output preservation and the harness byte limit.
- The full-file output grammar also replayed exactly through the actual remote
  `ktune.edits.apply_response` and `metaltune.searchreplace` with authorized range
  1..3; MLX was not loaded. This was a synthetic syntax/replay check, not candidate
  code execution or a measured optimization.
- Importer plus CLI targeted suites: 54 CPU tests passed. After the interruption
  fixes, the staged project/API plus CLI suites passed 62 CPU tests with
  `PYTHONPATH=src` (59 project/API, 3 CLI). The project checks include durable
  required/pending provenance, exact-report recovery, changed published evidence,
  unowned staging paths, and one-time verbose forwarding for train/evaluate.
  Ruff passed for `sft.py` and its project tests. Backend/GPU receipts are tracked
  separately.
- Portable accepted-receipt details, all-status counts and actual local negative
  audit: `ktune-evidence-audit.json`. The full per-record all-status audit remains
  in the workspace at `../../ktune-official-receipt-audit.json`.

Passing these checks means the receipt validator behaves as tested. It does not
mean the model learned to optimize kernels, or that the full tool has completed
long-context and execution-quality validation.
