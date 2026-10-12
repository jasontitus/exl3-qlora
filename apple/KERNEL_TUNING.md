# Training on the ktune archive

The importer turns complete historical edits into ordinary `prompt`/`completion`
SFT records. It preserves evidence, exclusions, lineage and complete token lengths
beside the records. It never executes an archived patch or considers a falling
language-model loss proof of a faster kernel.

The initial task is **historical patch-editing practice**. The default prompt
contains every old-side patch hunk and a generic optimization request; the answer
is the entire original edit. Its report says
`prompt_context=target_selected_preimage_hunks`: the old code predates the edit,
but the *choice of lines* comes from the answer. This is useful for checking the
training mechanism and practicing edit syntax. It is not a task-disjoint kernel
optimization benchmark or evidence of discovering a new optimization.

## Command-line recipe

The full-parent pilot **completed one optimizer update** using the complete
5,809-token example (5,808 input positions, 811 supervised answer tokens). Training
took **1,862.1 seconds (31.0 minutes)** and peaked at **73.95 GiB MLX memory** under
the 80 GiB cap. The saved checkpoint has 160 finite adapter tensors containing
3,153,920 parameters; all 80 zero-initialized LoRA B tensors became nonzero. Its
322 optimizer tensors are finite, checksums match, and Adam records step 1.
See the [checkpoint audit](reports/ktune-full-parent-checkpoint-audit.json).

That is approximately **3.12 serialized example tokens/s** or **0.44 supervised
answer tokens/s** across the complete update. Fresh full-context scoring has
measured **523–570 seconds per condition**; a base/adapter pair takes roughly
another 18 minutes for this example. These are training/scoring measurements,
not generation throughput.

**The operational check passed, but the first update's fitting check failed.**
Fresh scoring of the same 811 answer tokens gave base NLL **0.5012578592** and
adapter NLL **0.5053724668**, an increase of **0.0041146076** (lower is better).
The logits changed, but the original adapter made this recorded answer less
likely. See the [preserved first-update fit receipt](reports/ktune-full-parent-first-update-fit.json).
This is not an improved-fitting recipe or evidence of useful kernel quality.
The fitting check does not generate, compile or benchmark a new kernel.
The [combined CPU verification audit](reports/ktune-full-parent-verification.json)
records `training_target_nll_improved=false` alongside the operational checks.

A CLI `status=completed` or zero exit code means the requested computation
finished; it does **not** mean fitting passed. In `fit.json`, `loss_delta` is
adapter minus base answer NLL: negative is improved likelihood on the selected
rows, positive is worse. `quality_certified=false` remains explicit either way.

The original checkpoint and optimizer state are preserved. All three separate
inference probes also worsened NLL: +0.1 gave 0.5090817250, -0.1 gave 0.5019030712,
and adaptive +10 gave 0.5074643706. This bounded series is complete; no adapter
was promoted as an improvement and none of these probes is an optimizer update.
The [aggregate](reports/ktune-strength-diagnostics.json) records all scores, exact
A/B derivation checks and unchanged source/checkpoint identities. Native/head
and sparse-attention component VJP checks pass, and the current suite passes
**518 tests plus 27 subtests**. Those results do not resolve the fitting failure;
see the [diagnosis update](reports/first-step-diagnosis-update.md).
**5,809 tokens is the largest complete training example tested**, while 32,768 is
only the implementation ceiling.
The earlier 32 GiB attempt completed
the forward pass but failed during backward before an update; it is not an
established recipe for this length.

Run from the installed `apple` environment. The paths below identify the actual
October 12 pilot. Its snapshot and project already exist on this Mac: inspect
them rather than rerunning creation commands. For an independent run,
choose new output paths. The importer performs CPU tokenization before training
starts.

```bash
cd ~/experiments/exl3-qlora/apple
export EXL3_MODEL="$HOME/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"

uv run exl3-mlx ktune snapshot \
  --source jason-m5-macbook:/Volumes/Scratch/model-work/ktune \
  --output artifacts/ktune-20261012/source

uv run exl3-mlx ktune prepare \
  --source artifacts/ktune-20261012/source --model "$EXL3_MODEL" \
  --output artifacts/ktune-20261012/full-parent-inspection \
  --context-mode parent_files \
  --parent-repo jason-m5-macbook:/Volumes/Scratch/model-work/metaltune/data/repos/mlxfast-qwen38-125b-a6b-engine \
  --case-id official:qwen38-125b-a6b-mlx-v1:1a621870-4993-4c43-9472-6f7d8f745140
```

Inspect the inspection directory's `ktune-import.json`, `length-audit.jsonl`,
and whole `train.jsonl` prompt/answer. The actual training configuration is:

```bash
uv run exl3-mlx ktune finetune \
  --source artifacts/ktune-20261012/source --model "$EXL3_MODEL" \
  --output artifacts/ktune-20261012/full-parent-pilot-development \
  --context-mode parent_files \
  --parent-repo jason-m5-macbook:/Volumes/Scratch/model-work/metaltune/data/repos/mlxfast-qwen38-125b-a6b-engine \
  --case-id official:qwen38-125b-a6b-mlx-v1:1a621870-4993-4c43-9472-6f7d8f745140 \
  --steps 1 --rank 4 --learning-rate 1e-5 \
  --profile development --memory-gib 80 --reserve-gib 12 \
  --packed-cache-mib 8192 --decoded-cache-mib 0 --chunk-size 8192 \
  --verify-fit --fit-examples 1
```

This run uses an **80 GiB process cap and an explicitly selected 12 GiB system
reserve**, with 8 GiB packed cache and decoded caching off. The reserve was
chosen for the available startup headroom on this host. It changes the startup
requirement from the development preset's 112 GiB to 92 GiB free/reclaimable;
the watchdog still enforces the recorded reserve while running. This is a
run-specific choice, not a new default or permission to ignore a failed
headroom check. Use the preset's 32 GiB reserve when that headroom is available.
Caches are inside the process cap; the cap is not an OS hard allocation quota.

The development preset leaves the numerical chunk at 512. This pilot explicitly
chooses **8192**, which changes floating-point reduction order and is fixed in
the project's numerical contract. The measured same-chunk cache parity and
single-layer timings are in the [memory review](reports/high-memory-cache-review.md);
they do not establish full-model numerical parity or an end-to-end speedup.

The commands below describe deliberate continuation of the saved checkpoint.
The first update and all three tested strength variants failed the fitting
check; no further update was run in this bounded series. These commands document
continuation mechanics, not a recommended remedy or promise of improvement:


```bash
# Total target: 2 updates, so this adds one update after completed step 1.
# Omitting resource flags preserves this project's saved 80/12 GiB limits,
# packed cache, and immutable chunk 8192.
uv run exl3-mlx resume \
  artifacts/ktune-20261012/full-parent-pilot-development --steps 2 --no-compare

# Fresh base/adapter teacher-forced scoring after that continuation.
uv run exl3-mlx ktune verify \
  artifacts/ktune-20261012/full-parent-pilot-development
```

`resume --steps` is a total update target. To raise the reserve on a later resume,
add `--reserve-gib 32`; this is recorded for the attempt and inherited by later
resumes and verification. An explicit `--profile development` resets resource
defaults, including the reserve back to 32 GiB, unless further flags override
them. Neither form changes the project's chunk, learning rate, data or optimizer
contract. Resume requires the recorded engine and dependency identity; keep
source frozen until training and the automatic fresh fit checks have finished.

`--verify-fit` and `ktune verify` compare base/adapter likelihood on the first N
immutable training rows, labeled as a fitting check. `--fit-examples N` records
that subset size in the project; the default and this one-row recipe use 1.
Later `ktune verify` uses the recorded project setting. Results include
available/evaluated example counts, supervised token counts and the selected-row
hash. They do not certify kernel correctness, speed or held-out quality.
`ktune finetune` deliberately omits the short-answer generation comparison;
`--no-compare` preserves that choice during ordinary project resume.
Use `--prepare-only` to stop after creating the reproducible project. A smaller
`--max-length` excludes a whole oversized record instead of cropping it.

## Copy only the evidence needed

The snapshot helper uses the existing SSH trust configuration and `BatchMode=yes`.
It reads an explicit allowlist: README and library documentation, the small
optimization snapshot, official receipts and promoted diffs, the experiment
index, source manifest, and two provenance scripts. It excludes `.env`, `.git`,
the compressed general corpus, ignored build trees, and campaign runs. The default
transfer limits are 64 MiB compressed and 256 MiB expanded. Existing output paths,
links, path traversal, duplicate archive entries and unexpected files are rejected.

```python
from exl3_mlx.ktune_data import snapshot_ktune

snapshot_ktune(
    "jason-m5-macbook:/Volumes/Scratch/model-work/ktune",
    "artifacts/ktune-source",
)
```

`snapshot-manifest.json` records the remote Git revision and every transferred
file digest. Working files may differ from that revision; the helper does not
claim an atomic Git checkout or independent authentication of measurements.
The importer checks consumed files against this transfer manifest when present.
An existing local checkout with the same layout is also accepted.

## Import and inspect before training

```python
from exl3_mlx.ktune_data import import_ktune

report = import_ktune(
    "artifacts/ktune-source",
    "~/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw",
    "artifacts/ktune-data",
    max_length=None,
)
print(report["counts"], report["length_audit"])
```

`max_length` includes BOS, the serialized prompt, all answer tokens, and EOS.
`None` uses complete examples up to the backend's `MAX_CONTEXT_TOKENS` (32,768)
and reports the largest selected length. The resource guard can reject a run
below that ceiling. The importer does not truncate, chunk an answer, or discard
prompt context to make a patch fit. Every over-limit or unsupported example is
recorded with its reason, including when no examples remain eligible.

`ready_for_sft=true` means at least one complete eligible record was imported.
It is not a memory-feasibility or quality guarantee for the selected records.

The returned report includes `counts`, `max_length`, `ready_for_sft`,
`selected_case_ids`, `evidence_counts`, `selected_lineages`, all source/tokenizer
hashes, and `artifacts_sha256`. Output files are:

| File | Contents |
| --- | --- |
| `train.jsonl` | Canonical prompt/completion rows with case ID, source lineage ID and evidence tier |
| `valid.jsonl`, `test.jsonl` | Empty; this archive import does not manufacture held-out quality evidence |
| `length-audit.jsonl` | Complete prompt/target/total token counts, token digest, limits and disposition |
| `provenance.jsonl` | Per-edit source hashes, original receipts/admission limits, source bindings and selection reasons |
| `ledger.jsonl` | Negative, unverified, excluded, oversized, duplicate and unselected material; no negative positive-SFT targets |
| `ktune-import.json` | Import policy, source identities, limits, counts and artifact hashes |

The standard `SFTTrainer` accepts the output directory as `train_dataset`. Its
preparation stage copies and tokenizes the records under the ordinary project
immutability and resume rules. No validation examples are copied from training.

For a deliberate Python continuation after reviewing the failed fitting check,
reopen the already prepared project. This retains its evidence binding, complete
data, numerical settings and recorded resource limits:

```python
from exl3_mlx.sft import SFTTrainer

trainer = SFTTrainer.from_project(
    "artifacts/ktune-20261012/full-parent-pilot-development", max_steps=2
)
trainer.train(resume_from_checkpoint=True, compare=False)
trainer.verify_training_fit()  # known-answer fitting evidence, not kernel quality
```

Automatic short exact-answer generation is not an appropriate kernel-quality
test. A whole-patch completion often needs hundreds or thousands of output tokens.

## Choose a bounded pilot explicitly

`max_examples=1` selects the shortest complete eligible row, with ties broken by
case ID. This is an explicit pilot selection and every omitted row remains in
the ledger. `selected_case_ids=[...]` selects exact complete candidate identities;
unknown or duplicate IDs fail. Selection does not weaken evidence checks.

A tokenizer-only audit of the copied October 12 archive found an eligible
1,233-token record:

```python
pilot = import_ktune(
    "artifacts/ktune-source",
    "~/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw",
    "artifacts/ktune-pilot",
    selected_case_ids=[
        "official:qwen38-125b-a6b-mlx-v1:1a621870-4993-4c43-9472-6f7d8f745140"
    ],
)
```

That edit changes the prefill mixer activation M tile from 64 to 32 with its
matching dispatch/gate changes. The length is tied to the exact tokenizer and
prompt template; use the current report as authority. A one-example fit tests
mechanics and memorization, not generalization.

The **M1 Ultra is the training host; this particular historical kernel targets
the M5/NAX path**. Its official receipt names an M5 Max and its parent code gates
execution on NAX-capable GPU architecture and macOS 26.2 or newer. Training an
adapter on those source tokens does not require the training host to execute
that kernel. A future correctness/performance check must use the matching target
hardware and runtime and confirm that the changed dispatch path actually ran;
timing an M1 fallback cannot validate the archived M5 optimization.

The prompt preserves the source's shape/dtype/dispatch guards, but it does not
provide an independent task card specifying the measured workload, target chip,
or pre-change profile. The answer's comment assumes a 40-core GPU and a particular
1024-row workload. Those facts need a separately sourced, pre-candidate task card
before testing whether a model can choose the right optimization. They must not
be reconstructed from the successful answer and then presented as unseen input.

Whole-answer likelihood is especially limited here: the 2,341-character target
contains 254 characters of added executable code; most remaining text is patch
metadata, copied context, removed code and explanatory comments. A lower loss can
reflect better copying or edit formatting even when the coordinated code changes
remain wrong. Completion replay, numerical correctness and measured performance
are separate checks.

## What each evidence tier means

The default selected tiers are `official_promoted` and `historical_promoted`.

| Tier | Meaning and use |
| --- | --- |
| `official_promoted` | Archived official accepted/promoted result passes the receipt audit. A full text diff is required. This is historical evidence for the named harness, workload and machine. |
| `historical_promoted` | Checksummed library record reports an earlier scored promotion; the original evidence scope and prior retrieval-only restriction remain in provenance. The user's new training use does not retroactively grant full-edit efficacy admission. |
| `source_only` | Optional explicit selection; source code without demonstrated native correctness or performance. |
| `correctness_only` | Optional explicit selection; correctness evidence without measured improvement. |
| `historical_improved_superseded` | Optional explicit selection; historical improvement superseded later. |
| `official_unverified` | Receipt/source identities are incomplete or disagree. Needs evidence equivalence; not a finding that the optimization is incorrect. Ledger only. |
| Negative / nonpromoted / index-only | Failure lessons or insufficient evidence, preserved for inspection. Never positive SFT answers. Rejected score does not automatically mean correctness failure. |

For example, `tiers=("source_only",)` deliberately creates an unvalidated
source-editing dataset; it cannot acquire a performance claim by being selected.
Post-result notes, scores, mechanism summaries, and failure explanations are kept
out of the prompt. The complete archived edit is the target, including its own
comments. No generated source is executed by this importer.

## Lineage and time boundaries

All records from a repository/track and records linked by commits or identical
diffs are joined before selection. Known protected examples quarantine their
whole connected lineage. Unknown protected/index-only entries without source
hashes cannot be exhaustively matched; the report explicitly disclaims complete
contamination protection. The entire incremental official Qwen campaign stays
together, regardless of the number of patches.

`train_until="2026-09-12T00:00:00Z"` is an optional cutoff. A lineage with any
unknown or later timestamp is excluded in full. It does not create a test set.
Randomly splitting the same incremental campaign is not an acceptable substitute
for independent tasks.

## Resolve full parent files when available

For a local Git checkout or an SSH repository containing the archived source commits:

```python
report = import_ktune(
    "artifacts/ktune-source", model_path, "artifacts/full-parent-data",
    context_mode="parent_files",
    parent_repo="/path/to/the/pinned/challenge-repository",
)
```

For the copied Qwen archive, an available remote is
`jason-m5-macbook:/Volumes/Scratch/model-work/metaltune/data/repos/mlxfast-qwen38-125b-a6b-engine`.
Use that string as `parent_repo` with the exact pilot `selected_case_ids` above.
Read-only SSH Git queries retain host-key checks, quote every argument, and cap
each query at 30 seconds and 16 MiB of output. No clone or remote writes are needed.
The actual complete-parent pilot tokenized to **5,809 tokens: 4,998 prompt and
811 target**, without truncation. The one-step result above establishes training
feasibility for that complete example and its exact recorded configuration. It
does not establish that longer imported examples fit or that 32,768 tokens have
been trained successfully.

The October 12 full-parent official archive audit admitted **24 complete edits**
from one connected campaign, with lengths from **3,606 to 25,486 tokens**
(median 17,806.5). Nineteen complete records exceeded 32,768 tokens; the largest
was 141,917. Nine selected-tier records had unsupported or unavailable source
bindings, and one was a duplicate complete token record. Those records remain
in the ledger. The audit does not establish that the largest admitted context
fits in memory: begin with the 5,809-token pilot, then measure longer examples
under the chosen resource profile. See
[the pinned audit](reports/ktune-parent-context-audit.json) for counts, hashes,
complete selected lengths and exclusions. The median admitted example is over
three times the tested length, and the 5,809-token update already peaked at
73.95 GiB against the 80 GiB cap. **The 24 imported examples are not 24 examples
proved to fit in 80 GiB.** Expanding beyond 32,768 would require
additional backend and memory validation; silently cutting these examples would
discard part of the source or target.

This verifies the unified diff byte-for-byte against the pinned parent/head Git
objects, reads complete regular UTF-8 parent files, and records their hashes.
Search/replace records require unique search blocks, exactly the pinned changed
file set, and a reconstructed result equal to the pinned head, plus matching
recorded preimage hashes where available. Missing commits or unsupported
bindings exclude the example; the importer does not fall back to partial hunks.

The report still labels these `target_selected_complete_parent_files`: the
target selected which files to show. Full files alone do not supply the build
contract, baseline workload, cross-file dependencies, or independent task
definition. Build genuine optimization tasks separately using predeclared source
snapshots, shapes/dtypes/layouts, numerical tolerances, allowed edits, tests, and
timing workloads known before the candidate is generated.

## Verify the trained output

**Training-fit checks are available; practical long-patch generation is not yet
available in this engine.** Its current `generate`/chat path recomputes the full
prefix through all 40 layers for every new token, without a persistent KV cache.
The completed-update pilot's 5,808-position forward took about 584.2 seconds;
the earlier failed low-memory attempt's forward took about 835.3 seconds. These
are training-forward measurements under different cache/chunk configurations,
not generation benchmarks or a complete-update speedup comparison. They
illustrate why generating hundreds of patch tokens is impractical through this
path. A configured output-token limit does not make that generation fast.

`ktune verify` intentionally uses teacher-forced masked-answer NLL: it supplies
the recorded answer and performs one backbone forward per selected example for
each base/adapter condition, then scores the answer tokens. It does not generate
a new patch. Improved likelihood establishes a fitting result, not that the
adapter can produce a correct or faster kernel.

The generated-output steps below are acceptance gates, not a completed automated
quality workflow. Practical execution requires an adapter-compatible inference
integration with efficient KV caching, followed by the independent kernel
harness. An existing serving stack such as `mlx-serve` may be a future integration
target; compatibility with these adapters and model semantics is not established.

1. Check adapter-only updates, masked-answer loss and exact resume on the fixed
   imported data. Report these as training-engine checks.
2. For an unseen predeclared kernel task, give the baseline and adapter the same
   full context, token budget, decoding policy and allowed tool budget. Save both
   complete outputs, truncation/EOS state, prompts and identities.
3. Review and apply each patch in an isolated checkout. Reject missing edits,
   forbidden files, partial patches, tests that weaken the gate, and changes to
   benchmark or correctness machinery. Compile and run the real independent
   correctness harness; exact text similarity is only an edit-format diagnostic.
4. Only correctness-passing candidates enter alternating baseline/candidate
   timing with declared thermal/concurrency controls, fixed repeats and no
   repeat-until-favorable selection. Recompute the receipt from raw paired times.
5. Report task-level correctness, useful speedups, failures, regressions and
   uncertainty across independent task families. Keep development tuning separate
   from the final one-time test. The small historical dataset cannot certify
   general kernel-optimization quality by itself.

The evidence validator in `ktune_eval.py` audits archived receipts without
executing patches. Its result does not replace new measurements on the target
machine or authenticate a newly generated candidate.

## Hand a generated patch to ktune

Historical official targets use unified diffs; ktune's edit runner consumes
SEARCH/REPLACE blocks. Convert a complete generated patch against the exact
pre-change commit before submitting it to the existing runner:

```bash
uv run exl3-mlx ktune export-edit \
  --repo /path/to/kernel-repository \
  --parent FULL_40_CHARACTER_PARENT_COMMIT \
  --patch candidate.diff --output candidate.edit \
  --allow-path Runner/FastModel/TrackPrefillMixerAct.swift \
  --allow-path Runner/FastModel/TrackPrefillMixerActMetal.swift
```

The converter uses an isolated Git index, applies the diff strictly, and verifies
that its SEARCH/REPLACE output reconstructs the same files. It writes
`candidate.edit` and `candidate.edit.receipt.json`; it does not change the source
checkout or run the resulting code. Only ordinary existing UTF-8 text files are
supported. New/deleted/renamed files, binary edits, symlinks and mode changes fail
closed. Whole-file replacements must also be permitted by the receiving ktune
task's file allowlist and edit-size limit.

Archive the new runner's correctness and timing receipts with the generated
patch. Conversion success establishes edit compatibility only; the independent
kernel harness establishes correctness and any speedup.
