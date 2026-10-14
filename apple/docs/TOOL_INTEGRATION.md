# Turning experiments into reusable fine-tuning tools

The reusable product lives in `apple/src/exl3_mlx`, with its CLI, Python API,
Studio interface and regression tests. A successful experiment alone does not
qualify a feature for the product. Experiments keep immutable runtime snapshots;
new product changes must not alter an active run.

## Integration status (2026-10-13)

| Capability | Product status | Remaining work |
| --- | --- | --- |
| Streamed frozen EXL3 weights, input gradients, chunked loss, bounded caches | Package implementation | Continue checkpoint-specific numerical and capacity qualification |
| Standard data import, project preparation, training, resume and paired comparison | CLI, SFTTrainer and Studio | Improve end-to-end usability as orchestration is integrated |
| Complete agent edits and source evidence | Upstream/ktune import and replay tooling | Bring the newer cross-source assembly and family checks behind a documented reusable command |
| Packed-cache budget for gated corpus campaigns | Preparation script now accepts `--packed-cache-mib`; 40 CPU tests | A cache-size setting is not evidence of a speedup |
| Detailed paired token audit | Package module, `exl3-mlx-audit`, and automatic standard project comparison; local tokenizer/EOS categories plus optional reviewed categories | [Input contract and limits](EVALUATION_AUDIT.md); semantic code/rationale categorization and generated-patch quality remain separate |
| Mac Studio ingestion / M5 training | Experiment orchestration with reviewed lease and runner scripts | Generalize host/path configuration and integrate with project status/resume |
| Safe phase-boundary replacement of a campaign | Independently reviewed experimental operator | Integrate only after concrete identity checks and operational evidence; not a general public stop/recovery API yet |
| Exact-base generated-code evaluation | Experiment probes and separate compiler/correctness gates | General benchmark manifests and end-to-end generated patch evaluation |

The next product work should close these gaps rather than accumulate another
host-specific script. In particular, a user should be able to prepare a versioned
dataset, select a local or remote worker and resource budget, run preflight/train/
evaluate, and inspect a report through one project workflow. Until integrated,
the existing experiment coordinator must not be advertised as that workflow.

## Acceptance requirements for promotion

- General configuration: no fixed host, example count, checkpoint path or run ID.
- Dataset identity and complete-item coverage checks; explicit exclusions and no
  silent truncation. Evaluation partitions remain immutable across comparisons.
- Tests for failures as well as successful execution, including interrupted jobs,
  invalid memory budgets, mismatched masks and incomplete evaluation results.
- Independent review and a documented command/API. Claims distinguish CPU tests,
  hardware qualification, fitting and generated-code quality.
- A fresh versioned runtime for later runs. Preserve prior reports and active
  runtime snapshots so experiments remain reproducible.

## Cache control already promoted

The advanced `scripts/prepare_agent_corpus_run.py prepare` command accepts
`--packed-cache-mib` (default 2048). The selected value is bound into the project
and emitted phase commands. Invalid budgets fail during configuration validation.
The regular fine-tuning configuration already exposes this control; this change
closes the hardcoded-budget gap in the gated corpus preparation helper.

Increasing this cache can reduce repeated expert reads during recomputation, but
does not remove the initial reads of previously unseen frozen-prefix weights.
Choose a budget from observed resource headroom and measured phase timing, not
from an assumption that more cache accelerates every phase.

The M5 longest-item preflight supplied a concrete counterexample to that
assumption: increasing the packed cache from 2 to 6 GiB reduced logical packed
loads by 9,214,384,128 bytes, but the measured training intervals were 1,312.28
and 1,318.87 seconds. Peak MLX allocation increased from 67.32 to 71.32 GiB.
Loss and gradient norm matched exactly. These are single observations, exclude
initial payload hashing, and do not establish a wall-time speedup. Logical
application reads are not physical disk traffic. See
`apple/reports/m5-successor/preflight-cache-comparison.json`.

## Preparation must reproduce reviewed token identities

The advanced corpus preparation helper requires `train.prepared.jsonl` and
`test.prepared.jsonl` to be bound in the corpus manifest's `artifacts_sha256`.
An empty validation split can omit its prepared sidecar. After normal import,
the helper compares every full token sequence and `loss_start` against those
reviewed sidecars before binding provenance or emitting a runnable plan.

A mismatch leaves provenance pending. Rebuild the raw corpus and prepared
sidecars together, review the changed token identities, and use a fresh project.
Even a prompt-only change must be accounted for: it changes the model input and
can move the supervised boundary. Do not manually mark the project ready or
reuse an old baseline against changed inputs. The check reports the affected
split and row; the successful plan records per-split identities.

This stronger contract applies to the advanced reviewed-corpus preparation
helper. It does not require users of the ordinary SFT import flow to manufacture
sidecars manually. Active experiments continue to use their pinned helper and
separate, already-completed independent token-identity checks.

## Evaluation orchestration regression requirements

Independent review of the experimental successor evaluation sealer caught two
integration failures that isolated unit tests missed: the campaign runner emits
15 process records (emit/model/verify for each of five phases), and its controlled
environment helper must be entered as a context manager. Both were repaired and
tested in composed CPU flows before any evaluation launch.

When this orchestration is promoted into the project workflow, preserve tests
that exercise the actual runner record schema and actual environment lifecycle,
including restoration on child failure. Require the completed campaign, exact
final step/cursor, adapter identity and hashed logs before preparing comparisons.
Do not select checkpoints using held-out scores or infer completion from a
receipt filename. The independent review covers CPU preparation only; it is not
hardware qualification or a finding that model quality improved.

The review is retained in
`apple/reports/m5-successor/successor-evaluation-sealer-independent-review-v2.json`.
The sealer itself remains experimental; these are integration requirements, not
a claim that remote orchestration is already available in the public CLI.
