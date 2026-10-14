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
| Detailed paired token audit | Package module and `exl3-mlx-audit` CLI; independently reviewed, 69 CPU tests | [Input contract and limits](EVALUATION_AUDIT.md); automatic category generation and workflow integration remain separate |
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
