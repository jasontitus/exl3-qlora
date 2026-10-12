# Adversarial usefulness review: Apple EXL3 fine-tuning

Reviewed 2026-10-12 at commit `2e633062`. This is a source review plus bounded
CLI/data/integration probes, not a novice user study. No model weights were
loaded and no new training or generation ran for this review. Current numerical
and behavioral evidence remains in [QUALITY.md](../QUALITY.md).

**Verdict: useful as an experimental backend for one large model on this Mac;
not yet a simple, general-purpose EXL3 fine-tuning product for non-experts.**
A friendly interface would reduce setup errors, but cannot remove checkpoint,
context, output compatibility, throughput, or quality limitations.

## Is it built on an existing higher-level tool?

Partly. MLX supplies Metal/autograd/optimizers. MLX-LM supplies text/chat/completion
processing, masked causal loss, and LoRA forward code. Our loader, streamed
execution, training orchestration, checkpoint/resume contract, and CLI are custom.
We do not call MLX-LM's standard `lora` training workflow. The package dependencies
are pinned; this is actual library reuse, not end-to-end framework integration.
See [sources](../SOURCES.md), [data imports](../src/exl3_mlx/data.py),
[training](../src/exl3_mlx/training.py), and [workflow](../src/exl3_mlx/workflow.py).

| Existing workflow | What works today | What does not |
|---|---|---|
| MLX-LM | Reused components; overlapping raw JSONL text/chat/completion schemas | Its standard model loader, training CLI, adapter loading and fusion do not consume this complete streamed model/adapter workflow. |
| MLX-Tune | A possible familiar Python API to emulate or extend | No integration exists. Its inspected loader delegates to MLX-LM, and its SFT trainer uses MLX-LM's trainer or CLI. A wrapper alone does not supply our EXL3/Engram backend. |
| Hugging Face datasets | Locally exported rows matching our strict supported schemas | No direct dataset-ID ingestion or general column mapper. CSV, Alpaca and ShareGPT require conversion. |
| Transformers / TRL / PEFT | Familiar SFT/LoRA concepts; potentially reusable data and evaluation conventions | No trainer integration, PEFT adapter conversion or proven cross-runtime loading. A `.safetensors` suffix does not establish compatibility. |
| External inference applications / original CUDA EXL3 path | No round-trip integration demonstrated | Current adapters are private to this tool; no supported export, fusion, serving endpoint or application import. Own CLI generation works. |

MLX-LM already offers a coherent train/test/generate/fuse workflow for models its
backend supports. That is a useful convention to follow, not evidence our backend
is automatically supported. [MLX-LM LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

**Reproduced compatibility failures:** installed MLX-LM 0.32.0 rejects this model's
configuration because Engram layers 1 and 14 are unsupported. Its adapter loader,
called with an empty module to avoid any model allocation, rejects our actual
saved adapter configuration with missing `num_layers`, before reaching tensor
loading. It also expects different configuration fields and the plural filename
`adapters.safetensors`. Renaming alone would not map module paths, scaling,
orientations, or runtime behavior. Current upstream source still rejects Engram.
[MLX-LM model source](https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/deepseek_v41.py)

MLX-Tune provides an Unsloth-style API, but is a separate, unofficial project.
Its inspected `from_pretrained` delegates to `mlx_lm.load`, and the SFT path
calls MLX-LM training or its CLI. Integrating our engine requires an explicit
backend path. We did not install or run MLX-Tune here.
[Project](https://github.com/ARahim3/mlx-tune),
[loader](https://raw.githubusercontent.com/ARahim3/mlx-tune/main/mlx_tune/model.py),
[trainer](https://raw.githubusercontent.com/ARahim3/mlx-tune/main/mlx_tune/sft_trainer.py)

PEFT compatibility requires correct configuration and parameter mapping, including
LoRA scaling and matrix orientation, followed by numerical round-trip validation.
It is separate from whether a destination runtime can execute this base model.
[PEFT checkpoint documentation](https://huggingface.co/docs/peft/main/developer_guides/checkpoint)

## Ranked findings

### P0 — The advertised audience is wider than the supported model path

The full model constructor requires the specific DeepSeek V4.1 40-layer
configuration, MXFP8 trunk, and local expert/Engram streaming banks. It is not a
loader for arbitrary EXL3 repositories, the original 2.8-bpw layout, or Qwen.
The CLI has no complete original-checkpoint-to-supported-repack setup flow.
A newcomer with an EXL3 Hugging Face repository cannot follow a model-ID-to-training
journey. The existing `fetch-linear` only obtains a matrix for development.
Evidence: [constructor](../src/exl3_mlx/deepseek.py), lines 62–97;
[CLI](../src/exl3_mlx/cli.py), lines 127–137; [scope](../RECIPE.md).

**Release requirement:** a capabilities screen and model inventory that show exact
architecture/layout support before allocation, plus one reproducible acquisition
or import path for the supported layout. Do not label this “fine-tune any EXL3.”

### P0 — A user can train an artifact they cannot use in their usual application

Our private adapter works with our own slow generation CLI. There is no validated
serving integration or interoperable export. A successful training notification
therefore does not complete a typical user's goal of using a customized model.
The upstream adapter-loader rejection above makes this a concrete gap.
Evidence: [adapter format/load](../src/exl3_mlx/train.py), lines 89–142;
[recorded probes](usability-probes.json); [recipe](../RECIPE.md).

**Release requirement:** choose one actual inference runtime, implement exact
attention/expert adapter mapping and a fresh-process round trip there, and offer
“Try my model.” A local chat preview is a first delivery path, not a substitute
for claiming external compatibility. Keep base weights packed; do not make a
hundreds-of-gigabytes dequantize/merge operation the default export.

### P0 — Training completion is not evidence of useful model improvement

The verified codeword fit is causal but tiny. The fresh broader codeword question
failed even while target loss improved. Complete-model parity and representative
task quality remain open. No automatic final workflow presently distinguishes
“training completed,” “evaluation incomplete,” and “task improved.”
Evidence: [quality status](quality-status.json), [paired probe](quality-probe-paired.json).

**Release requirement:** independent numerical parity plus one realistic,
source-separated downstream task. Automatically show paired output scores,
regressions, truncation and sample counts. Keep a final test separate from the
validation set used to choose a checkpoint. Label inconclusive results honestly.

### P1 — Defaults do not compose, and the first-party help hides the main journey

A six-example customer-request fixture prepared successfully with default
`seq_length=128`. Loading that prepared training data with default training
`seq_length=16` failed: 41 target positions exceeded the limit. Its suggested
remedy is `--truncate`, even though preserving the entire answer is preferable.
Top-level help lists developer primitives as its actual subcommands and mentions
prepare/train/evaluate/generate only in an epilog. Train exposes 30 options,
many without help text or displayed defaults. Evidence:
[workflow defaults](../src/exl3_mlx/workflow.py), lines 39–80;
[CLI dispatch](../src/exl3_mlx/cli.py); [captured help and probes](usability-probes.json).

**Release requirement:** one run configuration governs preparation, training and
inference limits. A guided entry point shows supported models, data preview,
resource estimate, progress and results; technical tuning belongs in Advanced.
Never resolve a length mismatch by silently truncating answers.

### P1 — Common data imports and source-aware splitting are manual

The importer accepts strict raw JSONL schemas. Adding `source_id` to an otherwise
valid prompt/completion row causes rejection. Alpaca `instruction/input/output`,
ShareGPT `conversations`, and CSV are rejected. A two-example smoke dataset also
fails default preparation because nonempty train/validation/test splits are
required. These are all reproduced. Exact deduplication exists, but source-family
and near-duplicate separation require the user to build their own preprocessing.
Evidence: [data](../src/exl3_mlx/data.py), lines 77–116 and 148–207;
[probes](usability-probes.json).

**Release requirement:** explicit column mapping, previews of context and learned
answer, actionable row errors, and metadata-preserving group splits. Offer a
clearly labeled demonstration mode for tiny datasets; never silently reuse train
as validation. The inspected MLX-Tune SFT source has such a fallback, which is
one behavior we should not inherit when borrowing its API.
[Inspected trainer](https://raw.githubusercontent.com/ARahim3/mlx-tune/main/mlx_tune/sft_trainer.py)

### P1 — Low working memory does not establish practical speed or broad Mac support

The successful run measured roughly 76–81 seconds between logged updates while
supervising only three answer/EOS tokens each. Fresh CLI generation took 50.4
seconds including loading for two text tokens plus EOS. These are short-run
observations with checkpoint/setup overhead, not steady-state throughput or
larger-context forecasts. Four paired probes took 678 seconds and read about
580 billion expert bytes cumulatively; cached reads are not necessarily physical
SSD reads. The model's on-disk files total approximately 397 GiB.
Evidence: [fit](behavior-lowrate-passed.json), [generation](behavior-fresh-cli.json),
[paired probe](quality-probe-paired.json), [doctor accounting](../RECIPE.md).

Default startup requires 16 GiB process budget plus 32 GiB reclaimable reserve:
48 GiB headroom. These defaults cannot start training on a 16/32 GiB Mac. A
10.6 GiB observed short-run footprint on a 128 GiB M1 Ultra is not a tested
small-Mac configuration. The 512-token cap excludes long conversations/documents,
and even that maximum has not been established as a practical training size.
Evidence: [memory guard](../src/exl3_mlx/memory.py), lines 51–79;
[defaults](../src/exl3_mlx/workflow.py), lines 24–29.

**Release requirement:** check actual device/layout/disk/memory, then measure a
bounded pilot on representative user examples and report a time range before a
long run. Preserve the reserve; reducing a number does not make an unsupported
machine validated. Benchmark longer examples separately.

### P1 — Recovery and checkpoint choice require expert judgment

The implementation has strong atomic checkpoints and exact Adam/RNG resume, but
users must find a step directory, distinguish exact resume from warm start, keep
settings consistent, and choose a new run directory. `latest.txt` is written but
not accepted as a run-level resume abstraction. Retention preserves the latest
checkpoints, not the best validation checkpoint; an earlier better adapter can
be deleted. Validation is optional and the example configuration checks only two
validation rows. Evidence: [workflow](../src/exl3_mlx/workflow.py), lines 182–323;
[example configuration](../examples/sft.json).

**Release requirement:** one Resume action restores the saved contract, with a
separate deliberate “start a new experiment from these weights” action. Preserve
both latest and best validation checkpoints. Support graceful stop after a
completed optimizer step and show the last recoverable checkpoint after failure.

### P2 — The artifact is a reproducible research run, not a complete user project

Setup requires Python/uv and the correct repository subdirectory. Example paths
refer to this Mac; config paths resolve from the working directory. There is no
packaged novice launcher, model/data picker, stable run status API, or result page.
Users must interpret JSON reports and know which checks to invoke. Evidence:
[package](../pyproject.toml), [recipe](../RECIPE.md), [workflow](../src/exl3_mlx/workflow.py).
The zero-step config probe only checked parsing/early validation; this path
resolution finding is source-based, not an end-to-end runtime observation.

**Release requirement:** a project/run directory records resolved paths, dataset
splits, effective settings, metrics, adapters, and a readable before/after report.
Expose a stable API so a local interface and framework adapter share one engine.

## Useful capabilities to preserve

- Real packed EXL3 frozen-weight execution and activation gradients on Metal.
- Streamed expert/Engram access and block recomputation that enabled this large
  model's tiny fitting run in about 10.6 GiB observed process/MLX memory.
- Existing MLX-LM component reuse, checksum/nonfinite checks, explicit answer masks,
  atomic saves, exact optimizer/RNG continuation, and adversarial quality probes.
- Honest unsupported-layout errors and the absence of silent model conversion.

These justify continuing the backend. They do not justify recommending the
current command-line workflow to an unsupervised non-expert.

## Specific target: learning Metal inference optimizations from Git history

The [task assessment](metal-training-assessment.md) finds a stronger context
blocker for this intended use: 19 of 20 sampled optimization-related code changes
in the local engine exceed the current 512-token training limit; the median
minimal message-plus-diff case uses 5,319 training positions. Candidate commits
still need success verification and coherent case extraction. Raw log training
could fit messages while missing the actual optimization task. The report gives
a before/patch/rationale dataset design and an executed correctness/speed benchmark.

## Integration decision and next acceptance milestone

Keep the custom streamed execution backend and MLX-LM component reuse. Build a
small stable run API plus one guided local flow; expose familiar MLX-LM data and
config conventions where semantics match. MLX-Tune is a potential API integration
partner, not a drop-in engine replacement. Do not route host I/O and custom
recomputation through a compiled trainer without numerical and memory proof.

Prioritize a supported model import, one run-to-preview path, coherent defaults,
and automatic paired evaluation. Next add one verified serving-runtime adapter
integration. PEFT export and broader architecture support are separate milestones.
For ordinary models already supported by MLX-LM, its existing workflow is the
natural starting point; our distinctive value is the streamed EXL3 path.

The proposed [simple flow and acceptance checks](../SIMPLE_FLOW.md) makes the
product target concrete. It is a specification, not a claim that the interface
or integrations have been implemented. Before calling it non-expert ready,
observe at least three unfamiliar users complete setup → train → compare → use
and recover an interrupted run without editing Python or consulting an engineer.
That is a formative usability check, not a statistical quality certification.
