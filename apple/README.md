# EXL3 QLoRA on Apple silicon / MLX

An experimental port of [UnstableLlama/exl3-qlora](https://github.com/UnstableLlama/exl3-qlora)
with real EXL3 Metal decoding, activation gradients, streamed experts and a usable
DeepSeek V4.1 adapter workflow.

**Start with the [standard fine-tuning guide](SIMPLE_FLOW.md)** and run
`uv run exl3-mlx studio` for the local browser interface. The project CLI and
`SFTTrainer` Python API share CSV/JSON/chat import, safe resume and paired
comparisons. The [advanced recipe](RECIPE.md) covers engine controls.
Use `--profile development` for the explicit 80 GiB process budget and 8 GiB
packed expert cache, or select Development in Studio. Decoded caching stays off
after a slower measured MoE-layer experiment. It requires 112 GiB reclaimable
at startup including system reserve. Low-memory defaults remain available; see
[resource profiles and overrides](RESOURCE_PROFILES.md). Cache speedups require
measurement on the actual workload.
The tool reuses MLX-LM's data processors, loss and LoRA forward; see
[what is reused and why](SOURCES.md).

The [usefulness review](reports/usability-review.md) assesses beginner usability
and framework compatibility at the previous revision. The [standard workflow
review](reports/standard-sft-review.md) records the new implementation and fixes.
For learning Metal inference optimizations from Git
history, see the [task assessment](reports/metal-training-assessment.md) and its
previous context audit. See [kernel tuning](KERNEL_TUNING.md) for the ktune archive flow.

See [tool integration status](docs/TOOL_INTEGRATION.md) for which experimental
improvements are in the reusable package and which still need integration.
For cache-aware experiments, the read-only [cache inspector](docs/cache-inspection.md)
reports which complete project rows have matching cache metadata without loading
a model:

```sh
uv run python -m exl3_mlx.cache_inspect /absolute/path/to/project
```

Coverage is not worker compatibility or a verified hit; keep an existing frozen
worker unchanged and let its normal binding/checksum checks validate actual reuse.
The latest larger campaign completed **458 updates on 229 examples**; final
held-out evaluation is running. Its first/repeat training passes took20.64/1.55hours,
which demonstrates a throughput opportunity, not better generated kernels. See
[current progress](../docs/apple-m5-evaluation-progress.md) and the
[cache-first experiment plan](../docs/apple-cache-aware-experiment-plan.md).

For detailed teacher-forced comparisons, the [CPU evaluation auditor](docs/EVALUATION_AUDIT.md)
checks matching tokens, loss masks and reported metrics with `exl3-mlx-audit`.

## Kernel-tuning features and historical qualification

The [ktune flow](KERNEL_TUNING.md) imports complete archived edits, checks their
source/evidence identities, prepares standard SFT projects, and audits fitting
separately from kernel quality. Full-prefill attention now implements the model's
sparse index selection beyond 512 tokens, with a 32,768-token implementation
ceiling and explicit memory guards. The ceiling is not a tested operating limit.

A later final-block experiment fitted the complete 5,809-token Metal edit: NLL
fell from 0.5012578592 to 0.4735794807 over eight updates, with exact same-process
uncached final-logit parity. It trained one distinct change; most gain came from
comments/headers and added-code accuracy stayed 112/114. The separate BF16-mode
one-step experiment descended against its own baseline. A completed same-runtime
checkpoint-mode control regressed, but did not reproduce the historical update
exactly; a unique cause remains unproven. See the [current measured results](reports/metal-tuning-result.md).
The normal [three-change flow](reports/standard-three-change-run.md) completed
12 updates on three complete examples (3,606 / 5,809 / 5,822 tokens), four passes
per example. Fresh uncached NLL fell **0.35555322 → 0.32627658 (8.23%)**, and all
three rows improved. The saved adapter exactly reproduced the warm training
logits. Training took 30.14 minutes including the final probe and peaked at
53.08 GiB MLX. The [result and saved-adapter recipe](reports/metal-tuning-result.md)
include all category regressions: comments/headers account for 92.1% of the
loss decrease, and added-code top-1 improved by one teacher-forced token. All three share one lineage,
with no held-out quality claim. The first failed all40-layer trial below remains
preserved as historical evidence.

The new [upstream-history flow](docs/UPSTREAM_INTAKE.md) adds `ktune prepare-upstream`,
`finetune-upstream`, and the separate `compile-metal` gate. Structured examples
include baseline source, source-grounded task/hardware/bottleneck fields, rationale,
complete optimized source, and explicit verification status. Source captures retain complete originals; exact-edit targets replay every
retained non-documentation file change, while scoped prompts disclose omitted
context. The historical
[100-commit audit](reports/upstream-corpus-inventory.md) reconstructed 93 changes,
but only five whole-file examples fit; they include non-kernel changes. A reviewed
llama.cpp example fits 3,394 tokens and has passed actual standard-project
preparation with bound provenance. Scoped GPU parity tests also pass for that
convolution pair (16 cases plus a CPU oracle) and TensorFold's EXLR8 projection
(162 cases and all 65,536 codebook states). These are source-kernel parity
checks; the convolution also has an independent CPU oracle. Neither establishes
speed or model-generation quality. The integrated revision passed
1,040 tests plus 27 subtests. These new records have not been trained or proven
faster. See the
[one-command recipe](KERNEL_TUNING.md#structured-upstream-history-flow).

The earlier all40-layer **5,809-token historical kernel-edit example** completed one update
in **31.0 minutes**, peaking at **73.95 GiB MLX memory**, under an 80 GiB process
cap and explicit 12 GiB reserve. The checkpoint contains 3,153,920 finite adapter
parameters; all 80 LoRA B tensors changed from zero, and optimizer/checksum checks
passed. **Fresh fitting failed:** answer NLL increased from **0.5012578592 to
0.5053724668**, despite changed logits. The original trial is preserved. The
+0.1, -0.1 and adaptive +10 inference-only strength probes also worsened NLL.
The bounded series is complete, with no adapter promoted as an improvement.
This historical operational pass does not establish improved fitting,
32,768-token feasibility or kernel quality. See the [checkpoint audit](reports/ktune-full-parent-checkpoint-audit.json),
[failed fit receipt](reports/ktune-full-parent-first-update-fit.json),
[complete strength diagnostics](reports/ktune-strength-diagnostics.json), and
[exact pilot settings](KERNEL_TUNING.md#command-line-recipe).

That revision passed **518 tests plus 27 subtests**, and the new native
projection/head and sparse-attention VJP checks pass within their stated scope.
These component passes did not resolve that first-step failure. The latest
[pre-launch test receipt](reports/loss-fix-v3-pytest.xml) belongs to the standard
three-change runtime. See the
[diagnosis update](reports/first-step-diagnosis-update.md) and [JUnit receipt](reports/kernel-sft-pytest.xml).

This extension also fixes negative FP8 scale exponents in the earlier engine.
That changes forward arithmetic even on short inputs. Old adapters and numerical
receipts below belong to their recorded source revision; exact optimizer resume
across this arithmetic change is rejected. Independent full-model parity and
new-kernel speedups remain unverified. See the [long-context design review](reports/long-context-design-review.md)
and [ktune adversarial review](reports/ktune-adversarial-review.md).

## Previous revision's operational evidence

The following results were recorded before the scale fix, including standard
workflow revision `4352b83e`. They establish what those implementations did and
do not validate the changed engine's numerical behavior.

The new standard flow completed **two full-model updates plus a Studio resume to
step 3**, then comparison and adapter export. Observed memory peaked at **10.04
GiB** (8.61 GiB MLX). Answer loss on one synthetic held-out row fell from **9.34 to
3.70**, but both four-token previews remained `I can help,` and neither reached
EOS. This validates the workflow and demonstrates why likelihood improvement
alone must not be called answer quality. [Full evidence](reports/standard-sft-verification.json).
Actual-tokenizer preparation also passed for Alpaca, ShareGPT and multi-turn
system chat through the Python API: [format report](reports/standard-sft-api-formats.json).


A controlled language-model fine-tune now changes `Code?` from the four-token
base prefix `I'd be happy` to **`violet` plus EOS**. The successful adapter uses
one initial update plus four lower-rate updates, with exact adapter-off/reload
checks and fresh-process public-CLI confirmation. It peaks at 9.16 GiB MLX /
10.61 GiB observed process memory. An earlier high-rate trial failed and is
preserved. This demonstrates causal fitting, not broad task quality. See the
[quality gates and adversarial reviews](QUALITY.md),
[successful run](reports/behavior-lowrate-passed.json) and
[failed trial](reports/behavior-highrate-failed.json).

A real 40-layer model completed an Adam update with **18,681,856 trainable adapter
parameters**, including attention adapters in all layers and EXL3 expert adapters
in layer 0. Gradients and saved adapters were finite; 60 expert B tensors changed.
The system-reserve watchdog interrupted the next step when another workload
consumed memory. The complete step checkpoint survived, resumed to optimizer
step 2 in a new process, and passed **exact full-model prediction reload**.

The resumed run peaked at **8.78 GiB of MLX allocations** and **10.26 GiB observed
process/MLX memory**, under a 12 GiB process guard and 32 GiB system reserve.
These are four-target-token operational checks on an M1 Ultra with 128 GiB RAM,
not a throughput, quality or larger-context memory benchmark. Reports:
[initial update](reports/workflow-full-model-update.json),
[successful recovery and reload](reports/workflow-resumed.json), and
[complete workflow](reports/workflow-verification.json). The repeatable workflow
also verified **bit-exact adapter and Adam state versus uninterrupted training**,
base/adapter evaluation, and greedy generation. The four-token training fixture
loss fell from 5.074 to 4.983 after two updates; this is not held-out quality.
That revision's test suite passed **263 tests and 27 subtests**.

**Supported full-model checkpoint:** the local **3.0-bpw mtplx streaming repack**,
with native MXFP8/BF16 trunk and Engram banks. The requested
[2.8-bpw original checkpoint](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw)
is a different layout: its real matrices and a complete expert are verified, but
full-model loading of that original layout is not implemented. Qwen is not yet
supported. No existing model was deleted.

Independent native full-model logit parity and useful trained text quality remain
unverified. Activation FP8/FP4 rounding uses an explicit identity straight-through
gradient. This is experimental code, not a production quality claim.

## Setup and commands

```bash
cd ~/experiments/exl3-qlora/apple
uv sync --locked
uv run pytest
uv run exl3-mlx doctor \
  --model ~/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw \
  --memory-gib 2 --verify-records
```

| Command | Purpose |
| --- | --- |
| `exl3-mlx studio` | Guided local browser: check data, train/resume, compare, preview and export |
| `exl3-mlx finetune` | One project from CSV/JSON/JSONL; optional `--prepare-only` |
| `exl3-mlx ktune snapshot / prepare / finetune` | Read ktune archives, preserve complete edits and train a standard project |
| `exl3-mlx ktune prepare-upstream / finetune-upstream` | Structured Git history intake and a standard project with bound provenance |
| `exl3-mlx ktune compile-metal` | Compile a reviewed standalone before/after Metal source pair; no runtime claim |
| `exl3-mlx ktune verify / audit-evidence / export-edit` | Check training fit, audit receipts, and convert patches to ktune edits |
| `exl3-mlx resume / compare / chat / export-adapter` | Continue or use a saved standard project |
| `exl3-mlx status / stop` | Check progress or checkpoint after the current update |
| `exl3-mlx prepare` | Local JSONL text/chat tokenization, answer masks, deduplication and splits |
| `exl3-mlx train` | Streamed, eager adapter training with Adam, accumulation and validation |
| `exl3-mlx train --resume STEP` | Restore optimizer, RNG and sample cursor |
| `exl3-mlx evaluate` | Held-out token-weighted NLL/perplexity, base or adapter |
| `exl3-mlx generate` | Greedy generation using the saved adapter; recomputes the full prefix per token |
| `exl3-mlx doctor` | Header, bank-size, token-map and optional sampled-record checks |
| `exl3-mlx inspect / validate-linear / fetch-linear` | Inspect and validate original EXL3 matrices |
| `exl3-mlx train-smoke` | Small synthetic matrix test; not language-model training |

Every command accepts `--help`. `exl3-mlx-train` remains an alias for the full
trainer. [examples/sft.json](examples/sft.json) supplies editable JSON defaults;
CLI flags override them. Choose new directories for each run. See the recipe for
commands, input schemas and the exact continuation versus warm-start distinction.

For the repeatable, substantial real-model workflow check:

```bash
uv run python scripts/verify_workflow.py \
  --model ~/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw \
  --output artifacts/workflow-check
```

This sequentially checks two updates, exact optimizer continuation, adapter reload,
base/adapter evaluation and generation. All child processes retain the usual
memory guards; it downloads nothing. If an evaluation/generation phase is blocked
by memory pressure, rerun with `--continue-run --wait-for-memory 120` to reuse
completed phases. An interrupted training phase needs the trainer's `--resume`
or a new verification directory. Individual primitive checks are much smaller:

```bash
uv run exl3-mlx train-smoke --output artifacts/small-adapter
uv run python scripts/validate_target.py
uv run python scripts/validate_expert.py
```

The target validators fetch only four pinned matrices (under 12 MiB payload),
enforce byte limits, and record hashes. Weights and generated adapters stay ignored.

## How memory stays bounded

Selected EXL3 matrices are read through a bounded packed-weight LRU, and Engram
reads fetch only needed rows. The full expert/Engram banks never become resident.
The roughly 8 GiB frozen text trunk stays in RAM. Block recomputation rebuilds
one block's backward graph at a time and preserves gradients through shared KV.
This is end-to-end training, not independent layer-by-layer fitting.

Python MLX custom VJPs with `eval`/`stop_gradient` alone retained dense decode
graphs. The current implementation materializes activation boundaries through
small host buffers, preserving BF16 bytes, to release those graphs. Decode and
matrix math run on Metal. Synchronization and SSD traffic currently cost speed;
see [remaining porting work](PORTING.md).

Training defaults to a 16 GiB fail-fast process budget, 64 MiB allocator cache,
128 MiB packed-weight cache and 32 GiB system reserve. A 100 ms watchdog checks
MLX/RSS use; system headroom is checked every five seconds. MLX's allocator limit
is advisory: this is **not an OS hard quota**, and one allocation may overshoot.
Completed checkpoint directories are published atomically and remain recoverable.
The last two checkpoints created in each run are retained by default.

The implementation accepts up to 32,768 tokens with learned sparse index selection,
candidate-block filtering and query-tiled attention. Complete examples are never
silently truncated. This ceiling is not a claim that the longest run fits the
default budget or is fast; use the measured lengths and memory in the kernel-tuning
verification report. The frozen vocabulary head projects only supervised rows in
32-token chunks during training/evaluation. Text only, first-order eager reverse-mode gradients;
no vision, MTP, higher derivatives or `mx.compile`. Generation recomputes the
whole prefix and has no KV decode cache. Adapters use a private format loaded by
this tool. Unmodified external serving, PEFT compatibility and fusion are not
qualified. An isolated `mlx-serve` / `mlx-stream` adapter integration is under
development; see [serving integration](docs/SERVING_INTEGRATION.md) for its scope.

## EXL3 and architecture verification

The Metal decoder implements tail-biting ring windows, tensor-core fragment
permutation, all three procedural codebooks, integer 1–8 bpw, and mul1
1.5/2.5/3.5 bpw. The checkpoint's 2.8 bpw is an aggregate of matrix bitrates.
The base algebra is:

```text
W_effective = diag(suh) H128 W_inner H128 diag(svh)
y_base      = H128(H128(x * suh) @ W_inner) * svh
dx_base     = H128(H128(dy * svh) @ W_inner.T) * suh
y           = y_base + alpha/rank * (x @ A) @ B + bias
```

H128 is a normalized transform per 128-element block. Codebooks round to fp16;
transforms and accumulation use fp32, then cast to the activation dtype. This is
not bitwise parity with CUDA fp16/bf16 inference. NumPy is used for reference
checks, I/O, route planning and activation materialization, not weight computation.

Tests cover exhaustive codebooks, layout, gradients, frozen bases, selected-expert
routing, SwiGLU, hyper-connections, sparse-attention masks/sinks, quantization
rounding and its explicit identity straight-through gradient, shared-KV block
recomputation, checkpoint integrity, memory retention, and adapter serialization.
Source math follows the pinned official DeepSeek V4.1 inference implementation.

| Real 2.8-bpw fixture | Bits | Max forward error | Max dX error |
| --- | --- | --- | --- |
| `layers.0.attn.wkv` | 5 | 1.86e-6 | 1.37e-6 |
| `layers.0.ffn.experts.0.w1` | 2 | 1.78e-6 | 2.06e-6 |
| `layers.0.ffn.experts.0.w2` | 3 | 1.11e-6 | 2.75e-6 |
| Complete expert `w1/w2/w3` | 2 / 3 / 2 | 2.42e-6 | 4.31e-6 |

These compare against independent float64 NumPy math. The complete expert test
includes routing weights before the down projection and the exact SwiGLU clamps.
A separate 60-step matrix adapter fit decreased synthetic loss from 0.0696842 to
0.00278163, kept base weights unchanged, and reloaded exactly. Evidence lives in
[`reports/`](reports/), including the full-model probe and its measurement scope.

## Sources and license

Upstream baseline: `068f586c00724e3eabbf4165fb66c355da9af4df`. Format references:
[`tests/testlib/trellis.py`](../tests/testlib/trellis.py),
[`quant/codebook.cuh`](../exllamav3/exllamav3_ext/quant/codebook.cuh),
[`LinearEXL3`](../exllamav3/modules/quant/exl3.py), and
[`EXL3LoRAHadFunction`](../exllamav3/training/qlora_linear.py).
Architecture source:
[official DeepSeek V4.1 model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py).
The port and adapted math retain MIT licenses; see [`LICENSE`](LICENSE) and
[`configs/DEEPSEEK_LICENSE`](configs/DEEPSEEK_LICENSE). Model weights stay untracked.

### Development snapshot: M5 migration and research

See [RESOURCE_PROFILES.md](RESOURCE_PROFILES.md#m5-training-host-qualification-2026-10-13)
for the M5 numerical mode, full checkpoint verification and fresh-cache launch
requirements, and [KERNEL_TUNING.md](KERNEL_TUNING.md#expanded-library-and-next-experiment-2026-10-13)
for the expanded corpus and the next experiment's evidence limits.

The repository's `qwen-next/` directory contains isolated storage/decoder and
architecture reference work, synthetic tests and historical review receipts.
It is **not** a supported Qwen full-model trainer. Some source audits deliberately
bind the original local research checkout; configure those paths before reproduction.
The existing Qwen design document remains the staged implementation contract.

`kernel-library/` archives the collection, curation, packaging and audit scripts.
These research scripts expect the separately retained experiment evidence tree and
pinned runtime snapshots; they are not installed CLI entry points or a self-contained
published dataset. Large source captures, corpora, model weights and training outputs
are intentionally kept outside Git. Historical receipts distinguish observed results
from proposals and record the original evidence paths.

Reviewed portable bundles can use [prepare-reviewed](docs/REVIEWED_CORPORA.md) for strict edit replay, bounded diagnostic evidence bindings, frozen splits and ordinary SFT preparation. Existing mixed corpora require explicit version-2 evidence migration and review.

Evaluation and training-fit checks use [captured report snapshots](docs/EVALUATION_RECEIPTS.md) to validate the same bytes that are bound in their published results. These local consistency checks do not certify process authenticity or model quality.
