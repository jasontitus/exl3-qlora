# EXL3 QLoRA on Apple silicon / MLX

An experimental port of [UnstableLlama/exl3-qlora](https://github.com/UnstableLlama/exl3-qlora)
with real EXL3 Metal decoding, activation gradients, streamed experts and a usable
short-context DeepSeek V4.1 adapter workflow.

**Start with the [complete recipe](RECIPE.md).** It covers setup, data/masks,
training, memory limits, optimizer resume, held-out evaluation and generation.
The tool reuses MLX-LM's data processors, loss and LoRA forward; see
[what is reused and why](SOURCES.md).

## Verified status

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
The small test suite passes **111 tests**.

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
| `exl3-mlx prepare` | Local JSONL text/chat tokenization, answer masks, deduplication and splits |
| `exl3-mlx train` | Streamed, eager adapter training with Adam, accumulation and validation |
| `exl3-mlx train --resume STEP` | Restore optimizer, RNG and sample cursor |
| `exl3-mlx evaluate` | Held-out token-weighted NLL/perplexity, base or adapter |
| `exl3-mlx generate` | Greedy short-prefix generation using the saved adapter |
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

Maximum context is 512 tokens because longer-context sparse candidate/indexer
selection is not ported. This is not a claim that 512-token training fits the
default budget or is fast. Text only, first-order eager reverse-mode gradients;
no vision, MTP, higher derivatives or `mx.compile`. Generation recomputes the
whole prefix and has no KV decode cache. Adapters use a private format loaded by
this tool; external serving/PEFT mapping and fusion are not implemented.

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
