# EXL3 QLoRA on Apple silicon — initial MLX port

This directory contains working **EXL3 matrix and LoRA training primitives**, tested
on an Apple M1 Ultra (128 GiB unified memory), macOS 27.0.1, Python 3.12.13 and
MLX 0.32.3. It does **not yet load or fine-tune a complete language model**.
The CUDA/PyTorch package remains separate; this package never imports it.

Target: [jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw),
pinned at `915e1d0de8ee0a99944c5bc88ab802a639c63380`. The checkpoint's 2.8 bpw is an
aggregate: each matrix has its own integer bitrate. A 2.8-bit uniform decoder would
be incorrect. Upstream port baseline: `068f586c00724e3eabbf4165fb66c355da9af4df`.

## Run

```bash
cd ~/experiments/exl3-qlora/apple
uv sync --locked
uv run pytest

# Small synthetic adapter fit; choose a new output directory on each run.
uv run exl3-mlx train-smoke --output artifacts/demo

# Real target matrices: downloads ~8.63 MiB of tensor payload in total,
# plus small safetensors headers; never downloads complete model shards.
uv run python scripts/validate_target.py

# Fit a synthetic residual over the actual DeepSeek attention matrix.
uv run exl3-mlx train-smoke \
  --checkpoint artifacts/deepseek-v41-wkv.safetensors \
  --key layers.0.attn.wkv --output artifacts/deepseek-demo
```

`train-smoke` exercises adapter optimization and saves `adapter.safetensors`,
`adapter_config.json`, and a loss report. Only A/B are saved; reload is checked for
exact equality. This is a synthetic regression task, not text SFT. The artifact
format is `exl3-mlx-linear-v1`, with A `[in, rank]` and B `[rank, out]`; it is not
advertised as a full-model PEFT/native adapter.

The CLI can inspect local shards without loading their payloads:

```bash
uv run exl3-mlx inspect /path/to/exl3-checkpoint
uv run exl3-mlx validate-linear /path/to/exl3-checkpoint \
  --key layers.0.attn.wkv --report reports/my-matrix.json
```

`fetch-linear --help` exposes the bounded HTTP range downloader. It requires an
immutable Hub commit, checks range responses, enforces a payload budget, and saves
per-tensor hashes. Large model weights, environments, and training artifacts are
gitignored. Configuration, provenance, small reports, and `uv.lock` are tracked.

## Implemented

- Metal JIT decoding of the actual EXL3 tail-biting bitstream and CUDA tile order.
- Integer 1–8 bpw, plus 1.5/2.5/3.5 bpw with `mul1`; `3inst`, `mcg`, and `mul1`
  procedural codebooks, with their fp16 rounding semantics.
- Frozen base linear with normalized block-128 Hadamard transforms and `suh/svh`.
  Old packed `su/sv` sign vectors and optional biases load too.
- A custom MLX VJP that decodes again for `dX`; MLX differentiates the LoRA path.
  Leading batch/sequence axes, vectors, strided inputs, and fp32/fp16/bf16 inputs
  are covered. Only first-order reverse-mode differentiation is supported.
- Per-matrix safetensors loading, including scales split across shards, marker
  validation, and clear rejection of unsupported storage layouts.

The base computes

```text
W_effective = diag(suh) H128 W_inner H128 diag(svh)
y_base      = H128(H128(x * suh) @ W_inner) * svh
dx_base     = H128(H128(dy * svh) @ W_inner.T) * suh
y           = y_base + alpha/rank * (x @ A) @ B + bias
```

`H128` acts independently on every 128-element block. Transforms, matmuls and dX
use fp32; codebook values first round to fp16, and the output casts to the input
dtype. LoRA parameters are fp32. This is a correctness baseline, not a claim of
bitwise parity with upstream's fp16/bf16 inference arithmetic.

Metal decoding feeds MLX's optimized matmul. This version does **not** implement
a fused trellis GEMM. `chunk_size` limits each reconstructed column allocation;
MLX's lazy scheduler can keep several allocations live, so it is **not a strict
peak-memory bound**. No dense base weight is stored in the layer or retained as an
autograd residual; both passes reconstruct from the packed base. Full-model memory
and throughput must be measured after streaming is implemented.

## Verified results

The offline suite has 69 passing tests: all 65,536 inputs of each codebook are
bit-exact against an independent NumPy oracle; ring wrapping, tile ordering,
bitrate/slice handling, dense-reference and finite-difference gradients,
checkpointing through stacked layers, freezing, training and adapter reload are
covered. The oracle is adapted from the upstream repository's test reference,
not from the new Metal implementation. No CUDA hardware was used for validation.

Real tensors extracted from the target checkpoint:

| Tensor | Shape `[in, out]` | Bits | Decode | Max forward error | Max dX error |
| --- | --- | --- | --- | --- | --- |
| `layers.0.attn.wkv` | 5120 × 512 | 5 | bit-exact | 1.86e-6 | 1.37e-6 |
| `layers.0.ffn.experts.0.w1` | 5120 × 2304 | 2 | bit-exact | 1.78e-6 | 2.06e-6 |
| `layers.0.ffn.experts.0.w2` | 2304 × 5120 | 3 | bit-exact | 1.11e-6 | 2.75e-6 |

Errors compare fp32 Metal/MLX against fp64 NumPy on three random input/cotangent
rows. Reports include synchronized warm timings, which are matrix-level checks,
not model training throughput. A 60-step synthetic fit over `wkv` reduced loss
from 0.0696842 to 0.00278163, kept the entire base byte-identical, and reloaded the
saved adapter exactly. Evidence is in [`reports/`](reports/).

## DeepSeek v4.1 feasibility and next steps

The target repository is approximately **203.8 GB / 189.8 GiB**, before the
separate Engram tables (the model card describes two tables of about 101 GB each).
This exceeds this machine's 128 GiB RAM and approximately 107.52 GiB recommended
Metal working set. Unified memory removes a CPU/GPU copy boundary; it does not
make an oversized model resident. No full checkpoint download or full-model
training was attempted. See [`PORTING.md`](PORTING.md) for the remaining work.

The model card also specifies a TensorFold-specific serving layout and says stock
ExLlamaV3 cannot load this pack. The upstream training repo's DeepSeek V4 inference
code is not a DeepSeek V4.1 training implementation. The new loader can read its
individual standard trellis matrices; that does not establish model support.

## Differences from the proposed concept

The initial script used an affine four-bit toy layout; EXL3 instead needs the
ring windows, procedural codebooks, fragment permutation and scale/Hadamard
sandwich implemented here. It also compiled a metallib that its NumPy path never
called. This implementation actually runs custom Metal through
[`mx.fast.metal_kernel`](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html),
with the current `(primals, cotangent, output)` VJP API and explicit parameter
freezing. NumPy appears only in file I/O, fixture generation and validation, never
in the base forward/backward. A starts random and B starts zero; therefore a zero
A gradient on the first step is expected and is tested.

## Sources and license

This port retains the repository's [MIT license](../LICENSE). Format and math:
[`tests/testlib/trellis.py`](../tests/testlib/trellis.py),
[`quant/codebook.cuh`](../exllamav3/exllamav3_ext/quant/codebook.cuh),
[`LinearEXL3`](../exllamav3/modules/quant/exl3.py), and
[`EXL3LoRAHadFunction`](../exllamav3/training/qlora_linear.py).
The downloaded target configuration and tensors are from the model's MIT-licensed
repository; its weights remain untracked and their provenance is recorded.
