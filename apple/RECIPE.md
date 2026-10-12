# Fine-tuning EXL3 on Apple silicon

For the browser interface, CSV/Alpaca/ShareGPT import and project-based Python API,
start with the [standard SFT guide](SIMPLE_FLOW.md). This page documents advanced
engine controls. Project `resume --steps` means total updates; engine `train
--steps` below means additional updates.

This is an **experimental text adapter pipeline** for the
DeepSeek V4.1 **streaming repack** already on this Mac. It trains against frozen
EXL3 weights on Metal, saves adapters, resumes Adam, evaluates held-out text, and
uses the adapter for generation. It does not convert EXL3 into ordinary affine
4-bit weights. Full-model numerical parity and useful task quality still require
independent validation.

Supported full-model input: `mtplx-expert-manifest-v2`, native MXFP8/BF16 trunk,
both Engram banks and their token map. The available checkpoint is **3.0 bpw**.
The requested **2.8-bpw original Hugging Face layout is not yet a full-model input**;
its real individual matrices and complete expert have passed decode/gradient
checks. Qwen is not an implemented architecture. Do not point these training
commands at an arbitrary EXL3 directory and expect architecture conversion.

The earlier revision's [measured workflow report](reports/workflow-verification.json) records two
full-model updates, exact adapter/Adam continuation versus an uninterrupted run,
exact adapter prediction reload, evaluation and one-token generation. Peak MLX
allocation was **8.78 GiB** for four target tokens at rank 2, with attention adapters
in all layers and EXL3 expert adapters in layer 0. These tests kept the process
budget at 12 GiB and the system reserve at 32 GiB. That revision passed 263 tests
and 27 subtests. These receipts precede the FP8 scale and long-context changes;
they do not validate the current engine's numerical behavior.

## 1. Install and inspect

```bash
cd ~/experiments/exl3-qlora/apple
uv sync --locked
export EXL3_MODEL="$HOME/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"
uv run exl3-mlx doctor --model "$EXL3_MODEL" --memory-gib 2 --verify-records
uv run pytest
```

Tested environment: M1 Ultra, 128 GiB unified RAM, Python 3.12, MLX 0.32.3 and
MLX-LM 0.32.0. `uv sync` installs Python dependencies; kernels compile through
MLX's Metal API at runtime. No CUDA or separately compiled `.metallib` is required.
The CUDA package at the repository root is separate: use `apple/` as above.

`doctor` reads headers, validates bank sizes and token-map hashes, and optionally
hashes two expert records. It does not load the model or hash all 397 GiB of model
files. Its 2 GiB budget is for inspection only. Training defaults to 16 GiB plus a
32 GiB system reserve. A checkpoint contains roughly 190 GiB of EXL3 experts,
189 GiB of Engram tables, and the trunk; disk capacity is distinct from working RAM.
No full-model download or deletion is part of this recipe.

For an explicit 80 GiB process budget with an 8 GiB packed expert cache, add
`--profile development`. It preserves 32 GiB system reserve and requires 112 GiB
free/reclaimable at startup. Defaults remain unchanged. See
[resource profiles](RESOURCE_PROFILES.md) for cache overrides, project resume,
and the distinction between process memory and retained cache sizes.

## 2. Verify one real update first

```bash
uv run exl3-mlx train \
  --model "$EXL3_MODEL" --data examples/smoke-tokens.jsonl \
  --output artifacts/first-update \
  --steps 1 --seq-length 4 --rank 2 --alpha 4 \
  --verify-reload
```

This four-target-token fixture is an operational check, not a useful fine-tune.
`--verify-reload` evaluates, zeros the adapters, reloads the saved adapter, and
requires exactly matching predictions. Each output directory must be new.
The old `exl3-mlx-train` entry point remains an alias for `exl3-mlx train`.

By default `wq_a` and `wo_b` attention projections in **all 40 layers** get adapters.
Frozen EXL3 experts participate in both forward and activation-gradient passes.
To train adapters directly on the EXL3 expert projections too, add
`--expert-adapter-layers 0`. This adds A/B matrices for all 384 experts' three
projections in that layer; only selected routes receive nonzero gradients.
Use `--adapter-layers 0 1` to restrict attention adapters, or an empty
`--adapter-layers` with nonempty expert layers for expert-only adapters.

## Demonstrate a deliberate answer change

The recorded codeword run teaches `Code?` and `Code:` to produce `violet` plus EOS.
The selected adapter inherits one update at 0.003, then four at 0.0001 with a fresh
Adam optimizer. An eight-update high-rate trial was unstable and is retained as
failed evidence. The low-rate run changed generation, reduced `Code?` answer NLL
from 10.0753 to 0.05798, and passed exact adapter-off/reload and fresh-CLI checks.
See [the report](reports/behavior-lowrate-passed.json) and [quality gates](QUALITY.md).

To construct the same two-stage setup using new output directories:

```bash
uv run exl3-mlx train \
  --model "$EXL3_MODEL" --data examples/codeword-train.jsonl \
  --output artifacts/codeword-initial --steps 1 --seq-length 16 \
  --rank 4 --alpha 8 --expert-adapter-layers 0 --seed 8 \
  --learning-rate 0.003 --save-every 1
uv run python scripts/verify_behavior.py \
  --model "$EXL3_MODEL" --output artifacts/codeword-behavior \
  --resume-adapter artifacts/codeword-initial/step-000001 \
  --learning-rate 0.0001 --max-steps 16
```

The verifier records fresh base outputs, trains only the two fixed prompts, checks
for the exact target and EOS after every even step from four onward, then checks
ablation and reload. It returns a nonzero exit if the causal checks fail. The
optional `--baseline-report` reuses a completed earlier demo and records its hash;
its adapter-off check revalidates the trained-prompt baseline. Prefer fresh base
measurements when code, model, template or environment changes.

This tiny dataset is intentionally a fitting demonstration. Run the paired
contrastive probes in [QUALITY.md](QUALITY.md) to detect unwanted answer collapse;
use a real task and independent test split before claiming useful quality.

## 3. Prepare your data

Use local JSONL, one independent document or conversation per row:

```json
{"text":"A complete training document."}
{"prompt":"What should this tool return?","completion":"The expected answer."}
{"messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

Do not mix unrelated datasets just because these schemas can share a file.
Text records receive BOS/EOS and supervise every target token. Chat and
prompt/completion records use the checkpoint's chat template and, by default,
supervise **only the final assistant answer**, including its end marker. Earlier
turns provide context. `--all-tokens` also supervises the prompt. Tool calls,
images and structured multimodal message content are rejected.

```bash
uv run exl3-mlx prepare \
  --model "$EXL3_MODEL" --data /absolute/path/to/my-data.jsonl \
  --output artifacts/my-data --seq-length 32 --seed 8
```

Try `examples/chat.jsonl` to test the format; its six toy questions are not a
meaningful task dataset. Preparation tokenizes with MLX-LM's data processors,
removes identical token sequences **before** deterministic 80/10/10 splitting,
and writes `train.jsonl`, `valid.jsonl`, `test.jsonl`, and a tokenizer/checksum
manifest. Tiny splits reserve at least one validation and test row.

Long examples fail instead of silently losing their answers. Prefer shortening
examples or selecting a larger context after measuring memory. `--truncate`
explicitly allows prefix truncation; it still rejects a row if all supervised
answer tokens disappear. Examples never share attention or Engram history.
There is no packing or sliding-window continuation.

For precise control, input records may instead contain
`{"input_ids":[0,2,3,4,1],"loss_start":3}`. `loss_start` is the index of the first
supervised token in `input_ids`, not the position in shifted logits; here tokens
`4,1` are supervised. It defaults to 1. Prepared data records exactly this mask.

Inspect the prepared masks before a longer run:

```bash
uv run python - <<'PY'
import json, os
from exl3_mlx.data import load_tokenizer
from pathlib import Path

t = load_tokenizer(Path(os.environ['EXL3_MODEL']))
r = json.loads(Path('artifacts/my-data/train.jsonl').read_text().splitlines()[0])
print('Context:', t.decode(r['input_ids'][:r['loss_start']]))
print('Supervised:', t.decode(r['input_ids'][r['loss_start']:]))
PY
```

Deduplication removes exact token duplicates, not semantic or document-family
leakage. For serious evaluation, split by source/customer/document family yourself
and prepare each split with `--valid-fraction 0 --test-fraction 0`, then pass
explicit `--data` and `--valid-data` files to training. Keep an independent test
file for final evaluation.

## 4. Train

`examples/sft.json` is a conservative starting configuration. Paths in the JSON
are relative to the working directory; `~` expands. CLI options override the JSON.
Its 32-token setting is a next experiment, not a claim that it has been benchmarked.

```bash
uv run exl3-mlx train --config examples/sft.json
```

This performs 20 Adam updates, rank 4, alpha 8, learning rate 1e-4, gradient clipping
at norm 1, and four sequential microbatches per update. Accumulation weights each
microbatch by its supervised-token count. It does not keep four models or four
forward graphs in RAM. Data is shuffled reproducibly each epoch. Validation runs
on two held-out examples every five steps and at the end; use a larger validation
sample when affordable. Watch held-out loss, not only training loss.

The first completed update, every `save_every` step, and the final update are
saved atomically. `latest.txt` names the latest complete step. With `keep_last: 2`,
only the last two checkpoints **created in this run** are retained. Each contains:

- `adapter.safetensors` and `adapter_config.json` with base identity and layer selection;
- `optimizer.safetensors` with Adam moments/step and `random.safetensors`;
- `trainer_state.json` with dataset/settings identity, sample cursor and file hashes.

The run also writes `run_config.json`, `history.json`, and `training.json`.
A memory abort can lose work since the latest completed save; it cannot expose a
partially written step as complete. `--save-every 1` minimizes lost updates but
adds disk writes. Adapter weights are FP32; two Adam moments roughly triple the
checkpoint tensor bytes. A rank-2 run including expert layer 0 has about 18.7M
adapter parameters and roughly 214 MiB of adapter-plus-Adam tensors per checkpoint.

## 5. Resume or warm start

```bash
uv run exl3-mlx train --config examples/sft.json \
  --resume artifacts/my-run/step-000020 \
  --output artifacts/my-run-resumed --steps 20
```

`--steps` means **additional** updates. Resume restores the optimizer, RNG and
sample cursor. Keep the same training data, sequence/truncation settings, seed,
learning rate, accumulation, clipping, chunk size, and recomputation mode; a
mismatch fails before an update. Rank/alpha/layer selections come from the saved
adapter unless explicitly supplied, in which case they must match. A saved step
number advances to 21 in this example. Always choose a new output directory.

For a different dataset or learning rate, use `--resume-adapter STEP_DIRECTORY`
instead. This intentionally starts a fresh optimizer, shuffling schedule and step
counter. It is not an exact continuation.

## 6. Evaluate and generate

Compare the base and adapter on the same held-out test file:

```bash
uv run exl3-mlx evaluate --model "$EXL3_MODEL" \
  --data artifacts/my-data/test.jsonl --seq-length 32 \
  --report artifacts/base-test.json
uv run exl3-mlx evaluate --model "$EXL3_MODEL" \
  --adapter artifacts/my-run/step-000020 \
  --data artifacts/my-data/test.jsonl --seq-length 32 \
  --report artifacts/adapter-test.json
uv run exl3-mlx generate --model "$EXL3_MODEL" \
  --adapter artifacts/my-run/step-000020 \
  --prompt 'Your representative test question' --max-tokens 16
```

Evaluation reports token-weighted NLL, perplexity and a logits hash. It uses the
same masks as training. Generation is greedy and **recomputes the full prefix**
for every token; there is no KV decode cache yet. Use `--raw-prompt` for plain text
continuation. The tokenizer's local chat template controls reasoning delimiters.
Prompt plus generated text must stay within 32,768 tokens. Assess factual accuracy,
style and regressions separately: lower perplexity on a toy dataset is not quality
validation. Report paths must be new.

Adapters use this package's private `exl3-mlx-deepseek-short-text-v1` format. Load
them with these commands; do not assume PEFT, upstream CUDA or an external serving
engine can consume them. Fusion/requantization/export is intentionally unsupported.

Project-based `SFTTrainer` open/resume checks the recorded engine and dependency
identity. The low-level standalone adapter loader does **not** currently enforce
legacy `engine_semantics` metadata, so a successful load is not evidence of a
compatible migration across the FP8 scale or other arithmetic changes. Keep
older adapters with their recorded runtime; use a new project for the changed
engine rather than treating a standalone load as a validated continuation.

## Memory and throughput

The 5,809-token full-parent ktune attempt at 32 GiB completed its forward pass
but failed during backward before an optimizer update. The new project completed
**one update in 1,862.1 seconds**, peaking at **73.95 GiB MLX**, with an **80 GiB
process cap, explicit 12 GiB system reserve, 8 GiB packed cache, decoded cache off,
and explicit chunk 8192**. The checkpoint audit found all 160 adapter tensors and
322 optimizer tensors finite, all 80 LoRA B tensors nonzero, valid checksums and
Adam step 1. The fresh fitting check **failed**: same-answer NLL rose from
**0.5012578592 to 0.5053724668** (delta +0.0041146076; lower is better), despite
changed logits. This is an operational pass, not an improved adapter. The original
checkpoint is preserved. All three tested inference strengths (+0.1, -0.1,
adaptive +10) also worsened NLL; the bounded series is complete, with no adapter
promoted as an improvement. See the [aggregate](reports/ktune-strength-diagnostics.json).
Component VJP checks and the current
518-test plus 27-subtest suite pass, but do not resolve the first fitting failure.
See the [diagnosis update](reports/first-step-diagnosis-update.md).
The saved 27.66 GiB observed-process reading is an end-of-run observation, not
peak memory. See the [checkpoint audit](reports/ktune-full-parent-checkpoint-audit.json)
and [failed fitting receipt](reports/ktune-full-parent-first-update-fit.json).
There is no completed low-memory full-update baseline for a speed ratio.
The complete-update rate was approximately 3.12 serialized example tokens/s,
or 0.44 supervised answer tokens/s. Fresh full-context scoring took about
523–570 seconds per condition; these measurements are not generation speed.
Use the complete [kernel tuning recipe](KERNEL_TUNING.md#command-line-recipe) for
the pinned case, parent context, actual paths, and resume commands.

The 12 GiB reserve is a recorded choice for that run's available startup headroom,
not the development preset default. Its startup gate is 92 GiB free/reclaimable;
the unmodified preset keeps a 32 GiB reserve and requires 112 GiB. Resuming without
resource flags inherits saved limits. Explicit `--profile development` resets
resource defaults, so preserving a smaller reserve requires repeating its
explicit override. Changing only `--reserve-gib` preserves other saved limits.

Packed-only caching improved the observed real single-MoE-layer sequence from
57.902 to 53.728 seconds at chunk 512, and from 48.536 to 44.103 seconds at chunk
8192, with same-chunk bitwise output/dX parity. These are one-sequence measurements
using saved baseline receipts, not complete-model throughput. The prototype
8 GiB packed + 32 GiB decoded configuration was slower at both chunks, so decoded
caching remains off. See the [measurement and scope review](reports/high-memory-cache-review.md).
`--decoded-cache-mib 32768` remains an explicit experimental option; historical
prototype receipts retain their original numeric settings.

The memory profile retains the **512 chunk default**. This pilot's explicit
`--chunk-size 8192` changes reduction order and is immutable within the project;
cache parity at one chunk does not establish equality across chunks. Use a new
project to change it and retain full-update numerical and timing checks as
separate gates. See [explicit decoder chunks](RESOURCE_PROFILES.md#explicit-decoder-chunk-experiments).

This is **end-to-end backpropagation with streamed weights and block recomputation**.
It is not separately training layer 0, then layer 1. Base trunk weights stay
resident; selected packed experts and Engram rows are read as needed. Backward
recomputes blocks and re-reads expert matrices. Only adapter parameters and their
optimizer state are trainable. Selecting fewer adapter layers saves adapter state,
but still executes the full network.

The process budget (16 GiB by default, up to the explicit 80 GiB profile) checks MLX allocations/cache and process peak RSS every
100 ms. It also checks free/reclaimable system memory every five seconds and
preserves the configured reserve (32 GiB by default); startup requires budget
plus reserve. MLX's allocator
limit is advisory, **not an OS hard quota**. An allocation can overshoot before the
watchdog runs. Exit code 75 means the watchdog terminated this process; `memory-abort.json` records the reason when
it can be written. A failed
startup headroom check means another workload needs to release memory first.

If memory rises, lower sequence length first, then adapter rank and selected expert
adapter layers. Use smaller packed cache (`--cache-mib 64`) or decoder chunks
(`--chunk-size 128`) if needed; smaller chunks cost speed. Keep allocator cache
bounded and block recomputation on. A lower memory budget tightens the guard;
raising it or removing the reserve does not make the model more memory efficient.
No command here stops other servers or deletes existing models.

The largest complete training example tested is **5,809 serialized tokens**;
its update completed but fresh answer NLL worsened. The implementation context
ceiling is 32,768 tokens, not a tested training limit. Sparse index selection
and candidate blocks follow the frozen model; attention processes query tiles and
recomputes each tile's first-order VJP. The LM head processes supervised rows in
32-token chunks. Backbone block boundaries still consume memory proportional to
context. SSD traffic grows with the number of distinct routed experts. Start with
a complete short task, then increase lengths while recording time, reads and peak
memory. See [kernel tuning](KERNEL_TUNING.md) for the ktune importer and evidence gates.
Host activation materialization currently prevents retained decode graphs at the
cost of synchronization and copies. Multi-token training throughput is experimental.

## Reused work and remaining gates

We reuse MLX-LM 0.32.0's `TextDataset`, `ChatDataset`, `CompletionsDataset`, masked
causal loss and LoRA forward implementation, plus MLX's Adam and gradient clipping.
The LoRA initializer avoids allocating an unused dense base for each streamed
expert. The format/trellis math comes from the upstream exl3-qlora/exllamav3 code,
and architecture math from official DeepSeek inference. See [sources](SOURCES.md).

MLX-LM's inspected DeepSeek V4.1 implementation rejects Engram-enabled models,
so its full loader and compiled training loop cannot replace this streamed path.
PonyExl3's fused Metal inference kernels are a promising optimization reference;
we have not imported them or claimed an autograd implementation from them.

Before production use: independent full-model logit parity, a representative
held-out task evaluation, long-run numerical stability/throughput measurements,
and an adapter mapping to the intended serving engine. Original 2.8-bpw full-model
loading, faster long-context execution and Qwen architecture support remain separate work.
