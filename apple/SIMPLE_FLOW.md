# Standard supervised fine-tuning

This interface is implemented. It puts a local browser, a project CLI and a small
`SFTTrainer` Python API over the same streamed EXL3 training engine. Start with a
short pilot, inspect the saved comparison, then decide whether to train further.

## Launch the local interface

```bash
cd ~/experiments/exl3-qlora/apple
uv sync --locked
uv run exl3-mlx studio
```

Studio opens a local, authenticated browser page. Files stay on this Mac.

1. Enter the **model folder**, **dataset file** and a **new project folder**.
2. Select **Check data**. Review split counts, longest example, context and
   supervised answer previews. The tool rejects overlong examples instead of
   silently shortening them.
3. Leave **Total optimizer updates** at 2 and select **Train / resume**.
4. Select **Compare base vs adapter**. Read likelihood, the fixed prompt preview,
   and whether generation reached EOS. Try another prompt with **Generate**.
5. Increase total updates to continue, or export the selected adapter.

Stop finishes the current optimizer update and saves its checkpoint. The update
may take several minutes. Closing Studio does not terminate an active job; use
`exl3-mlx stop PROJECT` if needed. A browser refresh reconnects to the local job;
restarting Studio requires **Open existing project**. The project path field is
also used for opening an existing project. Uploads are limited to 16 MiB; larger
local files can be entered by path.

## A command-line run

The checked-in CSV is a synthetic plumbing demonstration. Its slot prompts all
share one answer; success on it is not evidence of domain or Metal capability.
Use a new project directory for each different experiment.

```bash
export EXL3_MODEL="$HOME/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"

uv run exl3-mlx finetune \
  --model "$EXL3_MODEL" --data examples/sft/slot-demo.csv \
  --output artifacts/my-pilot --steps 2 --prepare-only

# Review project.json and the exact context/answer previews in data/dataset.json.
uv run exl3-mlx resume artifacts/my-pilot --steps 2
uv run exl3-mlx chat artifacts/my-pilot --prompt 'What is the code for slot E?'

# This reaches 10 total updates, not 10 additional updates.
uv run exl3-mlx resume artifacts/my-pilot --steps 10 --no-compare
uv run exl3-mlx compare artifacts/my-pilot
uv run exl3-mlx export-adapter artifacts/my-pilot --output artifacts/my-adapter
```

Omit `--prepare-only` to import, train and compare in one command. With no path
flags in an interactive terminal, `finetune` asks for the three paths. All
commands accept `--help`. Advanced settings can be supplied with `--config
settings.json`; keys are `SFTConfig` field names and explicit CLI flags override
JSON values. `status PROJECT` reports progress and `stop PROJECT` requests a safe
stop. Comparison uses a fresh process for each condition and produces
`comparison.json` and `report.html`.

## Accepted datasets

| Input | Example / behavior |
|---|---|
| CSV | `question,answer` or `prompt,completion`; quoted multiline fields and Unicode are preserved |
| JSONL or JSON array | One of the schemas below per record |
| `prompt` / `completion` | User prompt and desired assistant answer |
| `question` / `answer` | Equivalent column names |
| Alpaca | `instruction`, optional `input`, `output`; instruction and input joined with two newlines |
| Chat | `messages` containing text `role` / `content`; optional system, alternating user/assistant, final assistant answer |
| ShareGPT | `conversations` with `from` / `value`; system/human/gpt mapped to the corresponding chat roles |
| Custom columns | `--prompt-column TASK --response-column PATCH` |
| Explicit splits | Folder containing `train.jsonl`, optional `valid.jsonl` or `validation.jsonl`, optional `test.jsonl`; JSON/CSV variants also work |

Single-file imports split by connected source groups, identical prompts and exact
token duplicates, with a fixed seed. Supply `source_id`, `document_id`, `group_id`
or `--group-column` to keep related examples together. Automatic splitting needs
at least three independent groups. With a smaller fitting demonstration, use
`--demo`; this produces training-only data and makes no quality claim. Explicit
train-only or train/validation folders remain valid; missing splits are never
filled from training data. Cross-split duplicates or source overlap are rejected.

Grouping cannot detect every paraphrase, common task template or related patch.
Use commit families, repository/task families and human-reviewed split policy for
real evaluation. Extra metadata stays in provenance and does not enter model
input. Ambiguous schemas, empty answers, malformed roles, tool calls and image or
audio records fail with a source row error.

The final assistant answer and EOS receive loss. System prompts and previous
turns provide context. `max_length` is **all serialized tokens**, including BOS,
prompt and EOS; it is inferred by default, at most 512. Backend shifted input
length is one token smaller. Generation separately requires prompt plus requested
new tokens to fit 512. No packing, all-assistant mask, or silent truncation is
performed by this interface.

## Python and Hugging Face datasets

```python
from exl3_mlx import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model="/absolute/path/to/DeepSeek-V4.1-streaming-repack",
    args=SFTConfig(
        output_dir="/absolute/path/to/my-project",
        max_steps=20,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
    ),
    train_dataset="/absolute/path/to/examples.csv",
)
manifest = trainer.prepare()  # inspect manifest["data"]["preview"]
result = trainer.train()
trainer.save_model("/absolute/path/to/exported-adapter")

# A new Python process can continue the same data, optimizer and RNG state.
trainer = SFTTrainer.from_project("/absolute/path/to/my-project", max_steps=30)
trainer.train(resume_from_checkpoint=True)
```

`train_dataset`, `eval_dataset` and `test_dataset` also accept iterables yielding
row dictionaries, including Hugging Face `datasets.Dataset` objects. If explicit
evaluation/test datasets are supplied, their splits are preserved; a train
iterable alone is grouped and split like a file. Import materializes at most
100,000 rows per split. `datasets` is an optional caller dependency, not needed
for local CSV/JSON. An existing project rejects a newly supplied dataset: use a
new output directory to change examples or preprocessing.

This API uses familiar SFT names, but is **not a drop-in TRL, Unsloth, MLX-Tune,
PEFT or mlx_lm trainer**. It reuses MLX-LM's tokenization/data processors, loss and
LoRA forward convention, plus MLX autograd/Adam, while the full-model streamed
backend is custom. GPU operations run in child processes so a memory watchdog
cannot exit a notebook. Unsupported options fail explicitly. See
[SOURCES.md](SOURCES.md) for the reused implementations.

## Saved state and selection

A project snapshots prepared and normalized data, provenance, tokenizer metadata,
configuration, implementation hashes and dependency versions. Resume requires the
same recorded environment and immutable settings; only the total step target can
increase. Kernel and Python changes invalidate exact continuation. Preserve the
checkout and environment for long-lived experiments. Model receipts fingerprint
metadata and verify bank structure; they do not hash every byte of the enormous
frozen weight files, which must remain immutable.

Each continuation uses a new attempt directory. Checkpoints save adapter weights,
Adam moments, RNG and sample cursor atomically. The latest checkpoint resumes
training; the lowest validation-loss checkpoint supplies comparison/generation/
export. Without validation the latest is selected. `save_total_limit` retains
recent checkpoints **per attempt** plus best checkpoints and dependencies needed
for their resumability; it is not a global disk quota. Repeated resumes therefore
accumulate checkpoints. No model files are deleted.

The default trains rank-4, alpha-8 LoRA on `wq_a` and `wo_b` in all 40 attention
layers. The EXL3 experts stay frozen and contribute activation gradients.
Microbatch size is one; use gradient accumulation for a larger effective batch.
The advanced [engine recipe](RECIPE.md) exposes expert adapters and narrower
layer selection, which this initial standard facade does not yet expose.

Exports contain inference adapter weights/configuration and a bundle receipt,
without a partial optimizer checkpoint. They load with this tool's
`exl3-mlx generate --model MODEL --adapter EXPORTED_FOLDER --prompt '...'`.
Training resume uses the original project. Export is not PEFT or mlx_lm format.

## How to interpret a result

Validation selects a checkpoint. `compare` evaluates up to four test examples by
default (configurable) or explicitly labels validation-only measurements as
`selected_development`. It compares token-weighted answer loss, not just training
loss. A fixed chat history is also generated with base and adapter, using the same
prompt IDs and token cap. Matching a prefix without EOS does not count as an
exact answer. Repeated or interrupted test access is durably counted and later
results are labeled `previously_observed_test`.

An improved loss is an observed likelihood improvement, **not a quality
certificate**. The sample count, full receipts, stop reason and predictions remain
visible. Select hyperparameters using validation; repeatedly checking and adapting
to the test set turns it into development data. Use representative task scores,
regression controls and independent test sets before claiming usefulness.

## Current boundary for Metal optimization training

Only the 40-layer DeepSeek V4.1 MXFP8-trunk/EXL3 **streaming repack** is supported
end-to-end. Original EXL3 shard layouts and Qwen do not become supported through
this interface. Most complete optimization diffs exceed the 512-token limit;
the earlier corpus audit found 19 of 20 candidate changes too long. See the
[Metal training assessment](reports/metal-training-assessment.md).

For examples that fit, turn a verified optimization into a task: pre-change code,
hardware/workload constraints and benchmark evidence in the prompt; the justified
patch in the answer. Keep repository, parent/commit hashes, benchmark result and
commit family in metadata/group IDs. A commit message claiming success is not
proof. Hold out related optimization families together and evaluate generated
patches by compilation, numerical correctness, repeated on-device speed/memory
measurements and regressions. These execution-based task metrics are not yet
built into Studio. The tool can train short examples today; it cannot yet
establish improved Metal optimization ability or support realistic long patches.
