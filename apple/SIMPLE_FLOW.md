# Standard supervised fine-tuning

This interface is implemented. It puts a local browser, a project CLI and a small
`SFTTrainer` Python API over the same streamed EXL3 training engine. Start with a
short pilot, inspect the saved comparison, then decide whether to train further.

For long kernel-patch data, use the [kernel tuning recipe](KERNEL_TUNING.md).
A completed final-block diagnostic improved fitting on one **5,809-token** edit.
The [normal three-change flow](reports/standard-three-change-run.md) then completed
12 updates on complete 3,606 / 5,809 / 5,822-token examples under an 80 GiB guard
(**53.08 GiB measured peak**, **30.1 minutes** including the warm probe). Fresh
paired scoring improved answer NLL **8.23%**, with all three rows improving and
exact warm/fresh uncached logit agreement. Only one additional added-code token
became correct; most loss reduction came from comments and headers. This is a
training-fit demonstration, not proof of useful generated kernels. The current
integrated source suite also passed **1,040 tests plus 27 subtests**; the
[quality guide](QUALITY.md) separates that test result from model-quality claims.

Verification preserved a successful 80 GiB baseline and retried only the adapter
score at 72 GiB after a startup-headroom failure; reserve remained 8 GiB and the
math did not change. Training used 80 GiB. The existing project can be inspected
with `artifacts/loss-fix/metal-tune status`, which uses its pinned runtime.
An 80 GiB budget does not certify 32,768-token training or model quality. The
short-text comparison/generation steps below are not practical long-patch inference.

## Launch the local interface

```bash
cd ~/experiments/exl3-qlora/apple
uv sync --locked
uv run exl3-mlx studio
```

Studio opens a local, authenticated browser page. Files stay on this Mac.
Under **Data and resource settings**, choose **Development (80 GiB)** when the
machine has at least 112 GiB reclaimable for the process plus system reserve.
Studio shows the project's saved limits. For an existing project, check **Apply
these resource settings to the next Train / resume** to change them. See
[resource profiles](RESOURCE_PROFILES.md) for cache limits and overrides.

1. Enter the **model folder**, **dataset file** and a **new project folder**.
2. Select **Check data**. Review split counts, longest example, context and
   supervised answer previews. The tool rejects overlong examples instead of
   silently shortening them.
3. For a short text pilot, leave **Total optimizer updates** at 2 and select
   **Train / resume**.
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

Add `--profile development` to `finetune` or `resume` for the explicit 80 GiB
budget with an 8 GiB packed cache; decoded caching remains off after a slower
measured MoE-layer result. Default ordinary SFT remains 16 GiB. For example:

```bash
uv run exl3-mlx resume artifacts/my-pilot --steps 10 --profile development --no-compare
```

An explicit CLI profile replaces resource limits from a JSON configuration;
explicit cache/memory flags then override that profile. Other training settings
retain their JSON values. The profile increases available caching, not a promise
of faster or better training.

For an explicit decoder experiment, new projects accept `--chunk-size 8192`.
Studio exposes **Decoder chunk size (new projects only)** under advanced settings
and shows the saved immutable value in the project summary.
Both resource profiles default to 512. Wider chunks change floating-point
reduction order, so chunk size cannot change during project resume. Read the
[decoder chunk guidance](RESOURCE_PROFILES.md#explicit-decoder-chunk-experiments)
and numerical benchmark results before selecting it; larger chunks are not a
claim of faster full-model training.

New projects can select attention adapter blocks with `--adapter-layers 38 39`
or `SFTConfig(adapter_layers=[38, 39], ...)`. IDs are zero-based: `39` is the last
block, while omitting the setting keeps the default **all 40 blocks**. Each
selected block adapts `wq_a` and `wo_b`; this option does not select expert
adapters. Studio exposes **Attention adapter blocks (new projects only)** and
shows the saved selection. Blank means all 40.

The full model context remains part of the computation.
Selecting fewer adapter blocks is useful for controlled fitting experiments,
but is not a demonstrated quality fix or a guarantee of a particular memory
reduction. The selection is immutable after data preparation: resume,
comparison, generation and export use the recorded scope. Create a new project
to change it; resource overrides cannot alter it. The same option is available
on `ktune finetune`.

An optional **Frozen-prefix cache (MiB)** reuses frozen earlier blocks for an
identical repeated example. It is off by default and included in the process
memory budget. `--prefix-cache-mib` is a resource-only setting that may change
on resume; it cannot change the saved adapter-layer selection. See the
[cache verification gate and limits](RESOURCE_PROFILES.md#optional-frozen-prefix-cache).

## Experimental activation arithmetic

The default `activation_mode="checkpoint"` preserves the existing FP8/FP4
activation roundtrips. A new project may opt into `--activation-mode bf16`
(`SFTConfig(activation_mode="bf16", ...)`, also available under Studio's
**Activation arithmetic** setting). This bypasses differentiable-path activation
roundtrips while retaining packed weights, FrozenIndexer quantizers, BF16 casts,
and normally recomputed expert/top-k routes. It retains quantization and rounding. A completed one-row, one-update comparison
observed BF16-mode descent (0.5061723590 → 0.5014906038) and checkpoint-mode
regression (0.5012578368 → 0.5033001955). This supports a bounded experiment,
not a general fix or a unique causal diagnosis; the modes have different forward
objectives and the control did not exactly reproduce the historical update. See
the [independent control review](reports/matched-control-final-independent-review.md).

Changing this mode changes the **zero-adapter forward pass**, so BF16 experiments
need their own freshly scored baseline. Project comparison and training-fit
checks pass the same saved mode to both baseline and adapter evaluations. The
mode is immutable after preparation and cannot be changed by resume or resource
overrides; use a new project for a different mode. Exported adapters record it.
The low-level `train`, `evaluate`, and `generate` commands inherit the mode from a
saved adapter when the flag is omitted and reject an explicit mismatch, including
an adapter-only warm start. A legacy missing mode means `checkpoint`; this does
not waive source/runtime identity checks or migrate old projects.

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
prompt and EOS; it is inferred by default, at most 32,768. Backend shifted input
length is one token smaller. Generation separately requires prompt plus requested
new tokens to fit 32,768. No packing, all-assistant mask, or silent truncation is
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
same recorded environment and immutable learning/data settings. The total step
target can increase, and resource-only overrides can change process/reserve/cache
limits. Resolved limits are recorded in the project and each attempt's
`resources.json`; later resumes inherit them. Kernel and Python changes invalidate
exact continuation. Preserve the
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
Use `--adapter-layers` for narrower attention-layer selection in this standard
interface. Expert adapters remain an advanced [engine recipe](RECIPE.md) option.

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
this interface. The backend now accepts up to 32,768 tokens with sparse selection
and bounded attention/head workspaces. Full-model memory still grows with context;
this maximum is an implementation ceiling, not a throughput or RAM guarantee.
The earlier all40-layer 5,809-token first update’s fresh fitting
check failed (NLL 0.5012578592 base versus 0.5053724668 adapter); separate strength
diagnostics at +0.1, -0.1 and adaptive +10 also worsened NLL. This bounded series
is complete, with no adapter promoted as an improvement; see the
[aggregate](reports/ktune-strength-diagnostics.json). That series did not establish improved fitting. The later one-row final-block
result and completed standard three-row verification are linked at the top of
this page. The completed [activation control](reports/matched-control-final-independent-review.md)
supports further quantizer testing but does not prove a unique cause. See the historical
[diagnosis update](reports/first-step-diagnosis-update.md) and
[verification audit](reports/ktune-full-parent-verification.json).
The existing generation path recomputes the whole prefix per new token; practical
long-patch generation still needs compatible inference with a persistent KV cache.

Use `exl3-mlx ktune --help` and the [kernel tuning guide](KERNEL_TUNING.md) to snapshot,
audit, prepare and train from the optimization archive. That flow records historical
evidence scope, preserves failures, audits complete prompt/answer lengths and keeps
related revisions together. A historical win or lower training loss does not prove
that newly generated patches improve an engine. Evaluate them by application,
compilation, independent numerical correctness and repeated on-device measurements.

The newer [upstream intake](docs/UPSTREAM_INTAKE.md) also accepts pinned Git
histories from mlx-serve, TensorFold, Sushi and llama.cpp. It records operation,
hardware, bottleneck and rationale with citations or explicit unknowns, preserves
complete source/evidence, and distinguishes reported claims from measured results.
`ktune finetune-upstream --source MANIFEST --model MODEL --output NEW_PROJECT
--prepare-only` imports and binds a normal SFT project without training. The
reviewed 3,394-token llama.cpp example has completed this prepare-only path;
it has **zero optimizer updates**. Preparation is separate from the three-change
trained adapter described above.

The bounded 100-candidate audit yielded 93 reconstructed source pairs and only
five whole-file examples within the token ceiling, not 100 ready kernel tasks.
Exact reviewed source units provide a practical way to keep complete kernels and
relevant dispatch/type context. Their task annotations remain explicitly
retrospective, and external dependency completeness needs review. See the
[inventory](reports/upstream-corpus-inventory.md).

Separately, the original llama.cpp convolution pair passed 16 fixed GPU/CPU
parity cases, and the original TensorFold pair passed 65,536 codebook states plus
162 FP32 projection parity cases on this M1 Ultra. These scoped checks used
captured, hash-bound compiled sources and canaries; they did not time kernels or
check any model-generated optimization. The [quality guide](QUALITY.md#completed-activation-control-and-upstream-checks)
links the raw receipts and excluded paths.

The long-context revision also fixes negative power-of-two activation scales. It
changes forward arithmetic from the early prototype. Existing projects retain
their recorded implementation; exact resume under changed code is rejected. To
reproduce earlier outputs, use their original environment. A deliberate adapter
warm start through the advanced CLI starts a new experiment and needs evaluation
again; it is not exact continuation.
