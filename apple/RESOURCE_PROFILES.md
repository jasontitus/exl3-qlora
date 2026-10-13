# Memory and cache profiles

The explicit `development` profile permits an 80 GiB process budget, leaves
32 GiB of system reserve, and retains up to 8 GiB of packed expert weights and
256 MiB of allocator cache. Decoded and frozen-prefix caching are disabled by default. Configured
caches count **inside** the process budget. Startup therefore requires at least
112 GiB free/reclaimable system memory. The watchdog still checks process usage
and system reserve; it is not an OS hard quota and individual allocations can
overshoot before the next check.

```sh
exl3-mlx finetune --model MODEL --data examples.jsonl --output PROJECT \
  --profile development --steps 2

exl3-mlx ktune finetune --source SNAPSHOT --model MODEL --output PROJECT \
  --profile development --steps 2

exl3-mlx resume PROJECT --steps 4 --profile development --no-compare
```

The default `low-memory` profile keeps the existing 16 GiB budget for ordinary
SFT/workflow commands and 32 GiB for the ktune training command. It retains
128 MiB of packed weights, disables decoded caching, and permits 64 MiB of
allocator cache. All defaults keep 32 GiB system reserve and block recomputation.

Explicit `--memory-gib`, `--reserve-gib`, `--packed-cache-mib`,
`--decoded-cache-mib`, `--prefix-cache-mib`, and `--allocator-cache-mib` override a profile. The existing
`--cache-mib` flag remains an alias for packed cache. Cache sizes must be integer
MiB values; zero disables a cache. Process limits are finite and within 1–1024 GiB; a nominated value must also pass the live hardware/headroom checks. The development preset remains 80 GiB.
Combined caches must leave at least 8 GiB, or half of a smaller process budget,
for model state and workspaces. This is a configuration check, not an estimate
that any particular context length fits; reduce caches when activations need
more memory.

### Optional frozen-prefix cache

`--prefix-cache-mib` (Python: `SFTConfig(prefix_cache_mib=...)`) sets an in-memory
activation cache, **0/off by default in both profiles**. It reuses the complete
context computed by frozen blocks strictly before the earliest adapted block.
For `--adapter-layers 39`, blocks 0–38 can be reused when the exact same example
recurs; block 39 is computed again. Adapting block 0 leaves no frozen prefix to
cache. The cache does not shorten examples or discard their causal context.

This cache is separate from packed/decoded weight caches, counts inside the
same process cap, and may evict entries to honor its MiB limit. Keep it disabled
until cold/warm forward and adapter-gradient parity have been verified for the
runtime. After that gate, **2048 MiB** is a starting cache budget for a small
repeated-example, last-block experiment, not a universal memory or speed claim.
Studio exposes **Frozen-prefix cache (MiB)** and shows its saved value.

Unlike adapter-layer selection or decoder chunk size, this exact-reuse cache is
a resource setting: a same-runtime project may change it during resume with
`--prefix-cache-mib 2048`, and subsequent resumes inherit the value. An explicit
profile resets it to zero unless overridden. Implementation identity checks
still apply; adding this feature does not permit resuming older projects under
changed source. Cache entries are not optimizer checkpoints or portable adapter
weights. The active diagnostic runtime remains frozen separately.

When `--config` points to JSON containing previously resolved resource limits,
an explicit CLI `--profile` resets those limits to the selected profile. Explicit
memory/cache flags then override the new profile. Without an explicit CLI
profile, the JSON resource values are preserved. This applies to both project
`finetune` and engine `train` commands.

```python
from exl3_mlx import SFTConfig, SFTTrainer

config = SFTConfig(output_dir="PROJECT", profile="development")
# Optional decoded-cache experiment; this slowed the measured MoE workload:
config = SFTConfig(output_dir="PROJECT", profile="development", decoded_cache_mib=32768)

trainer = SFTTrainer.from_project(
    "PROJECT", max_steps=4, resources={"profile": "development"}
)
trainer.train(resume_from_checkpoint=True, compare=False)
```

Project manifests contain resolved limits, and each training attempt writes
`resources.json` before launching its preflight. A resource-only resume records
the new limits and later resumes inherit them. Specifying a new profile resets
resource defaults before applying explicit overrides. Learning rate, optimizer,
data, chunk size and engine settings keep their checkpoint checks. Changing
source/runtime still requires the recorded environment; resource overrides do
not bypass implementation identity checks.

The completed one-update, 5,809-token full-parent pilot used a run-specific
`--profile development --reserve-gib 12 --chunk-size 8192` with packed cache
8 GiB and decoded caching off. That explicit reserve makes startup require
92 GiB free/reclaimable instead of the preset's 112 GiB. It was selected for this
host's available headroom; it does not alter the 32 GiB default or bypass the
watchdog. Its update took 1,862.1 seconds and peaked at 73.95 GiB MLX. The fresh
fit check failed: NLL rose from 0.5012578592 to 0.5053724668. The completed
[three-strength diagnostic series](reports/ktune-strength-diagnostics.json)
also found no improvement and promoted no adapter. Thus memory feasibility
passed, while improved fitting did not. The subsequent standard final-block
run completed three examples up to 5,822 tokens in 30.14 minutes including its
final probe, peaking at 53.08 GiB MLX. Fresh uncached loss improved on all three
rows and exact warm/reload logit parity passed. Its baseline evaluator used
80 GiB; the adapter evaluator used a declared 72 GiB retry with reserve8
unchanged after a startup-headroom failure. See the
[result and resource provenance](reports/metal-tuning-result.md). See the [checkpoint audit](reports/ktune-full-parent-checkpoint-audit.json),
[failed fitting receipt](reports/ktune-full-parent-first-update-fit.json), and
[complete pilot recipe](KERNEL_TUNING.md#command-line-recipe). Resume without
resource flags inherits the saved 12 GiB reserve; specifying the profile again
resets it to 32 GiB unless explicitly overridden.

Studio exposes the profiles and cache limits when preparing a project. Existing
projects retain their settings unless “Apply these resource settings to the next
Train / resume” is checked. Generation and comparison use the saved project
resources. Saved resolved limits are displayed separately, and opening a project
initializes the resource form from its manifest without overwriting edits on
each background refresh.

The [single-MoE-layer benchmark](reports/decoded-cache-moe-benchmark.json) used
5,808 synthetic normalized BF16 input rows with real layer-0 weights and routing.
The prototype's 8 GiB packed + 32 GiB decoded combination took 62.79 seconds versus
57.90 seconds for the 128 MiB packed/decoded-off configuration at chunk 512,
and 54.69 versus 48.54 seconds at chunk 8192. These totals include one forward
followed by a forward-plus-VJP phase. Same-chunk outputs and gradients were exact,
but the combined cache configuration was slower in this one-sequence experiment.
Order and first-use JIT can affect timing; this is not full-model throughput or
an isolated estimate of decoded-cache cost.

Consequently the current development profile leaves decoded caching off. To
repeat the experimental 32 GiB decoded-cache condition, explicitly add
`--decoded-cache-mib 32768`. Historical benchmark records retain their original
prototype profile labels and limits; use the numeric settings when comparing
them with current presets. Previously saved project limits remain explicit and
are not silently rewritten by the new preset.

The subsequent [packed-only experiment](reports/packed-cache-moe-benchmark.json)
observed 53.728 versus 57.902 seconds at chunk 512 and 44.103 versus 48.536 seconds
at chunk 8192, with bitwise same-chunk output/dX hashes. These are actual
single-MoE-layer forward-plus-recomputation/VJP sequences with synthetic input,
compared against the preceding baseline receipt; timing may depend on JIT,
filesystem cache and order. Comparing 44.103 against 57.902 changes both caching
and chunk size, so that full difference cannot be attributed to caching.

A separate synthetic final-block VJP probe retained 17.95 GiB before its block,
passed finite/nonzero-gradient checks and peaked at 58.76 GiB. It used an
80 GiB cap, **16 GiB reserve**, packed cache 8 GiB, decoded cache off and chunk
8192. Neither this probe nor the single-layer timing is a complete optimizer
update or proof of full-model throughput. The [review and raw receipt links](reports/high-memory-cache-review.md)
record their exact settings and limitations; the full pilot's 12 GiB reserve is a
separate run choice.

Decoded caching reuses frozen FP16 tiles. It does not enlarge the EXL3 column
chunk (512), disable block recomputation, or change adapter training targets.
Benchmark cache-disabled and enabled paths on the same inputs and verify
forward/gradient/update parity before claiming a speed improvement. A bounded
cache can thrash when the working set exceeds capacity; a larger budget alone
does not establish faster or better fine-tuning. Inspect configured limits,
streaming hit/load/eviction counters, retained bytes, peak MLX/RSS measurements
and elapsed time in the run reports.

## Explicit decoder chunk experiments

New SFT projects also accept `--chunk-size 8192` (Python `SFTConfig(chunk_size=8192)`).
The default remains 512 in every resource profile. Supported project values are
integer multiples of 128 from 128 through 32768. Chunk size changes matrix
reduction order, so results need not be bitwise equal even when they meet a
declared numerical tolerance. It is an immutable numerical setting, excluded
from resource-only overrides; changing it requires a new project.

```sh
exl3-mlx ktune finetune --source SNAPSHOT --model MODEL --output NEW_PROJECT \
  --profile development --chunk-size 8192 --steps 2
```

The saved choice reaches training, likelihood evaluation and generation, and is
checked in each optimizer checkpoint's contract. Exports record it in
`bundle.json`; pass the same `--chunk-size` explicitly when using a standalone
engine command with that exported adapter. A small matrix benchmark can suggest
a useful candidate, but does not establish full-model numerical quality,
training throughput or memory fit. Compare outputs and activation/adapter
gradients against the 512-column path, then measure a complete update before
selecting a wider chunk for a real run.

## M5 training host qualification (2026-10-13)

The new training target is an M5 Max with 40 GPU cores and 128 GiB unified
memory. Its initial inventory showed approximately 90 GiB of free pages and
2 TiB available on the Scratch volume. These observations are not a permanent
memory allowance. Recheck whole-system availability before each model launch.

Use an isolated environment with the frozen runtime-v5 source and dependency
versions. The exact DeepSeek checkpoint is being copied to
`/Volumes/Scratch/exl3-training/models/deepseek-v4.1-flash-exl3-3.0bpw`.
Its complete source inventory contains 365 files and 426,352,827,347 bytes.
`apple/scripts/checkpoint_digest.py` performs bounded full-byte SHA256 reads,
rejects empty/invalid inventories, and can verify the destination against the
source manifest. Six metadata hashes in the trainer's base identity are not
an equivalent full-payload check. Do not train on a partial copy.

The M5's default MLX arithmetic failed 13 of 35 small numerical tests.
`MLX_ENABLE_TF32=0` reduced this to one failure, without changing source or test
tolerances. The remaining strict RoPE test differs by at most 7.08e-6 from its
binary64-angle reference: the measured FP32 frequency gives 59.999996 instead
of 60 at position 600. An independent same-angle trigonometric reference agrees
within 2.39e-7. The original failure is retained in `reports/m5`; this is a
bounded arithmetic diagnosis, not full-model parity or long-context qualification.

`apple/scripts/run_m5_plan.py` wraps the existing five-phase experiment plan.
It requires the explicit TF32-off environment, a completed and hash-bound full
payload verification receipt, and a fresh project-owned prefix cache. It records
the numerical mode for every child. Independent CPU lifecycle tests cover these
gates; the concrete remote plan and longest complete-item memory preflight remain
required. Run baseline, prior-adapter and final-adapter comparisons afresh on the
M5 with identical arithmetic. Do not reuse M1 or default-TF32 prefix caches.

No M5 full-model training speedup or quality improvement has been measured yet.
Keep unrelated workloads running, retain the configured system reserve, and tune
bounded caches before considering a separately recorded shorter-item experiment.
