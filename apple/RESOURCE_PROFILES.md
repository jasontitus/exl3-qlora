# Memory and cache profiles

The explicit `development` profile permits an 80 GiB process budget, leaves
32 GiB of system reserve, and retains up to 8 GiB of packed expert weights and
256 MiB of allocator cache. Decoded caching is disabled by default. Configured
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
`--decoded-cache-mib`, and `--allocator-cache-mib` override a profile. The existing
`--cache-mib` flag remains an alias for packed cache. Cache sizes must be integer
MiB values; zero disables a cache. Process limits are finite and within 1–80 GiB.
Combined caches must leave at least 8 GiB, or half of a smaller process budget,
for model state and workspaces. This is a configuration check, not an estimate
that any particular context length fits; reduce caches when activations need
more memory.

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
passed, while improved fitting did not; no longer complete training example has
been tested. See the [checkpoint audit](reports/ktune-full-parent-checkpoint-audit.json),
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
