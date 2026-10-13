# Upstream and contest source inventory, 2026-10-12 PDT

The earlier 24 examples came from a narrow official-contest import, not the available source archive or a tool limit. The four requested upstream repositories contain substantial additional material. We have pinned their current default-branch histories and materialized eight illustrative changes, but have not yet established that 100–200 complete, independent, context-fitting training examples are ready. No candidate code was executed in this inventory.

## Why the prior import contained 24 examples

The full-parent audit considered 100 rows: 35 curated snapshot records and 65 official diffs. Its exclusions were 47 records outside the selected `official_promoted` evidence tier; 19 full serialized examples above 32,768 tokens; nine reconstruction failures (four binary edits and five unavailable pinned parents); and one exact complete-token duplicate. That leaves 24. The official tier started at 53 records. This filter was appropriate to that bounded experiment, but unnecessarily narrow as a general intake policy for the user-authorized upstream training corpus.

The 24 examples span 3,606–25,486 tokens, median 17,806.5, in one conservative historical lineage. Three distinct examples have now been trained, each four times. The largest tested training example remains 5,822 tokens; fitting under the configurable 32,768-token ceiling is not a demonstrated memory or throughput guarantee.

## Existing remote archive

The authorized read-only source is `jason-m5-macbook:/Volumes/Scratch/model-work/ktune`, inspected at Git revision `f6fa17a6c5031b8dff2d4e4139871c8597342507`. The primary corpus `library/corpus/mlxfast-submissions.jsonl.zst` contains 7,406 submissions, 7,344 with diffs, and 766 raw `promoted` or `improved_superseded` labels. Its compressed SHA-256 was independently checked against the manifest: `3a67c5dfc127f6225f12a8ca8772eac63d475a94cf1d5b3b7465563bf5233000`. No full corpus was downloaded or executed.

Prior evaluation protections matter: the entire historical Gemma track was assigned to validation, not merely its explicitly marked test rows; Laguna-dev is protected too. Removing their 267 and 116 historical score-raiser labels leaves an upper bound of 383 outside those whole-track protections: 188 Laguna, 145 Qwen27-MTP, 50 early Qwen125. These are not 383 eligible examples. Family/hunk protection, related revisions, actual evidence, complete parents and lengths remain to be checked. A promoted label with absent or heterogeneous metrics does not imply an independently verified result.

Two existing refresh batches contain 570 gathered, unreviewed candidate rows; the published project breakdown includes 70 mlx-serve and 22 llama.cpp rows. Raw PR archives also contain `prs/ddalcu__mlx-serve.jsonl` and `prs/ggml-org__llama.cpp.jsonl`. Duplicate alias pulls exist. The old registry contains `incoai/splash`, which is a different project from Sushi; it contains neither TensorFold nor Sushi. Curated retrieval admission is retained as provenance, not imposed as a new-training requirement.

Checksummed remote metadata, schema counts and protection records are saved under [upstream-inventory](../artifacts/loss-fix/upstream-inventory/), including `ktune_corpus_metadata.json`, `ktune_corpus_counts_schema.json` and `ktune_protection_metadata.json`.

## Current named Git sources

Fresh isolated, blob-filtered bare mirrors and full commit metadata are under `artifacts/loss-fix/upstream-inventory/mirrors`. Existing working trees were not modified. Capture time and full heads are in [inventory.json](../artifacts/loss-fix/upstream-inventory/inventory.json).

| Source | Pinned head | Reachable commits | Numeric-performance-text code candidates |
| --- | --- | ---: | ---: |
| [mlx-serve](https://github.com/ddalcu/mlx-serve) | `59973964ad0a3e6a69ed8bc2bd7fa54d3cabfe31` | 827 | 123 |
| [TensorFold](https://github.com/ashhart/TensorFold) | `0efbd44755b07b1bffb7a51c010992fe11a6142f` | 1,560 | 172 |
| [Sushi](https://github.com/beamivalice/sushi) | `20f6068d8c46d7d293d3392b41e693c4919a0660` | 1,439 | 139 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `8f574b0ad44c46d7b499ec028c29205aaa3fe80f` | 11,579 | 240 |

The last column is a deliberately broad metadata heuristic: single-parent commits changing code, with performance-related text and a number matching a timing/speed pattern. It includes non-Metal work, neutral/correctness fixes, negative measurements and unverified claims. It is not an admission count.

A narrower Apple/Metal-related heuristic yields 455 repository rows, **412 distinct commit IDs**, including 254 with numeric performance text. See [apple-candidate-metadata.json](../artifacts/loss-fix/upstream-inventory/apple-candidate-metadata.json). Across their entire histories mlx-serve and Sushi share 496 exact commits; 43 duplicated rows occur in this narrower pool. Cherry-picked or edited equivalents still require patch/content comparison. Commit count and optimization-pattern diversity are different measures.

## Complete-context check on eight examples

Eight purposively selected changes have all changed parent/head blobs, complete diffs, full messages, timestamps, Git object identities and SHA-256 receipts materialized locally. [The manifest](../artifacts/loss-fix/upstream-inventory/representative-manifest.json) provides source-grounded annotation quotations; post-change messages remain target/provenance evidence, not blind task-prompt evidence. These examples are author-reported improvements, not independently reproduced benchmarks.

A tokenizer-only preliminary serialization included complete changed parent files, the full commit rationale and full diff. Nothing was truncated. [The length audit](../artifacts/loss-fix/upstream-inventory/representative-length-audit.json) found:

| Example | Total tokens | Fits 32,768 |
| --- | ---: | --- |
| mlx-serve HC one-row launch `431ac5d7` | 13,665 | yes |
| mlx-serve MoE down/reduce `340016fc` | 1,202,372 | no |
| TensorFold dense SIMD layout `22f41e22` | 21,867 | yes |
| TensorFold EXLR8 rounding `51426b8e` | 16,804 | yes |
| Sushi NAX vector load `27e81cfc` | 1,221,882 | no |
| Sushi GPU expert windows `a3df965a` | 203,345 | no |
| llama.cpp transposed-convolution bounds `560445bf` | 162,109 | no |
| llama.cpp expert-id dispatch `073b7403` | 148,771 | no |

The large contexts are primarily real monolithic source files, including roughly 3.65 MB `transformer.zig` files and a 450 KB Metal file. Three of eight fitting does not estimate the corpus-wide acceptance rate: this is a small purposive sample, and the final importer will add its own exact serialization overhead. Even these fitting examples omit unreviewed dependency closure and exceed the 5,822-token training length already demonstrated.

## Actual structured importer and reviewed-unit result

The final structured importer uses complete optimized after files as the target,
rather than duplicating a full diff. Its actual serialization therefore differs
from the preliminary table above: **two of the eight** whole-file examples fit,
mlx-serve `431ac5d7` at **28,115 tokens** and TensorFold `51426b8e` at **23,493**.
TensorFold `22f41e22` becomes 35,375 tokens and is excluded. This is an explicit
format change, not a silent truncation or a revised claim about the earlier audit.

A manually reviewed llama.cpp `560445bf` example now passes that same real importer
at **3,394 tokens: 1,873 prompt and 1,521 target**. It includes the complete
transposed-convolution kernel, its function typedef and both float/half
instantiations, the argument struct, Metal declarations and complete unchanged
host dispatch. Exact byte ranges, full original blobs and complete edit replay
are retained. Hardware/operation/bottleneck fields are source-grounded but
retrospectively authored from the commit; the task is labeled target-derived.
See [the reviewed manifest](../artifacts/loss-fix/upstream-inventory/reviewed-units/llama-conv-manifest.json)
and [actual import report](../artifacts/loss-fix/upstream-reviewed-llama-conv/ktune-import.json).

The exact isolated before/after kernel pair also **compiled successfully** with
the installed Xcode Metal compiler, `metal3.1 -O3 -fno-fast-math`. The
[composition receipt](../artifacts/loss-fix/upstream-inventory/reviewed-units/llama-conv-compile-inputs/composition.json)
binds each assembled segment to its reviewed source offsets and full-file hashes;
the [compiler receipt](../artifacts/loss-fix/upstream-inventory/reviewed-units/llama-conv-compile-metal31/receipt.json)
retains toolchain, flags, outputs and AIR hashes. Only the Metal kernel/type/include
translation units were compiled. The surrounding host/runtime was not built;
GPU execution, numerical parity and timing remain untested. This does not promote
the example to `execution_verified`.

A separate bounded 100-candidate audit was selected before observing its lengths:
25 per repository, evenly spaced by commit timestamp over the metadata pool,
with Sushi exact ancestors from mlx-serve removed first. Its
[selection plan](../artifacts/loss-fix/upstream-inventory/bounded100-selection.json)
is purposive rather than representative random sampling. Materialization completed
at **342.9 MB**, below its 1 GiB bound: **93/100** source pairs reconstructed,
with seven binary/unsupported/unavailable-source exclusions retained.

The **actual final CLI admitted five complete-file rows**, excluded 88 for context
length and retained the seven source failures. There were 93 unique normalized
patches. The preliminary parent-plus-diff audit had 14 fits; the final complete
before/after target format is the authoritative tool result. See the
[actual audit receipt](upstream-bounded100-actual-audit.json) and
[import report](../artifacts/loss-fix/upstream-bounded100-fullfiles/ktune-import.json).
The [independent CPU review](upstream-bounded100-independent-review.json) rehashed
all six output bindings and retokenized the five selected rows, including exact
IDs, masks, EOS and lengths.

| Source change | Actual complete tokens | Curation interpretation |
| --- | ---: | --- |
| mlx-serve `cc2b07cf`: affine-4 MoE SIMD ownership | 17,793 | Source-reported kernel improvement; not independently reproduced |
| TensorFold `eca8490b`: n-gram lookup scale | 26,058 | Compatibility/integration change; not a new optimized MSL kernel |
| TensorFold `57097f7e`: training on older Apple GPUs | 7,782 | Host compatibility; source explicitly reports only 10/11 finite-difference checks within its bound |
| TensorFold `9a92139a`: sparse-attention rescale skip | 9,615 | Source-reported Metal improvement; not independently reproduced |
| TensorFold `e82ab694`: paced aligned model loading | 23,269 | Loading/memory behavior; not a new optimized MSL kernel |

These five are mechanically token-fit `source_only` records, not five manually
certified optimization tasks. They span two conservative lineages and have zero
execution-verified results. Every one exceeds the 5,822-token training length
already demonstrated. Their known gaps remain in the source rationale, including
the partial finite-difference result. Unknown task metadata is not fabricated.
The compact reviewed llama.cpp example above is separate from these five.

This fixed audit gives a concrete reason to invest in reviewed source units and
pattern curation: counting raw commits cannot deliver 100–200 useful complete
kernel tasks by itself. None of these counts substitutes for manual task/evidence
review or clears inherited held-out protections.

## Practical route to a useful 100–200, then 300–800

1. Keep the 412 distinct metadata candidates as an intake queue, not a claimed training set. Classify actual improvements, correctness fixes, neutral changes, rejected approaches and mixed-result commits from source evidence. Preserve unknown hardware/workloads rather than inventing them.
2. Retain complete parent/head blobs and diffs in provenance. For oversized changes, review complete MSL functions or source units, host dispatch, required types/helpers and validation harness dependencies. Do not treat regex-selected hunks as complete executable kernels. The user-requested structured task can have a compact reviewed context while its source provenance remains complete.
3. Build each task from pre-change operation, contract, hardware/workload and baseline evidence. Put source-reported profiling rationale, design choices, implementation and reported verification in the answer or sidecar with explicit attribution. A model-generated “tests passed” statement must never become a harness receipt.
4. Merge exact commits, patch-equivalent edits, related revision series and shared source lineages before splitting. Check known protected family/hunk matches; prior held-out code does not become training data merely to reach a count. Do not confuse generic motifs such as reduction with proved shared implementation ancestry.
5. Balance actual mechanisms—memory layout, vector loads, reductions, attention tiling, quantized decode, launch/dispatch, fusion, synchronization—after review. Repeated revisions of one optimization should not supply most of the nominal 300–800 rows. Count distinct source changes, supervised tasks and independent lineages separately.
6. Run exact complete prompt-plus-answer token audits before admission. Train the first small balanced set within measured capacity; then increase length and count independently. Source-reported examples may be useful for SFT before reproduction, but `execution_verified` requires separately bound compile, correctness and benchmark results on the stated target hardware/workload.

This inventory establishes a concrete source pool and the context-extraction bottleneck. It does not establish that the requested corpus size, independent held-out performance or generated-kernel usefulness has already been achieved.
