# Review of the optional 80 GiB training profile

Date: 2026-10-12. Independent design/code review by the quality-research agent, updated after inspecting the parent-run benchmark and probe receipts. No GPU workload was executed by this reviewer. The original 8 GiB packed + 32 GiB decoded cache proposal below is a **historical prototype**, superseded by the measured choice here.

## Current outcome and defaults

The optional `development` preset now uses **80 GiB process budget, 32 GiB system reserve, 8 GiB packed cache, decoded cache off, and 256 MiB allocator cache**. The conservative `low-memory` preset remains the general default. Cache limits are components inside the process budget. The development preset still requires approximately **112 GiB free/reclaimable memory at startup**; its 32 GiB reserve has not been silently reduced.

**Chunk 512 remains the default numerical setting.** Chunk 8192 is an explicit, recorded choice that changes floating-point reduction order. It is not implied by the memory preset or a resource-only resume override. Cache parity below holds at the same chunk size; the measured 512 and 8192 output hashes differ.

The actual single-MoE-layer benchmarks support packed-only caching for this workload. Keeping decoded FP16 tiles was slower despite cache hits. These are observed single-sequence timings, not a repeated statistical study or a whole-model speed claim:

| EXL3 column chunk | Low cache: 128 MiB packed, decoded off | Packed-only: 8 GiB packed, decoded off | Historical prototype: 8 GiB packed + 32 GiB decoded |
| --- | ---: | ---: | ---: |
| 512 | 57.902 s | **53.728 s** | 62.789 s |
| 8192, explicit | 48.536 s | **44.103 s** | 54.692 s |

Each total is an original forward plus a separate forward-and-VJP call, representing immediate recomputation/backward reuse. It is not a backward-only timer. Packed-only components were 17.063 + 36.666 s at chunk 512 and 14.001 + 30.102 s at chunk 8192; displayed components and totals are rounded independently. Actual layer 0 weights, 5,808 synthetic normalized activation rows and 349 routed experts were used. Both packed-only cases matched the saved same-chunk low-cache output and dX hashes **bit for bit**; the decoded-cache cases also retained same-chunk parity. The packed cache retained the layer's used matrices and needed no additional packed loads in its forward-and-VJP call. See the [packed-only receipt](packed-cache-moe-benchmark.json) and [low/decoded-cache receipt](decoded-cache-moe-benchmark.json).

The packed-only comparison reused the preceding run's recorded low-cache hashes and timings; it did not rerun that baseline in the same process. JIT, filesystem cache and execution order may affect the timing difference. Comparing 44.103 s against the old 57.902 s is approximately 23.8% lower elapsed for this single MoE sequence, but changes both cache and chunk size; that difference must not be attributed wholly to caching or called a full-model speedup. No completed low-memory full-model update exists for an end-to-end speed ratio.

## Completed full-model update; first-update fit failed

The subsequent project `artifacts/ktune-20261012/full-parent-pilot-development`
completed **one optimizer update** on the complete 5,809-token stored example
(5,808 input positions, 811 supervised answer tokens). It used **80 GiB process /
12 GiB reserve / 8 GiB packed / decoded off / chunk 8192**. This is a third,
explicit reserve choice: the preset remains 32 GiB and the earlier synthetic
probe used 16 GiB.

The [saved checkpoint audit](ktune-full-parent-checkpoint-audit.json) records:

| Measurement | Result |
| --- | --- |
| Complete training elapsed time | 1,862.057 s (31.0 min) |
| Optimizer history elapsed time | 1,861.757 s |
| Peak MLX allocation | 73.948 GiB |
| End observed-process reading | 27.657 GiB; **not the process peak** |
| Step-1 masked training loss | 0.50125784; not an after-update quality score |
| Gradient norm | 0.03829980 |
| Saved adapter state | 160 finite tensors; 3,153,920 parameters; all 80 LoRA B tensors nonzero from zero initialization |
| Saved optimizer state | 322 finite tensors; Adam step 1; recorded file checksums match |

The reviewer read the actual training receipt and matched its SHA-256 to the
parent's CPU checkpoint audit. This verifies the reporting chain; the tensor
audit itself was run by the parent, not repeated here.

Fresh-process teacher-forced scoring has now **failed the first-update fit gate**:
base answer NLL was **0.5012578592107731**, adapter NLL was
**0.5053724668388994**, and adapter-minus-base was **+0.00411460762812621**.
Both conditions scored the same 811 supervised tokens; logits hashes differ.
The [combined CPU verification audit](ktune-full-parent-verification.json)
records `training_target_nll_improved=false`; a completed CLI run
must not be read as a fitting pass.
The fresh base score agrees with the step's pre-update training loss within
**2.24e-8**, supporting repeatable base scoring under this input/mask, not
independent full-model parity. See the [unaltered first-fit receipt](ktune-full-parent-first-update-fit.json).

The update and checkpoint remain an operational pass, but the original adapter
made its training answer less likely. Changed logits and finite saved tensors do
not establish correct or useful optimization. The failed trial is preserved;
all three tested derived inference strengths (+0.1, -0.1, adaptive +10) also
worsened NLL. The bounded series is complete and no adapter was promoted as an
improvement. The [aggregate](ktune-strength-diagnostics.json) binds scores and
exact derivations while verifying unchanged original checkpoint, fit and engine
source. The new native/head/sparse VJP checks pass and the current
suite passes 518 tests plus 27 subtests, but covered components are not an oracle
for the full-model gradient. See the [diagnosis update](first-step-diagnosis-update.md).
Derived artifacts must not overwrite the saved optimizer checkpoint or
retroactively turn this result into a fitting pass. No resumed-update parity, held-out benefit,
full-model reference parity or useful generated-kernel claim follows.

The parent's phase timing reports **584.2 s for this forward versus 835.3 s for
the failed low-memory run's forward**. Cache and chunk configuration changed;
there is no completed low-memory update for an end-to-end comparison. Do not
attribute this entire forward difference to packed caching or call it a
full-update speedup. **5,809 tokens is the largest complete training example
tested; 32,768 is an implementation ceiling, not a validated training length.**

## Measured final-block memory feasibility

The [standalone final-block probe](final-block-memory-probe.json) passed using actual layer 39 weights, 5,808 positions, rank-4 attention adapters and synthetic hidden/KV state. Evaluated retained padding brought live MLX memory to **17.95 GiB before the block**, approximating the capacity occupied by the other blocks. Its recorded peak was **58.76 GiB**. All inspected gradients were finite; LoRA B and upstream hidden/pre/KV gradients were nonzero. LoRA A gradients were zero as expected for zero-initialized B.

This probe explicitly used **80 GiB process / 16 GiB reserve / 8 GiB packed / decoded off / chunk 8192**. The smaller reserve was a recorded override for this run, not the preset default or evidence that a 32 GiB reserve preflight passed. Total script elapsed time was **52.7 seconds**, including model load, synthetic-state setup, padding and verification; it is not a backward-only duration.

This is useful evidence that the formerly failing final-block backward can fit under the larger guard. It is **not a complete optimizer update or proof of whole-model training feasibility**. Synthetic routing, a scalar surrogate loss, dummy retained allocations and local causal selections do not reproduce the preceding 39-block autograd graph, vocabulary loss or allocator fragmentation. The subsequent full update and checkpoint audit are recorded above. Fresh baseline/adapted scoring completed and the first-update fitting gate failed, as recorded above. Controlled full-model throughput comparisons and new-kernel quality remain separate gates.

## Original prototype assessment (historical)

An optional larger memory profile is a reasonable way to reduce repeated packed reads and EXL3 decoding while retaining the existing training algorithm. Keep the conservative default. The original proposal used an 80 GiB process guard, 8 GiB packed cache, 32 GiB decoded cache and 32 GiB system reserve, with these distinct meanings:

| Allocation or limit | Meaning |
| --- | --- |
| 80 GiB process budget | Includes cached weights, resident trunk, activations, gradients, optimizer state and transient allocations |
| 8 GiB packed LRU | A component inside the process budget |
| 32 GiB decoded FP16 LRU | A component inside the process budget |
| Approximately 40 GiB remainder | Available within the process limit for everything beyond those two fully populated caches |
| 32 GiB system reserve | Additional headroom, checked independently; current startup policy requires approximately 112 GiB free/reclaimable memory |

These are GiB, not decimal GB. They are caps, not allocations made at startup. MLX's limit and the watchdog are not a hard OS quota; a single allocation can temporarily overshoot. Larger caches do not eliminate the need to observe both process/MLX memory and system headroom. Do not silently weaken the reserve to make the profile start.

The actual checkpoint config has hidden width 5,120, expert intermediate width 2,304, 384 routed experts, top-k six and 40 layers. Its three routed expert matrices occupy **25.3125 GiB per layer in FP16**, or **67.5 MiB per expert**. At nominal 3 bpw their packed matrix payload is approximately **4.746 GiB per layer**, plus small scale vectors. Thus the proposed caches can each retain one complete routed layer with margin. This is more precise than the earlier approximately 27 GiB estimate.

## Prototype placement and correctness requirements

1. Put the decoded LRU in the shared `MatrixStore`, independent of the packed LRU. Putting decoded entries on disposable packed layer objects makes packed eviction discard the useful warm data. The complete key must bind the matrix, codebook and column slice `(start, width)` within a store whose checkpoint identity is fixed.
2. Check source fingerprints on hits as well as misses, including a decoded hit after packed eviction. Clear must release both caches. Report separate current/peak bytes, hits, misses and evictions. Byte accounting must count actual retained evaluated FP16 arrays; neither logical element counts nor requested limits are sufficient.
3. Evict before allocating a cacheable miss. A tile too large for its cache should bypass retention without unnecessarily evicting useful unrelated entries. Retained cache size excludes an active cast/tile and evaluation scratch, which must still fit the process guard.
4. Materialize cached FP16 output so a cache entry cannot retain the packed decode graph or an enclosing gradient trace. The decoder already returns a codebook value rounded to FP16 before the requested output conversion. Reusing those FP16 values, then explicitly casting to FP32, can preserve the existing values. Do not replace the FP32 matmul or change the 512-column chunk sequence in this optimization.
5. Test cache-disabled, cold, warm and eviction paths for forward and input VJP parity. Include both projection orientations, final short tiles, all supported codebooks/rates, source mutation, independent packed eviction, exact budget boundaries and complete clear. Verify the cache is absent from trainable parameters and checkpoint tensor exports.
6. Record the selected profile and resolved limits in every run receipt. Memory/cache settings are resource choices; chunk size and math semantics remain numerical contract fields. A cache setting should not silently alter a resumed checkpoint's optimizer/data contract. Source identity and parity evidence still need to accompany a new implementation.

## Expected reuse and remaining costs (pre-benchmark assessment)

The original forward streams through 40 layers, much more than either cache can retain. During backward, a block is recomputed and its MoE then evaluates component VJPs. A one-layer decoded cache can serve those immediate repeated expert projections instead of decoding each call again. The final forward layer may also remain warm into its backward; earlier layers generally need to be loaded again.

This favors the design but does not establish a particular speedup. Attention, the indexer, dense trunk matmuls, Hadamards, output projection, Engram reads and host materialization remain. Caching FP16 also leaves the per-use FP32 conversion and matmul costs. An oversized cache may hurt system pressure or filesystem cache residency. End-to-end wall time, not cache hit rate alone, decides whether the profile is worthwhile.

## Original benchmark plan and remaining full-model gate

The plan below preceded the measurements above. Benchmarks ran sequentially after the failed baseline released Metal and memory. A full-model comparison remains outstanding.

| Stage | Measurement | Acceptance |
| --- | --- | --- |
| Numerical fixtures | Existing decode/codebook tests plus cold/warm/evicted forward, dX and adapter-gradient checks | Same declared arithmetic and outputs; no stale-source hits, retained-graph growth or exported cache tensors |
| Real projection or expert | Actual checkpoint w1/w3 (5,120→2,304, final tile 256) and w2 (2,304→5,120); forward plus dX, with approximately 96 token rows and optional 1/512-row cases | Raw output/gradient comparison, cache counters, peak memory and synchronized timings; label this a projection/expert benchmark |
| Layer access pattern | One whole routed layer's representative expert access order, then recomputation/component backward reuse | Cache occupancy stays bounded and measured hit bytes support the claimed one-layer reuse; do not infer this from one repeatedly accessed matrix |
| Full-model update | Identical stored example, source semantics, seed, starting adapters/optimizer and loss masking; low versus high profile | Finite matching numerics under the existing tolerance, completed checkpoint, lower complete-update wall time, process/system guards respected |

At 5,808 input positions and top-k six, average routed rows per expert are about **90.75**, although the real distribution is uneven. This motivates a 96-row microbenchmark rather than only a single-token decode case. A single warm expert is a favorable reuse case and cannot substitute for the 40-layer update.

Synchronize/materialize results before stopping a timer. Report compile/setup time separately from timed steady execution. Distinguish an empty application LRU from a cold filesystem cache; do not purge the user's OS cache to manufacture a disk-cold result. Alternate paired test order where affordable and retain individual timings, including slow runs. Before/after counter deltas should distinguish original forward, recomputation and backward so reduced decoding can be connected to the measured wall-time change.

If only the projection benchmark is affordable, publish that limited result. If only one full update per profile is available, report it as an observed pair without a statistical or universal speedup claim. The already running baseline can provide useful feasibility evidence, but comparisons across different code revisions must disclose that difference rather than attribute every timing change to caches.

## Review of the staged implementation (historical inspection cutoff)

Subsequent read-only review inspected `streaming.py`, `linear.py`, `resources.py`, `sft.py`, `sft_cli.py`, `workflow.py`, `deepseek.py`, and Studio's Python/HTML code. **No remaining blocking defect was found in the reviewed supported path.** GPU numerical/cache tests and timing were still awaiting execution at this inspection cutoff; the subsequently inspected benchmark/probe receipts are recorded above. The CPU checks below do not substitute for GPU evidence.

The implementation meets the central cache requirements: an independent store-owned LRU holds materialized FP16 tiles; a working tile is cast to FP32 before the same matmul/chunk sequence; every decoded request checks source fingerprints; and cache hits survive packed eviction. A final hardening change binds the supplied base object to its MatrixStore and key, rejecting accidental cross-key/store cache poisoning. The `_base` loop explicitly releases its working FP32 tile before the next decode request. Cache counters distinguish retained FP16 payload from transient FP32 work, and layer progress labels counts as cumulative.

The parent found that an oversized tile initially evicted useful entries despite being too large to cache. The revised implementation bypasses the cache without evicting those entries. It evicts before allocating a cacheable miss. `clear()` releases both caches. Source-stat checks are mutation detection within this local frozen-checkpoint workflow, not cryptographic authentication of a hostile filesystem.

Resource limits are validated as finite numbers and integer MiB cache sizes; booleans, NaN, infinity, negative values and combined cache overcommit fail. At this historical cutoff the prototype development preset resolved to 80/32 GiB process/reserve, 8/32 GiB packed/decoded and 256 MiB allocator cache. The final preset above supersedes it by disabling decoded retention. A minimum workspace allowance prevents configuring all process memory as cache; it does not promise that arbitrary examples fit the remainder.

Resource-only resume overrides are excluded from the optimizer/math contract, while data, adapter settings, chunk size, implementation and dependency identity checks remain. Selecting a profile resets prior resolved resource defaults before explicit overrides are applied. Partial overrides preserve other saved fields. Each attempt records resolved limits. **This does not migrate an older project's source identity.** The parent preserved the running baseline's source snapshot and will use a new project for the new implementation rather than replacing a recorded identity in place.

Two concrete Studio issues found in this pass were fixed by its owner:

1. Opening an existing 80 GiB project previously displayed the form's 16 GiB defaults even though training would use saved 80 GiB settings. Studio now displays saved resolved resources and initializes the form on project change. Repeated polls preserve unsubmitted edits.
2. A partial Studio API resource override previously reset unrelated fields to low-memory defaults. Partial overrides now merge with the saved resource fields, and an empty override is a no-op. Explicit profile selection still chooses that preset.

The UI states that cached weights are inside the process limit, gives the 112 GiB startup requirement, and says speed depends on reuse. It does not promise a measured speedup. The benchmark script likewise labels single-matrix warm reuse as limited evidence, separates an empty application cache from OS/JIT state, synchronizes measured work, checks exact same-chunk cache parity, and labels cross-chunk arithmetic differences. Use chunk 512 for the unchanged-arithmetic cache comparison; the optional chunk sweep is a separate numerical experiment.

### Independent CPU evidence

- The reviewer ran the resource/profile, project and Studio suites: **96 tests passed** after the UI/precedence fixes. These tests use CPU mocks and do not launch a model.
- A separate NumPy-backed execution of the actual MatrixStore class AST passed **1,800 randomized tile requests over six cache capacities**, including exact LRU order, retained-byte accounting, bypass/clear behavior and source mutation rejection before a hit. This preceded the final additional store/key object-binding guard; that guard was inspected afterward. No MLX was imported by this check.
- The actual Studio `resourceView` JavaScript passed a Node mock check for saved high-profile display, preservation of pending edits across polls, and reset on switching projects.

The implementation agent added GPU fixtures for exact cold/warm/evicted forward and dX parity, the direct-decoder baseline, source mutation, every codebook entry, all supported rates and short final tiles. At this review's cutoff these were inspected, not executed by this reviewer. Parent-run results must be recorded separately.

### Baseline outcome and remaining gate

The low-memory baseline completed all 40 forward layers in approximately 835 seconds, at 17.95 GiB active MLX memory, then aborted during layer-40 backward recomputation at 32.34 GiB against its 32 GiB guard. **No complete optimizer update was produced.** These parent-reported measurements are failed-run feasibility evidence, not a low-memory full-update timing baseline. A complete speed comparison therefore requires subsequent controlled runs; a successful 80 GiB run alone cannot establish a full-update speedup ratio.

Measure actual idle reclaimable memory after the baseline exits. RSS and MLX allocations can overlap; adding both to current free memory is not a reliable estimate of how much will be reclaimed. The default startup gate remains 112 GiB. Any explicitly chosen lower reserve or smaller cache profile needs its own recorded settings and outcome.

**Historical prototype source hashes at the inspection cutoff.** These are not a claim that the later final preset has the same source identity; benchmark/probe receipts carry their own implementation identities, and the full-model project must record its final source identity:

| File | SHA-256 |
| --- | --- |
| `streaming.py` | `e60c77729a89fd5a9a76cd2641b5325c5ccf8bc0ec4b80a19ff949a1579d03df` |
| `linear.py` | `51ecb3cbc657199e86f4db13b92f1ea6cd13d4550a31ad35602b2e352b3f6379` |
| `resources.py` | `7d123724454417ee8c26d1ee0f2a50e4d869141fc0c2701f170337e507c83e0d` |
| `sft.py` | `dd063cbfe3f0047fdd14fd7042f8463426550dab3a6f956a6a2b0c3af6d7eb44` |
| `workflow.py` | `40371fb3fa90bd5f32c30d78363f02612cc9f0fbba03ccdefb68aed39aead945` |
| `studio.py` | `a29153eece331ad5e5bb44865b779d4f3fe526f8107b82b522c3df91d8567b9c` |
| `studio.html` | `8c956ee12c960f25ce8881dedc32c9031623c58342dc6453a49233026064d3f5` |
| `sft_cli.py` | `0722a0448050db1d0e1a4261d71e592e4bf0024d36c06cdf44453aaea9c17562` |
| `deepseek.py` | `8ab19c61590f390d0fd1a9e72229b72f35332cb838244dd83c1446acb9f1073b` |
