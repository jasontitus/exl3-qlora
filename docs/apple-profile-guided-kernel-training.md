# Teaching profile-guided Metal optimization

The intended capability is: inspect a workload and its measured profile, choose a justified kernel or dispatch change, implement it, test it, and revise the decision from actual feedback. Correctness is necessary; the goal is faster execution on the requested workload. Current edit-training and affine results do not establish this capability.

This document extends the [active experiment plan](apple-cache-aware-experiment-plan.md). The152-item trial is complete; this work does not change its data or authorize an automatic training extension. Source collection remains paused. Profiling fixtures are preparation, not measured examples, until their GPU receipts exist.

## One training item is a decision with evidence

Use ordinary assistant-target SFT on standard chat/tool trajectories. The learning algorithm can remain the same; the content and evaluation must change. A complete item contains:

| Part | Contents | Evidence boundary |
| --- | --- | --- |
| User task | Operation, baseline kernel and necessary host/dispatch code, hardware, precision/tolerances, shapes/strides/alignment, representative workload weights and optimization objective | Actual available context; no answer or future optimized timings |
| Baseline tool result | Timing samples and method, launch counts/host overhead where measured, bytes or work estimates with formulas, available profiler metrics with units and source | Separate measured counters from estimates; unavailable metrics stay unavailable |
| Assistant decision | Evidence cited, bottleneck hypothesis, one proposed change, expected effect, correctness risks and a discriminating test | Concise decision record; hypothesis is not a measured causal fact |
| Assistant artifact | Complete edited kernel and dispatch or a validated agent edit | Supported agent-facing format, exact output, no hidden repair |
| Tool feedback | Actual compiler, numerical and timing results for that candidate, including failures and regressions | Produced by the trusted harness, not by the model |
| Assistant response | Accept, revise, retain baseline, or request another measurement, with reasons tied to the feedback | Failed attempt is context; accepted solution must be validated before labeling it successful |

The first practical version uses the existing prompt/completion or plain-chat route: baseline code and structured measurement reports in the user input, followed by the final assistant decision and complete implementation. For feedback examples, include prior code and actual harness feedback as clearly labeled context, with one next decision as the supervised target. This lets us start without changing the training math. It is not a claim of native tool-role trajectory support.

Source inspection confirms that current data.py uses one loss_start and sft.py supports last_assistant loss only. tool_data.py supports a terminal tool action but explicitly rejects tool observations. Therefore a full native multi-tool conversation is not currently a drop-in dataset format. A later serializer can emit one complete-context example per assistant decision, preserving actual native tool roles and supervising only its final answer; it needs template, token-boundary and mask tests outside runtime-v5. Training all assistant turns in one sequence would need multiple target spans. Keep prompt/profile/measurement tokens outside the targets and never claim that changing JSONL alone enables this. The completed 152 run and its data remain unchanged.

Save raw machine evidence separately with source/measurement hashes. Export the readable summary as standard JSONL messages with the exact deployed tool contract, then record tokenized length and target coverage. Do not silently truncate rationale, code or feedback. No public model's private reasoning trace is needed: a short explicit engineering decision is the target.

## Teach decisions that can change with the evidence

Start with a small set of measured pairs, not many fabricated profiles:

- The same operation on tiny versus large inputs, or short versus long rows. Determine experimentally whether launch overhead, serial work, memory traffic or another factor dominates. A fixed transformation should not be labeled universally best.
- Aligned contiguous versus odd-stride inputs. A vectorized path may be useful only when its preconditions hold; otherwise use a safe path or guarded dispatch.
- A promising transformation that fails numerical checks or makes the target workload slower. Teach interpreting that feedback and revising or keeping the baseline.
- An already efficient implementation or an insufficient profile. The correct action can be no change or a targeted measurement, rather than inventing a bottleneck.

These are task designs, not observations about our current checkpoints. Do not manufacture hardware counters from commit prose, assign a bottleneck from latency alone, or paste one workload's profile onto another workload and call it measured. Keep hardware-counter names and normalization specific to the actual Apple tool/device. Basic latency, launch/workload sweeps and controlled before/after measurements can support useful decisions even when detailed counters are unavailable. A timing sweep alone does not prove bandwidth saturation or register spilling.

Use matched workload variants whose best actions differ in real measurements. Group all variants, source lineage, explanations and repair traces together when splitting training and evaluation. Synthetic hypothetical profiles may test a parser or an explicit reasoning exercise, but cannot count as measured optimization training/evaluation evidence.

## Where the first examples come from

The bounded existing-library inventory screened24 changes, of which22 across20 known lineage groups were eligible for further validation. Its inspected receipt set contains no locally numerical-and-performance-qualified training-ready pair. Test-file edits and source-reported speedups remain useful provenance, not local outcomes. This is a statement about that shortlist, not a claim that every collected source lacks evidence.

Choose two to four feasible optimization cases first, alongside a few correctness cases. Candidates that skip unnecessary expert/GEMM work or change workload-dependent dispatch may be more relevant to this goal than correctness-only empty-reduction examples, but their host/build closure and measured speedups still need verification. Do not fill a16–32 target with cases that cannot support the lesson. Preserve existing lineage exclusions and protect dev41/protected16 plus the current evaluation tasks/oracles.

Use both machines: keep Studio on adapter generation, correctness and timing comparisons, and use the M5 Max for bounded profiling/verification of separate training cases when its training lease is released. The user explicitly authorized M5 profiling on October 15. The observed M5 has 40 GPU cores,128GB RAM,Metal4 and an available Metal/Swift toolchain. Check live ownership and fresh memory admission before each job; an idle observation is not a reservation. Keep one GPU job per host. Measure each baseline/candidate pair on the same host and record device,OS,compiler,workload and source identities. Never pool raw M5 and Studio timings or imply one chip qualifies another; record regressions on each. Capture baseline and candidate under the same conditions, including workloads where the change is unhelpful. Do not change the evaluation sources into training examples after seeing model failures. New source collection stays paused; additional measurements on already collected or separately authored training cases are distinct work.

The first M5 profiling pair is the existing Q6_K batch-reuse change (row48). Independent audits accepted all nine tiny cases and six larger cases, including exact packed-weight oracle checks and bitwise-equal before/after outputs. The tiny cases were slower after the change (paired before/after ratios 0.790–0.958). Larger cases measured 1.077/1.617/1.738× at 512×4096 and 1.107/1.342/1.843× at 4096×4096 for batches 1/3/4. Each used four warmups and twenty paired rounds on M5 Max, separate from the RMS adapter evaluation policy. The larger sweep is exploratory and also changes synthetic values/scales: it does not isolate shape as the cause or prove a hardware bottleneck. All fifteen variants are one source lineage, not fifteen independent changes.

These receipts can ground both an apply-change example and an observed-regression/keep-baseline feedback example. Packaging and source/context review remain required before training. Preserve the M1 Ultra author-reported measurements separately; no raw timing pooling, counter inference or engine-throughput claim. This is one locally validated pair, not a completed multi-family training pilot.

The first two-row draft also exposed a data-construction pitfall during review: the purported initial-decision prompt included the candidate's eventual timing. That is valid for an explicitly labeled post-test selection example, but leaks the answer for a baseline-profile → proposed-change example. Preserve the original draft and require a baseline-only revision; keep after-results in external receipts until a feedback turn. An initial proposal should predict a benefit as a hypothesis, not claim an unobserved measured success. No draft is activated merely because tokenization passes.

Curated explanations may be drafted from source and measurements, then reviewed for unsupported claims. A stronger model can help draft a hypothesis, but its confidence is not a substitute for the compiler, numerical reference or timer. Keep source-stated, inferred and experimentally supported claims distinguishable.

## Evaluate the behavior and the outcome

Use two small evaluation modes on separate development cases:

1. **One-decision mode:** provide baseline code, workload and pre-change profile. Require a concise decision and implementation before revealing candidate results. Run correctness and matched timing. Score useful faster outputs under the same generation budget for base and adapters.
2. **Feedback mode:** allow a fixed small number of edit/test rounds with the same profiling information, tool-call budget and generation allowance for each model. Record best verified speedup, attempts to first useful result, wall time/token/tool cost, regressions and all rejected attempts. Keep this score separate from one-decision results.

Use the existing compiler/numerical/timing acceptance policy. Speed must beat the supplied implementation on the predeclared target workload and hold up on the declared shape mix; report every regression. Preserve end-to-end latency when fusion or launch/dispatch overhead is the objective. A faster isolated kernel does not automatically improve engine throughput.

Add a small **profile-use check** after the first functional results: compare the selected adapter and base on matched workload variants, and on a profile-present versus profile-omitted version with the same code/constraints and allowance. Keep a no-profile result distinct; do not reuse it as if its prompt matched. We want better measured decisions when useful evidence is supplied, not merely different prose. Use actual, internally consistent profiles; never swap incompatible observations to construct fake hardware evidence. This bounded check need not block the first ready RMS comparison.

Review the concise decision separately for groundedness: does it cite available measurements, respect constraints, acknowledge uncertainty and propose a test that distinguishes its hypothesis? Do not score reasoning by verbosity or an LLM judge's preference alone. Alternative correct approaches may win; there is no need to match one canonical explanation. Plausible reasoning earns no performance pass if the kernel is wrong or slower. Even profile-sensitive successful behavior is evidence of useful decisions, not proof of an internal reasoning mechanism.

## Training and cache implications

The next pilot should jointly target correctness and optimization decisions. Retain verified code as a major target; a rationale-only dataset risks teaching explanations without executable skill. Keep the first trial to standard supervised fine-tuning. Measured preference pairs or execution-reward training may become useful later, after enough trustworthy outcomes exist, but no new RL/DPO stack is needed to start.

Adding profiles, reformatted code or feedback changes full token rows and requires new frozen-prefix work. Do not relabel current caches as compatible. Prepare a few complete representative items, measure cold and warm cost, then decide the small pilot size. Repeated experiments on an unchanged pilot can reuse its own correctly bound cache. Broader rank/depth trials remain conditional on practical cost and output evidence; no claim that profile-rich data guarantees transfer.

## Research grounding

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) provides the correctness-plus-speed evaluation pattern and iterative kernel evaluation infrastructure. [KernelPro](https://arxiv.org/abs/2606.26453) describes using profiling tools to turn metrics into bottleneck guidance inside an iterative optimizer. These support the design of the feedback loop; their CUDA results do not establish Apple performance or prove that our proposed SFT will learn the skill. The design here separates demonstrated inference-time tooling from the still-unproven benefit of training on those trajectories.

Apple’s [GPU performance timeline](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-a-visual-timeline) documents occupancy, limiter and bandwidth views, and its [bandwidth guide](https://developer.apple.com/documentation/xcode/measuring-the-gpus-use-of-memory-bandwidth) describes measurement sources. Use the actual supported measurements on the target Mac; this review did not capture counters or establish a headless counter-export integration.
