# Comparing adapters by generated Metal kernels

The purpose is to choose the next useful adapter and training direction from executed code. Success means more correct, measurably faster kernels than the base model produces under the same practical allowance. Teacher-forced loss remains a diagnostic. The [active outcome-driven plan](apple-cache-aware-experiment-plan.md) supersedes the old fixed learning-rate/rank/depth sequence.

Use the [execution guide](apple-kernel-execution.md) for the runnable example and captured-response/backend flow, and the [suite schema](kernel-eval-suite-schema.md) to compare adapters. The CLI currently imports captured generations; live generation still uses the existing reviewed serving controller. The [actual comparison](../apple/reports/kernel-eval-panel-v1/actual-v3/comparison.md) preserves all three affine outcomes and the pending second task.

## Current evidence and scope

The existing affine/ReLU/row-sum harness has run curator controls and a generated candidate on the Mac Studio. Its fixed 224 numerical cases caught 12 out-of-tolerance values in nine cases in the old 458-update adapter's mechanically extracted kernel. Both the old adapter and base failed strict patch replay. The step-104 lower-rate adapter also failed strict replay and its extracted code was refused for vector alignment. These failures remain evidence; extraction is never counted as raw edit success.

The final-188 lower-rate adapter completed 475 output tokens with EOS, but its exact replay failed hunk counts (declared 5/31, observed 5/20). The separate 152-example code-focused trial was live at 77/152 updates at the last check. Rank-8 serving has built and passed static review but is not yet GPU-qualified.

The first two task families are fused affine/ReLU reduction and row RMS statistic. RMS statistic returns one value per row; it is not full elementwise RMSNorm. Its controls passed after a per-case autorelease-pool fix: all 896 dispatch outcomes matched the original run, with 0.498 GiB peak aggregate RSS under the unchanged 2 GiB guard. The earlier resource failure remains preserved. No adapter has completed that task in the current comparison. Two related reduction tasks are a useful smoke panel, not broad evidence of kernel-optimization skill.

## Evaluation contract

Each suite fixes the task source, prompt, input domain, alignment, output ABI, tolerances, workload shapes, compiler flags, sampling settings, token budget and attempts per condition before generation. The same task and generation allowance apply to base and each adapter. Reuse an existing result only when those identities match. Record host and resource policy changes separately; generator runtime differences prevent a clean attribution to the adapter alone.

A model produces an agent-facing edit against the supplied source. The runner parses and replays it without executing model-generated shell commands. The execution backend is trusted harness code. Every scheduled attempt keeps its outcome, including malformed patches, token caps, source refusals, compiler errors, numerical failures and operational interruptions.

The stages are:

1. Generate or import the exact captured response with model/adapter identity and generation settings.
2. Replay the edit against the fixed baseline. Preserve the raw response and patch.
3. Validate the task ABI and supported dispatch configuration, then compile the reconstructed source.
4. Compare with the independent numerical oracle across fixed shapes, odd tails, alignment cases, adversarial values, and guard/input-immutability checks.
5. Only for numerically accepted candidates, warm both implementations and run interleaved paired timing on identical workload/precision/device settings.
6. Produce per-task outcomes and an adapter comparison with coverage and unresolved slots visible.

A refused launch, missing result or timeout in infrastructure is not a numerical failure and is not a pass. A candidate's compile or numerical failure is retained in the denominator of completed model attempts. Mechanical extraction or hunk-count normalization is a separately labeled diagnostic track; it cannot improve the strict score.

## What the comparison should show

The goal measure is usable faster kernels: valid edits, successful compilation, all required numerical cases passed, and repeatable speedup over the supplied implementation on the declared target workload. Compare the number of those outcomes across base and adapters under the same budget. Report counts, missing coverage and workload regressions. Correctness is a prerequisite; do not average speed over incorrect kernels or hide difficult tasks by dropping missing outcomes. The current product report ranks strict numerical usability and labels latency ratios descriptive; until qualified timing exists it cannot declare an optimization winner.

For the practical first pass use one deterministic attempt per task/adapter. Expand to several fixed-seed attempts only when results justify the extra generation cost; keep the same allowance across candidates. Compare the base, old458, final188 and code-focused152 candidates on the same ready panel. Intermediate104 remains diagnostic history, not a required additional run.

Recommendations should follow observed failure modes:

- Validity failures across adapters: improve edit-format training or the declared output protocol before spending on more layers.
- Valid edits but numerical failures: prioritize correctness/precision examples and verification feedback.
- Correct kernels but no speedup: prioritize workload-specific bottleneck and performance examples, then test capacity if worthwhile.
- A candidate yields more correct, faster kernels: continue that training direction and check a second family before broad claims.
- Incomplete or tied results: report no winner; select the next inexpensive discriminating task rather than repeating an entire baseline campaign.

## Expanding the panel

After the two reduction tasks, add distinct families: vectorized layout/copy with odd strides, softmax, elementwise fusion, GEMV, quantized decode-plus-matvec, and a small attention operation. Each addition needs a working baseline, independent oracle, positive control and deliberately wrong controls. Existing successful controls should be reused by exact identity rather than rerun for every adapter.

New task source is evaluation material, not renewed repository mining. Keep it out of training while comparing these adapters. Concepts can overlap pretraining or the training corpus; do not describe curator-authored tasks as proof of unseen-domain generalization.

## Resource and operational policy

M5 trains while the Studio evaluates when resources admit. Each GPU phase uses the existing shared lease and memory guards. Cache reuse is permitted only under matching bindings. Source collection stays paused and Qwen stays parked. The current Studio profile adjustment preserves the 32 GiB system reserve and the exact model/prompt/generation limits.

## Related evaluation designs

[KernelBench](https://github.com/ScalingIntelligence/KernelBench) combines correctness and performance through its `fast_p` metric: the share of tasks whose generated implementation is correct and faster than a chosen threshold. Its [evaluation guide](https://github.com/ScalingIntelligence/KernelBench/blob/main/EVAL.md) also tests input mutation and other adversarial evaluation behaviors. We use the same correctness-before-speed principle, with Apple-specific execution and explicit incomplete coverage. Our current report does not claim a qualified `fast_p` score until matched timing evidence exists.

[MetalBench](https://github.com/Lazarus-931/MetalBench) and [METAL-SCI](https://github.com/vicgalle/metal-sci-kernels) are relevant Apple-kernel evaluation references for later backend/task integration. They were inspected as framework references, not ingested as new training data or executed in this work. The initial implementation reuses our already-qualified Metal runner rather than introducing another dependency or assuming external benchmark results apply locally.

## Profile-guided decisions

The [profile-guided design](apple-profile-guided-kernel-training.md) adds baseline profiling evidence, a concise testable hypothesis, matched workload variants and a bounded feedback mode. Report correct speedups and cost-to-result separately from whether explanations cite the evidence. The current captured-output report does not yet score profile use or run a multi-turn optimization agent. Build those additions around the existing gates; they are not prerequisites that postpone the next ready RMS comparison.
