# Can this tool teach better Apple silicon Metal optimization?

Assessment date: 2026-10-12. Target: better model-generated inference-engine
optimizations, measured by correctness and actual speed/memory results.

**The idea is credible; this trainer is not ready for the full task today.**
It can perform short-context supervised adapter updates on the supported EXL3
DeepSeek repack. It cannot yet ingest repository history, train on realistic
multi-thousand-token optimization examples, learn tool-use trajectories, or run
an execution-based Metal quality evaluation. The verified codeword experiment
establishes none of those abilities.

## A measured context mismatch

I inspected a bounded read-only sample from the local
`/Users/jasontitus/experiments/mlx-stream` repository. Selection used the newest
120 non-merge commits and optimization/performance-related message keywords,
then the first 20 with nonempty code diffs. These are candidate commits, not
certified successful optimizations. The exact selection, commit IDs and lengths
are in [the corpus audit](metal-corpus-fit.json).

Using the actual DeepSeek tokenizer and chat template, a minimal prompt plus
commit message and code diff needed a median **5,319 training positions**.
**19 of 20 exceeded 512**; the minimum was 188 and the largest bundled change
645,223. Message-only length had a median 209.5 tokens, which explains why raw
logs can appear to fit while omitting the code needed to teach the task.
The size probe omits the original surrounding code, hardware and profiling
context. It is not a proposed training objective or an estimate of the eventual
curated corpus; extracting complete independent functions/changes can shrink it.

Current support is at most 512 input positions, with only much shorter runs
measured. This is an architecture limitation involving sparse attention/indexer
selection, not a configuration knob we can safely raise. Generating a realistic
patch also needs enough output room. Full diffs or agent trajectories cannot be
made valid by cutting them into arbitrary 512-token chunks.

## What git history can teach

| Training material | Expected value | Main limitation |
|---|---|---|
| Commit messages alone | Vocabulary, repository conventions, concise explanations | Omits the decision context and the code transformation; may mainly improve commit-message imitation. |
| Parent code plus successful patch | Recurring code transformations and conditional optimization patterns | Needs enough context, verified success and source-separated evaluation. |
| Parent code, workload and profile plus patch and rationale | Better bottleneck diagnosis and selection of an appropriate optimization | Commit prose may not contain measurements or pre-change evidence; missing evidence must stay missing. |
| Tool calls, compiler/test feedback, revised patch and final measurements | The iterative workflow of testing and revising optimizations | Structured tool messages/agent trajectory training are currently rejected by this importer. Plain-text transcripts are not equivalent to validated tool-use training. |

My engineering judgment: curated examples could improve familiar optimization
patterns and repository-specific habits. Novel fast-and-correct Metal code is
plausible but unproven; I would not promise a likely net improvement from raw
logs or from this current port. A large base model may already know the generic
techniques; a small repetitive corpus may teach verbosity, overfit one engine,
or reduce correctness without adding useful optimization ability.

There is relevant evidence, but not a direct result for our setting. PIE studies
fine-tuning and retrieval from human performance-improving C++ edits with tests
and controlled execution measurement. It supports trying the approach; its
reported gains cannot be transferred to Metal, this model, or our quantization.
[Learning Performance-Improving Code Edits](https://arxiv.org/abs/2302.07867)

SWE-smith demonstrates training software-engineering agents from executable task
instances and trajectories. Its Python repair setting is different from GPU
optimization and does not validate training on commit messages alone.
[SWE-smith](https://arxiv.org/abs/2504.21798)

## Build examples as optimization cases, not raw log continuation

For each coherent change, preserve:

| Location | Contents |
|---|---|
| Prompt | Parent-version code plus required interfaces, target chip/OS/compiler, tensor shapes/dtypes/layouts, prefill/decode/batch scenario, numerical constraints, and available pre-change profile or baseline timing. Ask for an applicable patch and concise explanation. |
| Desired answer | The successful patch plus the recorded reason and a short validation procedure. Include observed results only when genuine evidence exists. Do not invent profiler observations or hidden reasoning. |
| Sidecar provenance | Repository, parent/child hashes, source files, change family, measurement commands, correctness receipts, timing distributions, chip, build flags, memory result and any later revert/fix. |

Only information available before the optimization belongs in the input. A
commit message saying exactly how the problem was fixed is usually an answer or
label, not a legitimate evaluation prompt. Keep measured post-change results
out of the prompt when testing whether the model discovers the optimization.

Reconstruct parent and child in isolated checkouts when validating a case. Do not
check out historical revisions in the user's active tree. A merged commit or a
subject containing “faster” is a candidate, not proof of speedup. Separate changes
that combine refactoring, accuracy changes and performance work. Label whether
an improvement affects kernel time, end-to-end latency, throughput or memory.
A microbenchmark gain that slows the inference engine is not a general success.

Successes are useful positive SFT data. Also preserve unsuccessful attempts and
regressions in the corpus ledger and benchmark. They expose situations where a
technique is harmful. Do not supervise their bad patches as desired answers;
verified alternative pairs could support a later preference objective, which is
not implemented in this tool.

Split by optimization family and code lineage before making multiple examples
from a commit. Keep sibling hunks, cherry-picks, reverts, fixes, copied kernels
and derived explanations in the same split. A chronological split alone may
still leak repeated kernels. Keep final-test solutions out of retrieval too.

## How to determine whether fine-tuning helped

Create an initial 20–30 task feasibility panel of independently held-out
optimization cases, then expand it if results justify the cost. That count is a
pilot proposal, not a guarantee of statistical power. Reconstruct each parent
state, hide the successful patch and post-change explanation, and supply the
same usable context and constraints to every evaluated condition.

Compare four conditions when context/runtime support exists: base, base with
retrieved training examples, adapter, and adapter with the same retrieval policy.
A retrieval baseline tests whether access to the history already provides the
benefit without weight changes. Give every condition the same tool permissions,
number of candidates/revision attempts and execution budget. Report one-shot and
fixed-budget iterative results separately; do not compare one base attempt with
many adapter retries.

Score executed artifacts rather than explanations or similarity to the historical
patch. Alternative implementations may be better than the recorded commit.
KernelBench similarly combines correctness with measured speedup; it is a method
to adapt, not an Apple Metal benchmark we have already run.
[KernelBench](https://arxiv.org/abs/2502.10517)

1. **Build and contract:** patch applies to the parent, builds, preserves required
   interfaces and does not edit evaluation tests, timers or reference outputs.
2. **Correctness:** independent numerical references and predeclared tolerances;
   varied random and adversarial inputs; tails, non-multiple sizes, zero/large
   values, shape/dtype boundaries and quantization rules. Use available Metal
   API/shader validation separately from performance timing.
3. **Performance:** measured on the intended Apple GPU, with shader compilation
   and warmup separated from steady-state timing, explicit synchronization,
   repeated interleaved baseline/candidate runs, and controlled competing load.
   Record uncertainty, thermal/power state, kernel time, end-to-end prefill/decode
   metrics and peak memory. Re-profile instead of assuming a CUDA tactic applies.
4. **Primary result:** fraction of tasks producing a correct patch whose speedup
   exceeds a predeclared practical threshold and measurement noise, within the
   same candidate budget. For example, 5% is a possible project threshold to
   validate against the noise floor, not a universal standard.
5. **Failure accounting:** invalid or wrong patches fail regardless of apparent
   speed. Report compile/correctness rates, correct-but-slower cases, regressions,
   memory growth, geometric speedup on correct cases and total task coverage.
   Do not drop failures from the denominator and publish only winning timings.
6. **Transfer:** hold out at least some kernel families and shapes. A gain on M1
   Ultra does not establish a gain on M4/M5; label hardware coverage explicitly.

Apple's profiler exposes occupancy, bandwidth and limiter information, and its
concurrent versus serial replay modes have different timing implications. Those
controls and validation tools are appropriate evidence for the benchmark.
[GPU profiling](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance),
[Metal developer workflows](https://developer.apple.com/documentation/xcode/metal-developer-workflows)

Use paired comparisons grouped by independent task/family. A confidence interval
and task-level results matter more than a lower training loss. The held-out cases
must not become an iterative target for prompt, learning-rate or checkpoint
selection; use development cases for that work.

## What to build next for this specific use case

1. A Git-history importer with a candidate ledger, parent/patch extraction,
   provenance, measured-success status, and a token-length report. It should
   propose coherent examples for review, not blindly label commits as successes.
2. A benchmark-first workflow on selected small Metal/host-integration tasks.
   Reuse the engine's existing correctness and performance harnesses where they
   are independent of the candidate patch. Build the retrieval baseline from the
   training split before investing in a large fine-tune.
3. A longer-context training and inference path with independent model parity.
   Determine the required window after curating actual cases; many minimal sampled
   examples already need several thousand tokens. Larger GPU memory limits alone
   do not implement the missing attention/indexer behavior.
4. An adapter-enabled runtime with enough generation context and a bounded
   compile/test/profile loop. Export/serving correctness is part of the product,
   not an optional step after training.
5. Then a small SFT pilot on verified cases, followed by the held-out execution
   comparison. Use the present trainer only for genuinely self-contained examples
   that fit its measured scope; label that as a narrow pilot, not the full goal.

If improving Metal work soon is the priority, corpus curation plus retrieval and
execution feedback is the first useful deliverable. Longer-context EXL3 training
can proceed as an engineering track with its own correctness gates. No Metal
corpus training or execution-quality benchmark was performed in this assessment.
