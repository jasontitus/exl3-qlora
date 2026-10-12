# Quality verification plan for the Apple EXL3 trainer

Research date: 2026-10-12. Scope: the experimental MLX trainer in `~/experiments/exl3-qlora/apple`, the local 3.0-bpw DeepSeek V4.1 streaming repack, and short text contexts. This report proposes tests; it does not claim that the proposed tests have run. No GPU workloads were started for this research.

**A changed answer proves that an adapter influences this implementation. It does not prove that the implementation matches DeepSeek, that its gradients are correct, or that the adapter improves useful tasks.** We should publish separate results for each claim.

## What existing evidence establishes

The local `RECIPE.md` reports 111 passing tests, real updates, exact Adam/RNG continuation, adapter reload, and low-memory execution. The inspected suite already compares EXL3 forward/dX with a float64 reference, checks LoRA gradients with finite differences, tests token-weighted accumulation, and compares block recomputation gradients. Those are substantial component and workflow checks. The recipe explicitly leaves independent full-model logit parity and useful task quality unverified.

The current codeword experiment is an appropriate next causal check: base → trained adapter → adapter disabled → adapter reloaded, using identical token IDs and decoding settings. Its two training prompts and one held-out paraphrase are a fitting demonstration, not a statistical generalization result.

## How established practice maps to this tool

| Evidence type | Established practice | Application here |
| --- | --- | --- |
| Custom backward correctness | Compare analytic derivatives with independent references and numerical differences. PyTorch warns that low precision and nondifferentiable points complicate gradcheck. [Gradcheck documentation](https://docs.pytorch.org/docs/main/generated/torch.autograd.gradcheck.gradcheck.html) | Use tiny NumPy float64 reference operators; compare Metal outputs and activation/LoRA gradients. Test the actual custom VJP separately from the forward kernel. |
| Training objective | Completion-only and assistant-only loss are explicit SFT settings, dependent on the data/template. [TRL SFT documentation](https://huggingface.co/docs/trl/sft_trainer) | Record exactly which answer and EOS tokens contribute; compare manual shifted cross-entropy against the training and evaluation paths. |
| Language-model likelihood | Perplexity depends on tokenization and available conditioning context. [Transformers perplexity documentation](https://huggingface.co/docs/transformers/v4.43.0/en/perplexity) | Compare base and adapter with identical tokenizer, context, masks and token weighting; label answer-only perplexity as conditional answer perplexity. |
| Reproducible model evaluation | Prompting/setup changes can materially change results. [EleutherAI evaluation paper](https://arxiv.org/abs/2405.14782) | Store rendered prompts, token IDs, checkpoint hashes, task versions, decoding parameters and per-example predictions. |
| Useful quality | Evaluate several dimensions rather than relying on one aggregate score. [HELM](https://arxiv.org/abs/2211.09110) | Separate task accuracy, output validity, generalization, regression and runtime/memory. |
| Open-ended assessment | Model judges have position, verbosity and self-preference biases. [MT-Bench/Chatbot Arena paper](https://arxiv.org/abs/2306.05685) | Prefer deterministic graders; for prose use blinded pairwise human review. An optional model judge needs swapped answer order and human spot checks. |

QLoRA itself evaluates task outcomes as well as training behavior, and its authors warn that chatbot benchmarks can misrepresent performance. Its NF4 implementation is evidence for an evaluation approach, **not an EXL3 format or numerical oracle**. [QLoRA paper](https://arxiv.org/abs/2305.14314)

## Proposed staged gates

These acceptance criteria are project proposals, not thresholds mandated by the cited sources. Pin criteria before measuring the final test set.

### 1. Independent numerical correctness — highest-priority missing gate

- Capture reference outputs using an independently implemented inference path **with these same packed weights**, tokenizer, positions and short-context settings. Comparing against an unquantized checkpoint mixes port error with quantization error. Capture input IDs, logits, selected attention/Engram states and MoE route IDs; record the reference code revision and model hashes.
- Start with three fixed short prompts, including repeated tokens and a multi-token prefix. Stop at the first divergent layer. Compare relative L2 error, maximum absolute error, KL divergence, top-1/top-k and target log probabilities; a matching single greedy token is insufficient.
- Require exact unpacked bitstream/code indices and discrete routing where inputs are separated from route ties. Retain the existing component tolerances: forward/dX FP32 `atol=rtol=2e-5`, FP16 `2e-3`, BF16 `2e-2`; existing LoRA dense-gradient comparison `rtol=2e-4`, `atol=2e-6`.
- Do not transplant those local tolerances onto a 40-layer logit comparison. Establish the reference's own repeatability and dtype/kernel error envelope first; freeze per-layer and final-logit limits before judging the port. A mismatch requires a documented cause, not a tolerance widened until it passes.
- Add randomized directional derivative checks on small differentiable blocks, covering nonzero A and B, shared KV gradients, multiple routed experts and chunk/cache variants. Keep route choices away from top-k boundaries. FP8/FP4 straight-through estimators must be compared against an independent implementation of the **declared surrogate derivative**; ordinary finite differences of quantized forward values cannot validate an identity STE.
- A zero gradient for A at zero-initialized B is expected. Require B to receive a gradient first and A after B becomes nonzero. Do not “fix” this expected initialization behavior.

MLX custom functions explicitly replace autodiff transformations when custom VJPs are registered, so forward success does not establish backward correctness. [MLX custom-function documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html)

### 2. Optimizer and pipeline correctness — inexpensive fixtures

Preserve the existing successful tests, then fill gaps with small fixtures rather than repeating expensive full-model runs:

- Hand-compute masked shifted loss with unequal answer lengths; require training, evaluation and accumulation to match the same global token-weighted objective.
- Check one Adam update against an independent NumPy calculation, including clipping and zero-gradient leaves. Compare activation, A and B gradients, not just loss.
- Snapshot frozen **contents** in small fixtures. Object identity alone does not prove immutable values; for the full model use read-only base files plus sampled content fingerprints and exact zero-adapter/base logits.
- Verify zero-learning-rate and disabled-adapter controls, checkpoint/recompute on/off equivalence, repeated invocation order independence, and cache eviction/chunk-size invariance.
- Keep exact uninterrupted-versus-resumed comparison on this pinned Mac/runtime, including optimizer step/moments, RNG and sample cursor. Cross-version/device results should use documented tolerances, not promise universal bitwise identity.
- Negative cases: corrupted adapter/optimizer, wrong base/tokenizer/template, unsupported context, all-answer truncation, invalid token IDs, interrupted save and absent final manifest must fail clearly before another update. Verify the completed previous checkpoint remains loadable.

### 3. Causal behavior change — current run

Pass only if the chosen trained prompt reaches the exact intended answer **and EOS**, its answer loss falls, greedy output differs from base, disabling the adapter restores base logits, and reloading the saved adapter reproduces trained logits/output. Preserve every attempted step and the held-out wording result. Confirm the saved adapter through the public CLI in a fresh process.

Add two cheap controls after the run: an unrelated short question and a different-code instruction. Answering `violet` to everything is a regression, not broad successful learning. This fixture should not become the permanent output-quality benchmark.

### 4. Useful task improvement and regression — staged inference cost

Choose one real task before training, such as short structured extraction, concise classification, or a domain response format. Prefer deterministic exact-match, field-level accuracy or JSON/schema validation over subjective prose scores.

1. Split by source/document/template family **before** tokenization; remove exact duplicates and inspect near-duplicate families. Keep train, tuning validation and final test separate. Selecting checkpoints using a held-out example turns that example into validation.
2. Run a **12-case smoke panel** (four in-domain, four unseen phrasings/entities, four unrelated controls). This detects broken templates, answer collapse and prompt leakage; report raw counts, not a quality claim.
3. Expand to a **32-case fixed pilot** with preassigned categories and exact same inputs for base/adapter. Use teacher-forced likelihood for inexpensive screening; generate answers for selected cases because likelihood cannot certify free-running behavior or EOS handling.
4. For a release claim, enlarge the independent test set until uncertainty resolves the chosen practical target. A starting design is at least 100 in-domain examples plus a separately sized regression set, bounded to the supported context; sample size must follow observed variance, not a magic minimum.

Proposed release criterion: task accuracy improves by at least **5 percentage points**, with the paired 95% interval excluding zero; the lower interval bound for unrelated-control accuracy change must remain above **−3 points**; all critical schema/format constraints pass; no systematic answer collapse. These margins can be changed for the user's task, but only before final evaluation. A tiny panel cannot establish a narrow non-regression margin.

Report paired per-example results and a bootstrap interval that resamples independent source families, recomputing the aggregate metric each time. For token-weighted NLL, resample examples/families and recompute total NLL/total tokens; tokens from one answer are not independent observations. Bootstrap/statistical tests are standard NLP comparison tools, but they do not remove dataset leakage or fix an unrepresentative sample. [Statistical testing guide](https://aclanthology.org/P18-1128.pdf)

For open-ended answers, blind model identities and review correctness, instruction compliance, unsupported claims and usefulness separately. Show representative failures as well as wins. If a model judge is introduced later, save its model/version/prompt and swapped-order verdicts; do not use the trained model to be its own sole judge.

### 5. Stability, memory and reproducibility

After correctness and pilot quality pass, run a bounded endurance sequence with existing short examples, recording gradient norms, adapter norms, finite checks, tokens/sec, bytes read, peak MLX/RSS, swap and completed checkpoints. Scale sequence length gradually; the 512-token architectural cap is not a tested operating envelope. The process watchdog is advisory rather than a hard memory quota.

Pass when every completed step is finite, memory stabilizes after warmup within a predeclared budget, no corrupted checkpoint is exposed, and recovery reaches the same next update as uninterrupted execution. Preserve failures and abort records. Run one GPU workload at a time on this Mac; independent reviews and CPU scoring can remain parallel.

## Small reusable evaluation tool to build

Add a comparison runner around the existing public load/evaluate/generate APIs, rather than another training implementation. Input: versioned JSONL cases with `case_id`, `source_family`, `category`, prompt, expected answer/validator and token limit. Output: JSONL with exact prompt/target IDs, masked token count, per-case summed NLL, output IDs/text, EOS/truncation status, grader result, runtime and source fingerprints; then a deterministic CPU summary of paired deltas and uncertainty.

The manifest should bind base identity, adapter hash, code commit plus dirty-diff hash, dependency lock hash, macOS/chip/MLX versions, dataset/split hashes, chat-template hash, seed, mask policy, rank/alpha/layers, memory/chunk/recompute settings and decoding settings. Reject mismatched manifests when comparing runs. Do not reuse cached base outputs across changed models, templates or preprocessing.

For broader benchmarks, implement a thin `lm-eval` backend later: `loglikelihood` and `generate_until` over this loader, strict rejection beyond 512 tokens, and exact token-boundary handling. `loglikelihood_rolling` should remain unsupported until correct context/Engram semantics are implemented. The standard harness can supply task definitions, graders, sample logging and integrity checks; it cannot validate our custom model math. [Backend interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md), [evaluation CLI](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)

Recommended order on this Mac: finish the current causal demonstration; add CPU comparison/reporting and negative fixture tests; capture independent reference logits; run the 12-case smoke and a real-task pilot; then decide whether a larger quality run is justified. Publish a status per gate (`passed`, `failed`, `not run`, `insufficient evidence`) instead of a single ambiguous “verified” label.

## Follow-up recorded 2026-10-12

This research snapshot predates the completed fitting/probe runs and the new hardening tests. The current suite has **142 passing tests and 27 passing subtests**. See [current gate status](quality-status.json) and [quality plan](../QUALITY.md) for observed results; recommendations above should not be read as executed checks.
