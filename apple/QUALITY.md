# Standard workflow verification update

The new browser/CLI/Python SFT facade and adversarial fixes are documented in
[standard-sft-review.md](reports/standard-sft-review.md). The integrated suite now
passes **263 tests and 27 subtests**; [JUnit report](reports/standard-sft-pytest.xml).
Real-model workflow evidence is in
[standard-sft-verification.json](reports/standard-sft-verification.json).
The previous numerical, causal fitting and quality evidence below remains valid
within its stated scope. New standard interfaces do not remove the 512-token or
native full-model parity limitations.

# Quality gates for the Apple EXL3 trainer

Tool correctness, adapter influence, and useful task quality are separate claims.
We record a status and evidence for each. A changed answer or falling loss alone
does not certify this port or a fine-tuned model.

The current supported target is the local DeepSeek V4.1 3.0-bpw streaming repack.
These checks do not establish support for other EXL3 layouts or architectures.

## Current evidence

- **Causal fitting demonstrated:** the base starts `I'd be happy` for `Code?`
  (generation capped at four tokens); the selected adapter answers `violet` and
  EOS. Conditional answer NLL falls from 10.0753 to 0.05798. Disabling the adapter
  restores the exact base logits; reloading restores the exact trained result.
- The selected adapter inherits one update at learning rate 0.003, followed by
  four updates with a fresh Adam optimizer at 0.0001. The separate eight-update
  0.003 trial failed, producing `vvvv`; its report is retained. This is a
  controlled fitting demonstration with an observed learning-rate correction.
- `Code please?` also produces `violet`, but is a development probe after observing
  the first trial. It is not an untouched final test or a generalization estimate.
- Peak MLX allocation for the successful run was 9.16 GiB; observed process peak
  was 10.61 GiB under a 16 GiB process guard and 32 GiB system reserve. This tests
  only the short fixture, not the entire 512-token architectural limit.
- Fresh paired probes retain all three controls: arithmetic (`4`), literal copy
  (`blue`), and an explicit alternative codeword instruction (`amber`). There are
  zero regressions and zero exact-answer improvements across the four probes.
  The broader wording `What's the codeword?` still yields the truncated prefix
  `I don't have` under both base and adapter, despite target NLL improving by
  5.97. Falling loss therefore did not imply a better generated answer.
  See [paired results](reports/quality-probe-paired.json) and
  [run manifest](reports/quality-probe-manifest.json).
- Family ablation confirms both adapter families affect the logits. On `Code?`,
  all-on / attention-only / expert-only / all-off answer NLL is respectively
  0.05798 / 0.04982 / 7.65663 / 10.07533. Attention-only retains all three correct
  teacher-forced next-token argmaxes; expert-only does not. Expert adapters have
  measurable influence, but their inclusion slightly worsens this prompt's NLL.
  All-off exactly restores the baseline logits. These are teacher-forced family
  diagnostics, separate from the fresh greedy generation check.
  See [family ablation](reports/behavior-family-ablation.json).
- **142 tests and 27 subtests pass**, including corruption/nonfinite loader
  rejection, masked loss, gradient checks, exact resume, report provenance and
  probe grading. See [test receipt](reports/quality-pytest.xml).
- **Independent complete-model numerical parity and representative task quality
  remain unverified.** The recorded gate statuses and measured behavioral controls
  are in [quality status](reports/quality-status.json).

## Run paired behavioral checks

Use the same base, template, input IDs, generation cap and greedy decoding for
both conditions. The runner uses the existing evaluation and generation APIs;
it contains no alternative trainer.

```bash
cd ~/experiments/exl3-qlora/apple
export EXL3_MODEL="$HOME/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"
export EXL3_ADAPTER="artifacts/behavior-demo-20261012-lowrate/step-000004"
uv run python scripts/quality_probe.py \
  --model "$EXL3_MODEL" --adapter "$EXL3_ADAPTER" \
  --cases examples/quality-development-probes.jsonl \
  --output artifacts/my-quality-probe
```

Each output directory must be new. A fixture row has exactly these fields:

```json
{"case_id":"copy-blue","source_family":"literal-copy","category":"control","prompt":"Repeat only: blue","expected":"blue","max_new_tokens":4}
```

The runner rejects duplicate IDs, invalid tokens, template/prefix mismatches,
context overflow and caps that cannot fit the expected answer plus EOS. It
records per-case target NLL, generated IDs/text, exact match, EOS/truncation,
base-pass/adapter-fail regressions and unintended `violet` answers. A correct text
prefix cut off at the cap is not a completed-answer pass. A case both models fail
does not by itself establish fine-tuning damage.

Outputs include a manifest with code/runtime/dependency, dataset, adapter and
tokenizer hashes; tokenized cases; incremental results; and `paired.json`.
The base identity is a metadata fingerprint, not a hash of all model payloads.
Four fixed probes produce descriptive counts, not a statistical quality verdict.
Once a probe informs training choices, it is development data.

To measure each adapter family's influence on the trained prompt:

```bash
uv run python scripts/adapter_family_ablation.py \
  --model "$EXL3_MODEL" --adapter "$EXL3_ADAPTER" \
  --report artifacts/my-family-ablation.json
```

It reloads before each all-on, attention-only, expert-only and all-off condition.
It reports nonzero B tensors, norms, NLL and logit differences. These nonlinear
effects are not additive attribution or a substitute for expert-only training.

## Gates before a useful-quality claim

| Gate | Required evidence |
|---|---|
| Artifact and objective integrity | Correct shifted answer/EOS masks; immutable split identities; checksum and nonfinite rejection before loading; exact optimizer/RNG/sample-cursor resume; checksum-validated reuse receipts. |
| Independent numerics | Same packed weights and token IDs through a separate reference; decoded weights, Engram/attention/block states, route IDs, full logits and declared surrogate gradients agree within predeclared dtype-specific tolerances. |
| Causal learning | Intended complete answer, lower target NLL, same-prompt base/adapter comparison, adapter-off restoration, fresh-process reload, family influence and contrastive controls. |
| Useful task quality | Source-family-disjoint train/dev/test; duplicate/near-duplicate audit; predetermined scoring; paired improvements and uncertainty; unrelated-task and instruction-following regression limits. |
| Operations | Bounded multi-step stability, increasing context tests, finite gradients, memory/throughput records, forced watchdog/interrupted-save recovery and last-checkpoint usability. |

A new independent NumPy float64 Engram oracle passes forward comparison and
finite differences of all 192 input-stream coordinates against MLX's VJP. It
covers separate stream normalization and both smooth signed-root branches with
a dense FP32 projection. It does not cover the clamp boundary or quantized STE.
See [the test](tests/test_engram.py).

Next prioritize mixed-precision block checks and an offline full-logit export
from the native inference implementation
using the **real EXL3 expert provider**. Its existing injected-expert parity test
is not full-model evidence. Record source revisions and exact inputs; stop at the
first unexplained divergent stage. Set cross-engine tolerances from independent
primitive and rounding error evidence, not by loosening limits until a run passes.
FP8/FP4 identity straight-through gradients require a matching surrogate reference;
finite differences of the quantized staircase are not a valid STE oracle.

For a real downstream task, start with a 12-case smoke panel, then a 32-case pilot
and a separately held final set sized to resolve the desired improvement. Prefer
executable scoring (classification, extraction/schema correctness) where possible.
Proposed release margins are at least +5 percentage points task accuracy with a
paired 95% interval excluding zero, and a control-accuracy lower interval bound
above -3 points. These are project proposals to agree before final evaluation,
not universal thresholds. Bootstrap independent source families, not individual
tokens. A tiny probe cannot establish these margins. For open-ended prose, use
blinded paired human judgments; calibrate any model judge and swap answer order.

## Adversarial review and hardening

The [code review](reports/adversarial-code-review.md) reproduced loader checksum
bypass and nonfinite-value acceptance. The loader now verifies saved receipts
before using adapter configuration to allocate a model, and checks every tensor
before changing parameters. New standalone adapters include a weight checksum;
older training checkpoints retain support through `trainer_state.json`. Legacy
standalone directories with no checksum now fail clearly and require a trusted
resave. Checksums detect changes; they do not authenticate an untrusted publisher.

The [process review](reports/adversarial-process-review.md) identified constant
answer collapse, reused development examples, weak frozen-object identity checks,
incomplete generation evidence and stale report reuse. Generation now records
the exact prompt IDs/cap/EOS/stop reason. `verify_workflow.py --continue-run`
rejects changed or missing source/runtime/model-metadata/evidence fingerprints.
Historical reports remain readable but cannot be silently reused under new code.

Full base-payload verification, independent complete-model parity, broader
Engram/mixed-precision gradient fixtures and long-context endurance remain open.
The current adapter-off comparison is strong causal evidence on its routed paths;
Python object identity is not a full frozen-content audit.

The [sourced research notes](reports/quality-research.md) explain the choices using
[MLX custom VJPs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html),
[PyTorch gradcheck](https://docs.pytorch.org/docs/main/generated/torch.autograd.gradcheck.gradcheck.html),
[TRL SFT](https://huggingface.co/docs/trl/sft_trainer),
[lm-eval](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md),
[HELM](https://arxiv.org/abs/2211.09110), and
[paired statistical testing](https://aclanthology.org/P18-1128.pdf).
