# Independent loader-hardening review — 2026-10-12

The repaired staging candidate has **no remaining concrete blocker identified for the workflow’s supported fixed-rate, default Adam optimizer**. Independent CPU verification passed **196 tests**. Real MLX loading and bitwise continuation remain an integration gate: the author prepared 16 cases, but this reviewer did not run GPU work. The active runtime-v2/v3 snapshots were not edited.

This review supersedes the initial isolated patch’s `deepcopy` design. The current candidate creates a fresh Adam schema; it no longer relies on copying installed MLX optimizer instances. Exact reviewed source/test hashes and the CPU JUnit hash are recorded in [the independent receipt](loader-hardening-independent-receipt.json).

## Findings reproduced and repaired

1. **Initialized destination moments could define the accepted input schema.** With a current parameter shaped `(2,2)`, setting both the destination and incoming `m` moment to `(1,4)` was accepted. Deleting `m` from both nonempty state dictionaries was also accepted. The installed MLX `init` preserves nonempty moment dictionaries instead of reinitializing them. Thus initializing a copied destination was not an independent schema check. The repair initializes a fresh `optim.Adam(learning_rate=1.0)` from the current trainable parameters. Its placeholder learning rate supplies the schema only; the verified saved value is restored unchanged. New regressions also cover missing `v`, wrong dtypes, extra moments/parameters and a valid payload replacing a malformed destination.
2. **A fractional optimizer counter could pass through integer conversion.** A destination float32 step allowed an incoming float32 step `1.5` to match trainer step `1`, because validation used `int(item())`. The repaired schema requires Adam’s canonical scalar counter dtype/shape and compares the actual value exactly. Regressions cover fractional, signed and nonscalar counters, and invalid learning-rate dtype/shape.

These malformed-state cases were reproduced using the exact loader function body with the CPU safetensors fixture before the repair. They do not describe failures of the active pinned training runs. The repaired regressions reject before any model, optimizer or RNG mutation.

## Atomic rejection and compatibility

The loader checks the contract, strict step/cursor types, accumulation relation, all file checksums, exact canonical optimizer keys/shapes/dtypes, finite moments, nonnegative second moments, positive finite learning rate, saved rate agreement, RNG structure and exact step agreement before replacing state. Adapter loading separately verifies saved arithmetic/mode/base identity, checksums, tensor structure and finite weights before replacing model parameters.

For this supported workflow, the final assignments contain no additional ordinary payload-validation gate after the adapter mutation. This supports rejection without partial mutation for malformed payloads. It is **not transaction rollback after an arbitrary driver/device failure**, and checksums do not authenticate a payload whose receipt was also replaced. No concurrent-file-replacement guarantee was established.

Current default-Adam checkpoints and compatible legacy contracts still pass. A missing legacy activation mode defaults to `checkpoint`; an absent legacy engine tag remains unknown numerical provenance, not proof of migration compatibility. An explicit incompatible tag or activation mode is rejected. Older contracts missing learning-rate/accumulation fields retain their prior interpretation rather than acquiring invented metadata.

The loader now explicitly accepts only default Adam betas, epsilon and bias-correction settings, without a scheduler. It rejects Adam subclasses, alternate optimizer algorithms and changed settings that the saved contract cannot represent. This matches the standard workflow’s constructor; it is not a generic MLX optimizer interchange API. A malformed destination state can be replaced by a valid canonical checkpoint, because destination arrays no longer define the input schema.

The installed MLX source was inspected read-only to confirm `init`, state assignment and Adam moment construction. The old shared-leaf/deepcopy assumptions are no longer on the repaired path. Real binding behavior still needs the prepared integration checks.

## Verification boundary

Independent command:

```sh
PYTHONPATH=src /Users/jasontitus/experiments/exl3-qlora/apple/.venv/bin/python -m pytest -q \
  tests/test_loader_hardening_cpu.py tests/test_activation_mode_config.py \
  tests/test_sft_project.py tests/test_resource_profiles.py \
  --junitxml=reports/loader-hardening-independent-cpu.xml
```

Result: **196 passed**; Ruff and static Python compilation passed for both loader files and both new regression files. The CPU harness executes the actual loader AST with NumPy safetensors and a stateful optimizer stub; it does not establish real MLX numerical continuation.

The author’s 16 pending real-MLX cases cover eight failure-preservation cases, three exact valid resumes (including replacement of malformed destination state), and five matching malformed destination/payload cases. They should run with the existing workflow/checkpoint tests in the coordinated GPU window. No full-model run or performance claim follows from this review.

Installing changed source hashes does not grant existing projects permission to resume under a different implementation. Keep their pinned runtime unless a separately explicit migration is validated. This source-only loader hardening does not alter model arithmetic or the active experiments.
