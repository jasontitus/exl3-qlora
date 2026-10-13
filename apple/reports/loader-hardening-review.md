# Isolated loader hardening — 2026-10-12

> Later independent review (2026-10-12): the repaired staging candidate replaces the deepcopy template with a canonical fresh-Adam schema after two adversarial findings. **196 CPU tests pass; 16 real MLX cases remain pending.** See [current review](loader-hardening-independent-review.md) and [current receipt](loader-hardening-independent-receipt.json). The original 179-test isolated-patch report below is a historical cutoff.

Prepared for review only. No active engine source, actual repository source, or
pinned runtime was changed. `loader-hardening.patch` applies relative to `apple/`;
`git apply --check` passed against the installed source. The patch changes only
two loader modules and adds two regression files.

## Changes

1. `load_adapter_config` rejects an explicitly recorded incompatible
   `engine_semantics` after integrity verification and before model allocation or
   parameter loading. Current versions remain valid. Legacy adapters with no
   version field remain readable; absence means unknown numerical provenance,
   not a certified migration. Missing legacy `activation_mode` still normalizes
   to `checkpoint`.
2. `load_checkpoint` builds its expected optimizer structure using an isolated
   optimizer copy, preserving immutable array leaves. Rejected payloads no
   longer initialize or otherwise mutate the destination optimizer. It rejects
   nonfinite optimizer tensors, negative Adam second moments, invalid scalar
   learning rates, and invalid step/cursor counters. Current contracts additionally
   require the stored learning rate and accumulation/cursor relation to match.
   Older contracts without those fields retain their prior interpretation.

The model, optimizer, and RNG are changed only after all payload validation.
This is a validation guarantee, not rollback protection for an arbitrary driver
failure during the final assignments. Checksums detect accidental corruption;
they do not authenticate a receipt whose payload and checksums are both replaced.

Project source-identity and exact resume contracts are unchanged. Installing
this patch will change implementation hashes, so existing projects must continue
with their pinned implementation for exact resume. Do not install it into an
active project's runtime.

## Verification

CPU-only checks passed: **179 tests**, including **58 new regression cases**;
Ruff and Python compilation passed. The CPU tests execute the actual loader
function bodies with NumPy safetensors and a stateful MLX-style optimizer stub.
They cover initialized and uninitialized destinations, rejection without any
array/state/RNG mutation, finite large moments, current and legacy contracts,
and current/missing/incompatible arithmetic versions. Existing activation,
project identity, and resource tests also passed.

Exact command:

```sh
PYTHONPATH=src /Users/jasontitus/experiments/exl3-qlora/apple/.venv/bin/python -m pytest -q \
  tests/test_loader_hardening_cpu.py tests/test_activation_mode_config.py \
  tests/test_sft_project.py tests/test_resource_profiles.py \
  --junitxml=reports/loader-hardening-cpu.xml
```

**Metal integration remains pending.** Ten new real-MLX tests exercise rejected
loads on initialized/uninitialized Adam optimizers, and bitwise next-update,
optimizer-state, RNG, and evaluation equivalence after a valid exact resume.
These tests were authored and statically checked but deliberately not run while
the real training job uses the GPU. The later integration gate is:

```sh
PYTHONPATH=src /Users/jasontitus/experiments/exl3-qlora/apple/.venv/bin/python -m pytest -q \
  tests/test_loader_hardening.py tests/test_workflow.py tests/test_checkpoint.py
```

In particular, the real tests must validate the isolated `deepcopy` template
against installed MLX Adam; the CPU stub is not evidence for that binding.
No optimizer algorithm, model arithmetic, adapter format, or engine semantic
revision was changed.

Receipt and patch hashes are in `reports/loader-hardening-receipt.json`; the
original two source files are retained in `baseline/` for a bounded diff.
