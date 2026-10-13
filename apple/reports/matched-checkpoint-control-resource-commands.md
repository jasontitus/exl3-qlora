# Prospective matched-control resource amendment

The immutable original 80 GiB control plan remains in `matched-checkpoint-control-plan.json`. The separate `matched-checkpoint-control-resource-plan.json` binds it, the original config, the observed standard-fit startup failure and both prior training memory reports. No failed control launch is being claimed.

This run uses **76 GiB for training and 72 GiB for fresh evaluation, with reserve8 GiB throughout**. The BF16 treatment used80 GiB in both phases. Runtime-v2, all math and optimizer settings, chunk8192 and the explicit cache sizes are unchanged. The audit must state that resources differ; successful execution cannot establish timing equivalence. Historical checkpoint-mode MLX peak73.9483 GiB leaves about2.05 GiB margin, but this is not a sampled process RSS peak or a guarantee against memory failure.

The 56 CPU tests cover the existing evidence tamper gates, distinct phase budgets, immutable original plan, mathematical and cache settings, redundant resource claims, ordinary signal cleanup and dependency preflight. Actual `--verify-only` passed. No GPU subprocess was started by that check. Source hashes are in `matched-control-resource-preflight.json`.

Run serially after the active adapter evaluation and other GPU work finish. This launcher records actual PIDs/argv/environment, separate logs, timestamps, exit status and before/after hashes. It never retries or selects a new setting.

```sh
cd /Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple
/Users/jasontitus/experiments/exl3-qlora/apple/.venv/bin/python scripts/run_matched_control.py --plan reports/matched-checkpoint-control-resource-plan.json --receipt artifacts/loss-fix/matched-control-76g-launch.json
```

After both processes exit successfully, run the CPU audit:

```sh
/Users/jasontitus/experiments/exl3-qlora/apple/.venv/bin/python scripts/audit_matched_control.py --plan reports/matched-checkpoint-control-resource-plan.json --training-report artifacts/loss-fix/checkpoint-control-all-layers-76g-one-step/training.json --post-evaluation artifacts/loss-fix/checkpoint-control-all-layers-76g-post.json --output reports/matched-checkpoint-control-76g-audit.json
```

The new checkpoint, post report, launch receipt/logs and audit must not exist before their respective commands. A startup failure, interrupted update, regression, unresolved outcome or historical mismatch is retained, never silently repaired by rerunning. Inspect actual launch provenance separately from the result-consistency audit. Even a reproduced regression-to-descent contrast applies to one inspected training example and this bundled quantizer intervention; it does not certify held-out quality, generated Metal correctness/performance or a unique root cause.
