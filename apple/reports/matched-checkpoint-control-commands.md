# Matched checkpoint-mode control launch sheet

Prepared before the fresh control run. The BF16 result was already known; this is a development experiment on the original inspected row. Run these commands serially only after the active training and fresh-fit chain has released the GPU. No command below was executed during preparation.

Plan SHA256: `3606cd6f3d01af73ff6ebc85b23a91c22f14b61b1030dac0e4d4b758957f757a`. Config SHA256: `4aead354935a2b2337d4e0d815d52474f2b699e63f30f9d471548d1538e3fa80`. Runtime-v2 snapshot SHA256: `707e6a6996f08c60ef99aafc4b690a0d3ea360710b7aec61110980b053de2917`.

The config differs from the BF16 treatment only in `activation_mode=checkpoint` and its new output directory. It retains the 80 GiB process budget, 8 GiB reserve, packed cache 8 GiB, decoded/prefix caches disabled, all 40 attention adapters, rank4/alpha8, seed8, chunk8192, learning rate1e-5, fresh uncorrected Adam and exactly one complete 811-target update. No resume/warm start.

Use a new launch receipt, stdout/stderr logs, and capture start/end times and exit codes. Verify all planned hashes immediately before launch. Do not silently alter settings after a resource-preflight failure.

From `/Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple`:

```bash
PYTHONPATH=/Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple/artifacts/loss-fix/runtime-v2/src /Users/jasontitus/experiments/exl3-qlora/apple/.venv/bin/python -m exl3_mlx.workflow train --config /Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple/artifacts/loss-fix/checkpoint-control-all-layers-one-step-config.json
```

Only after training exits successfully with a complete step-000001, evaluate that new control checkpoint in a separate process:

```bash
PYTHONPATH=/Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple/artifacts/loss-fix/runtime-v2/src /Users/jasontitus/experiments/exl3-qlora/apple/.venv/bin/python -m exl3_mlx.workflow evaluate --model /Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw --data /Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple/artifacts/loss-fix/original-one-row.jsonl --adapter /Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple/artifacts/loss-fix/checkpoint-control-all-layers-one-step/step-000001 --activation-mode checkpoint --chunk-size 8192 --seq-length 5808 --profile development --reserve-gib 8 --prefix-cache-mib 0 --token-metrics --report /Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple/artifacts/loss-fix/checkpoint-control-all-layers-post.json
```

Then run the CPU audit. It uses the checksummed runtime-v2 reader, requires exact configuration/source/data/token evidence, validates the saved first Adam update, and compares every adapter, optimizer, and RNG tensor to the historical checkpoint. It records non-reproduction and negative outcomes without treating them as data to discard.

```bash
/Users/jasontitus/experiments/exl3-qlora/apple/.venv/bin/python /Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple/scripts/audit_matched_control.py --plan /Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple/reports/matched-checkpoint-control-plan.json --training-report /Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple/artifacts/loss-fix/checkpoint-control-all-layers-one-step/training.json --post-evaluation /Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple/artifacts/loss-fix/checkpoint-control-all-layers-post.json --output /Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple/reports/matched-checkpoint-control-audit.json
```

Auditor SHA256 at preparation: `7ba04ed1ded2dafe6765b63d2d2eab8aa1bc3bb0f428a648fb414eabac785b4d`. Its 24 CPU tests passed, including expansion of all prior treatment input hashes, a repaired-checksum treatment-mode relabeling attack, missing prior paths, stale report/state, and explicit BF16 modes. Fifteen additional launcher lifecycle/runtime tests passed; Ruff passed. No MLX imports or GPU calls were used.

Primary decisions use an absolute NLL band of 1e-6 around each mode’s own complete pre-update baseline. If checkpoint mode regresses while BF16 descends, report that bypassing the specified activation-quantizer bundle changed this fixed one-step outcome. If the new control descends, the matched control does not reproduce the failure; investigate before attributing the old failure to the bundle. A successful contrast does not identify a unique quantizer or prove general fine-tuning quality.

## Recorded serial launcher

The CPU-only preflight has passed against the actual unmodified treatment: all 18 prior input hashes matched and the pinned treatment auditor reproduced its receipt exactly. The control plan/config hashes above are unchanged. The auditor now reports `matched_evaluation_runtime_and_resolved_settings`; training runtime provenance must also be checked in the launch receipt.

After the active GPU chain has finished, the following launcher runs the two commands above serially and creates `artifacts/loss-fix/matched-control-launch.json`. It records actual Popen argv/PID/cwd/PYTHONPATH, start/end timestamps, exit status, separate new stdout/stderr logs and hashes, and plan/config/snapshot/tool hashes before and after execution. Training failure or changed launch inputs prevent evaluation. It never retries or changes settings. Then run the separate CPU audit command above.

```bash
/Users/jasontitus/experiments/exl3-qlora/apple/.venv/bin/python scripts/run_matched_control.py \
  --plan reports/matched-checkpoint-control-plan.json \
  --receipt artifacts/loss-fix/matched-control-launch.json
```

Add `--verify-only` for a CPU preflight that creates no outputs and starts no subprocesses. Only this verification mode has been executed during preparation.

Launcher SHA256: `c09e4a0de0f2b5a9e747012ba810f24fb692fcf718cd48c91b7a3bf92a9d15b0`.

The launcher handles SIGTERM and SIGINT through child process-group cleanup, preserves the failure receipt, and restores prior signal handlers. It verifies the exact planned Python executable, Python version, and all six recorded dependency versions using CPU metadata before launch and after each phase. This does not cover SIGKILL or machine failure. Updated preflight receipt: `matched-control-tool-preflight-v2.json`.
