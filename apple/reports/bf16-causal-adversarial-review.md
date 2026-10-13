# BF16 one-step evidence review

The measured BF16 configuration passes the predeclared one-row training-fit test. Its complete pre-update objective was **0.50617235899**, and a separate saved-adapter evaluation reported **0.50149060382** over the same 811 targets: **−0.00468175517 NLL (0.925%)**. This clears the predeclared absolute 1e-6 decision band. I found no blocker to reporting that bounded result. This review ran only CPU checks and did not repeat model inference.

I reran the pinned CPU auditor and obtained an identical report, verified all 18 recorded input hashes and all 42 runtime-v2 snapshot files, matched the evaluation's 38 source hashes and dependency lock to the snapshot, and checked its stdout JSON and token counts/means/predictions. The two checkpoints have matching data, context IDs, masks, rank/alpha, all 80 initial A matrices, chunk size, and first-step optimizer contract. All 80 B matrices changed; saved tensors are finite; saved moments agree with the declared fresh, uncorrected Adam update. The launch receipt records the resource-only reserve reduction from 12 to 8 GiB before the run; the process ceiling stayed 80 GiB. These are consistency checks, not independent proof of execution or a full base-weight payload hash. Zero initial B follows the fresh-run configuration and initializer; a pre-update B snapshot was not recorded.

| Experiment | Own baseline NLL | Saved-adapter NLL | Change |
| --- | ---: | ---: | ---: |
| Historical checkpoint activation mode | 0.50125785921 | 0.50537246684 | +0.00411460763 |
| New BF16 activation mode, runtime-v2 | 0.50617235899 | 0.50149060382 | −0.00468175517 |

The new post-update NLL is still **0.00023274461 above the historical unadapted checkpoint-mode baseline**. Each mode defines a different forward objective, so this result is not evidence that BF16 has better absolute quality. The audit correctly records `matched_causal_claim: false` and `quality_certified: false`.

The causal interpretation remains limited. There is no current-runtime control update yet. The intervention bypasses several differentiable-path activation quantizers together; it does not identify which one mattered. FrozenIndexer quantization is retained, but changed backbone activations can still change index selections and expert routes. BF16 rounding and discrete routing also remain. The first-order prediction was −0.00071972927, whereas the measured change was about 6.5 times larger; descent does not validate the surrogate gradient or prove the original failure's mechanism.

This is one update on one previously inspected training example, with teacher-forced targets. There is no held-out result, generated-patch correctness test, or kernel speed result. The BF16 run has no pre-update token-level baseline, so its improvement cannot yet be attributed to added code, comments, or another target category. The historical checkpoint-mode token metrics cannot substitute for its own baseline.

## Bounded historical-control replay

The planned runtime-v3 three-row base evaluation can supply the original row's forward-control replay without another runtime-v2 base pass, provided all of these checks pass:

- The row has the exact original 5,809 IDs, `loss_start=4998`, and 811 supervised targets; use the per-row result, not the three-row aggregate.
- Its supervised-logits SHA is exactly `aa93ec9398b2ccad0e07827d378bd8776af72141a26f2e8d1ca7e3f333b6e6d3`, with scalar NLL within 1e-6 of 0.5012578592107731.
- It uses checkpoint activation mode, chunk 8192, prefix cache 0, the original base identity, and no loaded adapter. Record the prepared row, receipt, source, and snapshot hashes.
- Runtime-v2 and v3 model/math modules remain byte-identical. The inspected differences are CLI/SFT/workflow probe plumbing and `training.evaluate`'s new `preserve_mode=False` parameter; its default evaluation arithmetic is unchanged. The new `fit_probe.py` module does not run in ordinary evaluation.

After the active run releases the GPU, replay the historical saved adapter under runtime-v2. From `loss-fix/apple`, use a fresh output path and capture stdout/stderr, actual command, runtime snapshot hash, input/adapter hashes, start/end timestamps, exit status, and the evaluation JSON. Proposed command (not run by this review):

```bash
PYTHONPATH="$PWD/artifacts/loss-fix/runtime-v2/src" \
  /Users/jasontitus/experiments/exl3-qlora/apple/.venv/bin/python -m exl3_mlx.workflow evaluate \
  --model /Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw \
  --data "$PWD/artifacts/loss-fix/original-one-row.jsonl" \
  --adapter /Users/jasontitus/experiments/exl3-qlora/apple/artifacts/ktune-20261012/full-parent-pilot-development/attempts/attempt-1791843982943262000/train/step-000001 \
  --activation-mode checkpoint --chunk-size 8192 --seq-length 5808 \
  --profile development --reserve-gib 8 --prefix-cache-mib 0 --token-metrics \
  --report "$PWD/artifacts/loss-fix/checkpoint-control-post-v2.json"
```

Require exact supervised-logits SHA `d76ed2371b074714b4d1acf25dd15219467e717a00582a85412d4d8a4319ebf4`, NLL within 1e-6 of 0.5053724668388994, exact target IDs/mask, and the expected adapter/data/source identities. A mismatch is a failed replay requiring investigation, not permission to widen tolerance or replace the reference. A successful pair establishes that the historical forward regression is reproduced by the current evaluator. **It still does not demonstrate that the current checkpoint-mode training code would produce the historical update.** A fully matched training contrast requires a fresh control update under the same pinned runtime, or independently demonstrated exact control gradient and update reproduction. Keep that stronger claim false until then.

Evidence: [CPU rerun](bf16-all-layers-one-step-audit-review-rerun.json), [independent checks](bf16-causal-adversarial-checks.json), [predeclared protocol](activation-causal-plan-v2.json), [fresh evaluation](../artifacts/loss-fix/bf16-all-layers-post.json), [launch receipt](../artifacts/loss-fix/bf16-all-layers-launch.json). No runtime, engine source, or existing evidence was modified.
