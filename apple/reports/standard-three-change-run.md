# Standard fine-tuning on three complete Metal changes

Status recorded October 12, 2026: the normal SFT project completed **12 updates on three distinct changes**, four exposures each. Fresh-process, uncached answer NLL fell from **0.3555532213 to 0.3262765797**, an **8.23% reduction**, with all three rows improving. The saved adapter's fresh logits matched the three warm-cache probe rows exactly. Training plus the final probe took **1,808.36 seconds (30.1 minutes)** and peaked at **53.08 GiB MLX** under an 80 GiB guard. See the [paired fit audit](three-changes-standard-fit-result.json), its [independent CPU reproduction](three-changes-standard-fit-independent.json), and the [checkpoint audit](standard12-training-independent-audit.json).

This verifies fitting and the saved-optimizer/adapter-reload/cache path for these three historical answers. It does not verify generated-kernel correctness or speed. Added-code likelihood improved in aggregate but worsened slightly on two rows; comments and diff headers account for about 92.1% of the total NLL-sum reduction. The launch uses [protocol v2](three-change-standard-protocol-v2.json), the immutable [runtime-v3 snapshot](../artifacts/loss-fix/runtime-v3/snapshot.json), and this [project manifest](../artifacts/loss-fix/three-changes-standard-prefix/project.json). It uses the normal SFT trainer, separately from the earlier one-row diagnostic.

## What is being trained

The narrow official-contest import admitted **24 complete historical edits**. The archive and current upstream Git histories contain additional candidates; 24 is not their total size. That is an evidence/data count, not a count of trained edits or proof that all examples fit in 80 GiB. This run selects **three distinct changes**, all from one conservative source lineage:

| Row | Historical change | Complete tokens | Prompt tokens | Supervised answer tokens, including EOS |
| ---: | --- | ---: | ---: | ---: |
| 0 | SIMD ownership constant, two rows to one | 3,606 | 3,402 | 204 |
| 1 | Prefill tile 64 to 32, with matching dispatch | 5,809 | 4,998 | 811 |
| 2 | Two-row GDN Metal recurrence and prefill selection | 5,822 | 3,945 | 1,877 |

Normal preparation preserved every input ID and answer mask. There is no truncation, packing, validation split or test split. One pass has 15,237 serialized tokens and 2,892 supervised targets. Twelve updates expose each row four times in seed-8 order `[2,1,0,0,2,1,1,0,2,0,2,1]`, totaling 11,568 supervised target presentations. Repeated exposures, modified files and diff hunks are not additional independent changes.

Only block 39's `wq_a` and `wo_b` adapters train: four FP32 tensors and 78,848 parameters, rank 4/alpha 8. The model starts with fresh seed-8 A matrices, zero B and fresh Adam at learning rate 1e-4. This is not continuation of the successful one-row diagnostic. Blocks 0–38 remain frozen; the in-memory prefix cache may reuse their exact states after each example's first complete pass. Final-block attention and MoE routing still run normally for every update.

The standard run retains `activation_mode=checkpoint`. The separate completed BF16-mode result does not change this run's mode, initialization, data or budget.

## Exact normal workflow

The project below already exists and completed training and paired verification. Inspect the preserved result before launching another workload. For a separate reproduction after this run, choose a **new** project path; preparation deliberately refuses to overwrite an existing project.

```bash
cd /Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/loss-fix/apple
export EXL3_PYTHON="$HOME/experiments/exl3-qlora/apple/.venv/bin/python"
export EXL3_MODEL="$HOME/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"
export EXL3_RUNTIME="$PWD/artifacts/loss-fix/runtime-v3"
export PYTHONPATH="$EXL3_RUNTIME/src"
export EXL3_PROJECT="$PWD/artifacts/loss-fix/three-changes-standard-prefix"

"$EXL3_PYTHON" -m exl3_mlx.cli status "$EXL3_PROJECT"
```

For this preserved project, `artifacts/loss-fix/metal-tune status` is a convenience
launcher pinned to runtime-v3. Its `resume --steps TOTAL`, `verify`, `export`,
`chat` and `studio` actions use that same runtime; `resume` means a total update
count. Long-patch chat remains impractical with the current generation path.

The creation command records the actual configuration:

```bash
"$EXL3_PYTHON" -m exl3_mlx.cli finetune \
  --model "$EXL3_MODEL" --data artifacts/loss-fix/three-metal-changes-raw.jsonl \
  --output "$EXL3_PROJECT" --demo --prepare-only --no-compare \
  --steps 12 --max-length 5822 --learning-rate 0.0001 \
  --gradient-accumulation-steps 1 --rank 4 --lora-alpha 8 \
  --adapter-layers 39 --activation-mode checkpoint --chunk-size 8192 \
  --profile development --memory-gib 80 --reserve-gib 8 \
  --packed-cache-mib 8192 --decoded-cache-mib 0 --prefix-cache-mib 2048 \
  --allocator-cache-mib 256 --save-steps 1 --save-total-limit 12 \
  --comparison-examples 3 --final-fit-examples 3
```

`--demo` explicitly permits this one-lineage training-only dataset. It does not create held-out examples. `--max-length` includes the entire serialized prompt and answer; the backend sequence length is 5,821, and oversized rows are rejected instead of silently truncated. Preparation should report counts `train=3, valid=0, test=0` and prepared train SHA `ca6fd8c939be9de5f7715e79d5f1ffc9354d251d1092aa9a3cc77d5d5ce80cb3`.

The 80 GiB process guard includes all caches. The explicit 8 GiB reserve was chosen against the measured startup headroom for this launch; the development profile still defaults to a 32 GiB reserve. The watchdog remains enabled. Resume without resource flags inherits saved limits. The 2 GiB prefix cache is optional and exact; restarting a process empties it, so the first pass over each example is expensive again. Chunk 8192 changes floating-point reduction order and remains fixed in this project's numerical contract.

Train to a **total** of 12 updates, then verify using separate fresh processes:

```bash
"$EXL3_PYTHON" -m exl3_mlx.cli resume "$EXL3_PROJECT" --steps 12 --no-compare
"$EXL3_PYTHON" -m exl3_mlx.cli ktune verify "$EXL3_PROJECT"
```

A completed optimizer checkpoint is saved after every update. The optional final probe scores all three rows in the training model's existing mode without updating parameters. Fresh verification reloads the saved step-12 adapter and explicitly disables prefix caching in both its base and adapter child processes. It compares the probe's exact target IDs and per-row logit hashes, plus per-token NLL within the declared 1e-6 tolerance. Real warm-cache parity is claimed only when all three probe forwards registered cache hits with no new misses.

The follow-up CPU report reads those native receipts and every retained checkpoint. It does not load the model:

```bash
"$EXL3_PYTHON" scripts/summarize_standard_fit.py \
  --project "$EXL3_PROJECT" --runtime "$EXL3_RUNTIME" \
  --protocol reports/three-change-standard-protocol-v2.json \
  --categories artifacts/loss-fix/token-categories-0.json \
               artifacts/loss-fix/token-categories-1.json \
               artifacts/loss-fix/token-categories-2.json \
  --decode-context --output reports/three-change-standard-fit.json
```

The post-run CPU auditor was hardened after launch to bind each case label to
the hash-checked raw source and retained prepared-row provenance. Its
[version, script SHA and 42-test receipt](standard-fit-auditor-hardening.json)
identify that later auditor explicitly; it is not represented as the frozen
launch script. The runtime-v3 engine and active project remain unchanged.
The later [v3 recovery auditor and 54-test receipt](standard-fit-auditor-recovery-hardening.json)
adds explicit validation/presentation of the 80→72 GiB evaluation resource
exception and both preserved failed starts. The earlier 42-test receipt remains
a historical cutoff.

## Verification status

The current integrated development source passed **1,040 tests plus 27 subtests**
([JUnit](loss-fix-integrated-pytest.xml)). That later source check is separate
from this experiment's unchanged runtime-v3 and its raw run receipts.

Training, fresh paired scoring and the independent CPU evidence audit are complete. The CPU audit re-read exact data/masks, all twelve checkpoints, runtime identity, actual sampler cursor, raw process receipts and the resource amendment. Its result is value-identical to the first paired report.

| Check | Result |
| --- | --- |
| Complete token IDs/masks after normal preparation | Three rows, zero truncation, exact prepared SHA |
| Twelve finite optimizer updates and four exposures per row | All 12 checkpoints verified |
| Fresh base/adapter token-weighted NLL | 0.3555532213 → 0.3262765797 (−8.23%) |
| Equal-row macro NLL | 0.3718079316 → 0.3265421289 |
| All three answer rows improve | Yes; see row table below |
| Added-code NLL and top-1 correctness | 0.06224455 → 0.05903664; 1,501 → 1,502 / 1,527 |
| Actual final-probe cache reuse | Three hits, zero new misses |
| Warm versus fresh uncached adapter logits | Exact per-row hashes; token NLL within 1e-6 |
| Training peak MLX and elapsed time including final probe | 53.08 GiB; 1,808.36 seconds |
| Independent generated-kernel correctness or speedup | Not run |

| Row | Answer NLL, base → trained | Added-code NLL, base → trained | Added-code top-1 correct |
| --- | --- | --- | --- |
| 0: SIMD ownership | 0.31742359 → 0.24123094 | 0.06447918 → 0.06488212 (worse) | 12 → 12 / 13 |
| 1: Prefill tile | 0.50125786 → 0.46115149 | 0.06160583 → 0.06207966 (worse) | 112 → 112 / 114 |
| 2: GDN recurrence | 0.29674234 → 0.27724395 | 0.06227581 → 0.05873457 | 1,377 → 1,378 / 1,400 |

The one corrected added-code prediction was ` if` in row 2, where the base predicted ` const`, given the correct prior answer tokens. No added-code argmax prediction regressed. The categories are coarse lexical labels, not semantic kernel tests. EOS argmax correctness regressed from 1/3 to 0/3 despite pooled EOS likelihood improving; this is another reason not to infer successful autonomous patch completion from loss.

The original fresh adapter child failed before model loading because 86.48 GiB of measured headroom was below its 80+8 GiB startup requirement. The recorded [resource amendment](../artifacts/loss-fix/three-changes-standard-prefix/fit-checks/recovery-72g/recovery-launch.json) retried only adapter evaluation at **72 GiB**, keeping reserve 8 GiB, source, checkpoint, data, arithmetic and cache settings unchanged. That retry completed successfully. The existing successful 80 GiB baseline was reused, not rerun. A separate command-line data-path typo also failed before model loading; both failure logs remain bound in the report. These failed evaluation starts added no training updates. Training itself used the saved 80 GiB guard.

The [recovery review](fit-recovery-independent-review.md) and [54-test auditor receipt](standard-fit-auditor-recovery-hardening.json) describe the validation boundary. The result binds locally checksummed evidence, not independent process attestation. A negative `fit.json.loss_delta` establishes better likelihood on these selected training answers; three related edits cannot supply a credible generalization estimate.

## How this relates to the completed experiments

The earlier final-block diagnostic trained **one** change for eight updates and lowered answer NLL from 0.5012578592 to 0.4735794807. Full uncached recomputation in the same process matched its final cached logits exactly. Added-code accuracy remained 112/114; most of the loss decrease came from comments and diff headers. See the [measured result and arithmetic audit](last-block-training-result-and-recipe.md). That proves this implementation can fit that complete historical answer, not that it generates a better optimization.

The separate all40-layer control comparison is now complete. Checkpoint mode rose from its own baseline **0.5012578368 → 0.5033001955**, while experimental BF16 mode fell **0.5061723590 → 0.5014906038**. Both used one update on the same previously inspected row, with matched numerical settings and pinned runtime; BF16 mode still retains ordinary BF16 rounding, indexer quantization and normal routing. Their forward objectives differ, so compare within-mode changes. Treatment used an 80 GiB cap, while the prospectively amended control used 76 GiB for training and 72 GiB for fresh evaluation; reserve stayed 8 GiB and explicit caches were unchanged.

The completed [audit](matched-checkpoint-control-76g-audit.json) and [independent review](matched-control-final-independent-review.md) support investigating the activation-quantizer bundle, not a unique cause or universal fix. The control was designed after seeing the treatment outcome. It reproduced the historical baseline but not the exact historical update: 78 B matrices differed, and the two checkpoint-mode post losses differ although both regress. Repeated-index gradient accumulation is a plausible repeatability limit, not a localized explanation. This separate one-row comparison adds no distinct training changes to the three-row run.

## Structured upstream preparation and kernel checks

The broader intake now supports pinned mlx-serve, TensorFold, Sushi and llama.cpp histories with structured source-grounded fields and explicit unknowns. The [bounded audit](upstream-corpus-inventory.md) selected 100 candidates before token lengths were known: 93 reconstructed, five whole-file rows fit the 32,768-token ceiling, with mixed kernel/integration scope and no independently verified performance admission. Reviewed complete units yielded a 3,394-token llama.cpp convolution example with full source/patch provenance. Its [normal project](../artifacts/loss-fix/upstream-llama-prepared-project/project.json) is **prepared only, with zero optimizer updates**, one training row and no held-out split. These new candidates have not been trained into the three-change adapter.

Separately, original upstream code was compiled and checked on the M1 Ultra. The [llama.cpp pair](../artifacts/loss-fix/llama-conv-parity-v2/gpu-parity.json) passed all 16 fixed float/half-weight cases against both the original kernel and an exact dyadic CPU oracle, with finite values and intact input/output canaries. The [TensorFold EXLR8 pair](../artifacts/loss-fix/tensorfold-exlr8-parity-build-v3/parity.json) passed all 65,536 codebook states and 162 FP32 projection cases bit for bit against its original kernel, with trailing canaries intact; prompt tensor-unit, CUDA, BF16-output and surrounding host paths were excluded. Neither harness timed kernels or evaluated a generated patch. These are scoped source-pair receipts, not whole-commit verification or evidence that the fine-tuned model produces faster code.

## Remaining usefulness limits

All three prompts come from one historical lineage and use parent files chosen using the recorded edit. The full old source is bound, but the choice of files is target-conditioned. Surrounding repository dependencies, a pre-candidate workload/build contract and independent target-hardware tasks are still needed for an optimization benchmark. The historical target is M5/NAX; the M1 Ultra is the training host.

Teacher forcing supplies the correct previous answer tokens. A more likely diff or a corrected code token is not evidence that a prompt-only generated patch applies, compiles, is numerically correct or runs faster. KernelBench likewise measures both functional correctness and speedup, rather than syntax or language-model loss ([paper](https://arxiv.org/abs/2502.10517), [official evaluation](https://github.com/ScalingIntelligence/KernelBench#-evaluation)).

The current generation path recomputes the entire prefix for each new token. Long-patch generation remains impractical with this path and has no validated adapter-compatible persistent-KV integration. This experiment measures training fit and tool behavior; it does not complete the new-kernel quality test.

The other 21 eligible examples are mostly much longer: the full 24-example set has median 17,806.5 and maximum 25,486 tokens. The prior complete 5,809-token all40 update already peaked at 73.95 GiB. Eligibility and the 32,768-token implementation ceiling do not establish that all 24 examples fit or train usefully under 80 GiB. The largest completed and checkpoint-audited training length is now 5,822 tokens; this does not validate the larger archived examples or the 32,768-token ceiling.
