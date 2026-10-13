# Standard three-change SFT review — 2026-10-12

Later-results pointer (2026-10-12): this body preserves its review cutoff. The
[completed one-row diagnostic and BF16 own-mode result](last-block-training-result-and-recipe.md)
and [active normal three-change recipe](standard-three-change-run.md) supersede
pending-result wording below. The launched normal run uses protocol v2,
runtime-v3, an explicit 8 GiB reserve and all 12 checkpoints retained. Its fresh
three-row fitting/cache-parity results are still pending.

The completed eight-step immutable diagnostic is evidence of real one-case fit:
base NLL 0.5012578592 to 0.4735794807, added-code NLL 0.0616058 to
0.0596247, and exact fresh full-backbone uncached final-score parity. It is not
a held-out or executable-kernel quality result. The proposed next run exercises
the ordinary project, optimizer, checkpoint, cache and verification path.

## Selected data and training contract

Use the three shortest complete `official_promoted` examples from the existing
checksummed ktune snapshot, with complete pinned parent-file contexts:

| Exact case ID suffix (prefix `official:qwen38-125b-a6b-mlx-v1:`) | Total tokens | Supervised tokens |
| --- | ---: | ---: |
| `c0e42c7e-268c-454f-a9cb-e705f952b819` | 3606 | 204 |
| `1a621870-4993-4c43-9472-6f7d8f745140` | 5809 | 811 |
| `8a690716-8068-4da8-9532-26047c96aa9c` | 5822 | 1877 |

These are the lengths already recorded in
`artifacts/ktune-20261012/full-parent-official-audit/length-audit.jsonl`.
The importer must revalidate them before the new project is accepted. Explicit
case IDs plus `--max-examples 3` give this length-sorted order. Do not truncate.
The automatically inferred total serialized length should be 5822. The 2892
supervised tokens include each complete target and EOS; prior turns and parent
files provide context, with the final assistant response alone supervised.

All three belong to one repository/track-connected lineage. No validation/test
split is available or implied. Files and target locations were selected using
the historical edits; this is historical patch editing with target-selected
context, not independent discovery of a new optimization. Keep this disclosure.

Use rank 4, alpha 8, attention adapter block `[39]`, no expert adapters,
checkpoint activation mode, chunk size 8192, Adam LR 1e-4, clip 1, seed 8,
microbatch/accumulation 1, 12 updates, checkpoint every update, keep last two.
The existing Adam default is bias_correction=False; do not silently change it.
No held-out selection means step 12 is the selected checkpoint. The seed-8
sample order is `[2,1,0,0,2,1,1,0,2,0,2,1]`, exactly four visits per case.

Use the development resource profile, 80 GiB process budget, 12 GiB reserve,
8192 MiB packed cache, zero decoded cache, 2048 MiB prefix cache and 256 MiB
allocator cache. The three BF16 prefix states, four shared attention leaves,
pre-mixes and exact token keys should total about 645 MiB. Expected training
cache counters are three misses and nine hits, with no eviction. These are
predictions to verify in the actual training receipt, not measured results.

## Concrete invocation

Run after freezing the final tested source. The module invocation avoids an
editable install resolving to another tree. Set `PYTHONPATH` to that immutable
runtime's `src` and use the existing pinned environment's Python.

```sh
python -m exl3_mlx.ktune_cli finetune \
  --source /Users/jasontitus/experiments/exl3-qlora/apple/artifacts/ktune-20261012/source \
  --model /Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw \
  --parent-repo jason-m5-macbook:/Volumes/Scratch/model-work/metaltune/data/repos/mlxfast-qwen38-125b-a6b-engine \
  --context-mode parent_files --tier official_promoted \
  --case-id official:qwen38-125b-a6b-mlx-v1:c0e42c7e-268c-454f-a9cb-e705f952b819 \
  --case-id official:qwen38-125b-a6b-mlx-v1:1a621870-4993-4c43-9472-6f7d8f745140 \
  --case-id official:qwen38-125b-a6b-mlx-v1:8a690716-8068-4da8-9532-26047c96aa9c \
  --max-examples 3 --adapter-layers 39 --steps 12 --learning-rate 1e-4 \
  --rank 4 --activation-mode checkpoint --chunk-size 8192 \
  --profile development --memory-gib 80 --reserve-gib 12 \
  --prefix-cache-mib 2048 --fit-examples 3 \
  --output /Users/jasontitus/experiments/exl3-qlora/apple/artifacts/ktune-20261012/standard-three-case
```

Add `--prepare-only` for an inspection checkpoint before starting the standard
project through `SFTTrainer.from_project(project).train(compare=False)`. Do not
use `--verify-fit` if the follow-up below is used; it would repeat evaluation.
For explicitly fresh uncached base and adapter scores, after successful step12:

```python
from exl3_mlx.sft import SFTTrainer
trainer = SFTTrainer.from_project(project, resources={"prefix_cache_mib": 0})
fit = trainer.verify_training_fit()
```

The standard verifier starts separate new base and adapter processes. It now
requests per-token metrics and binds exact source/runtime identity, selected
rows, data SHA, command, model, checkpoint, activation mode, supervised count,
and per-row target IDs. Keep the generated fit receipts. No extra pre-training
baseline pass is necessary: the fresh zero-B base is independent of initialization
A because B is zero, with the same checkpoint activation policy and chunk size.
A full baseline and final verification still require six cold backbone passes;
three cold training prefixes and twelve final-block updates put the rough total
at 65–85 minutes using prior timings. This is a planning range, not a benchmark.

## Audit result and bounded fix

A real standard-tool blocker was found by the full suite: the cache's Python
array-ID guard rejected `nn.value_and_grad`, which temporarily wraps even frozen
arrays. It is fixed using evaluated shared-memory storage identity plus original
owner retention. The latter prevents allocator-address reuse from accepting
changed weights. The focused receipt `prefix-cache-final-tests.xml` records
21 passed in 2.31 seconds: 11 CPU and 10 Metal cases. F32/BF16 cold/hit loss,
gradient, Adam state and parameter equality, suffix reload, expert boundary,
mode/token separation and true frozen mutation rejection pass. Independent
process review found no additional blocker. The parent runs the complete suite
before creating the final runtime.

The immutable-prefix contract still prohibits in-place frozen tensor mutation,
private behavior changes, and cached-array writes. Cache state is private,
model-local, bounded, excluded from trainable/checkpoint tensors, and not saved
across restart. Base metadata/stat guards do not constitute full weight-payload
hash verification.

## Remaining actual-cache verification gap

At review time, `train_run` records pre-update per-example loss, while the standard
fit check starts fresh evaluation processes. Fresh final scores certify the
saved adapter's behavior, but by themselves do not compare actual-model warm
cache output to uncached output at identical final parameters. The ten small
Metal tests establish that parity only for fixtures.

A low-cost optional final probe can reuse the three training-mode prefix entries
without any update: score all complete training rows at final parameters while
preserving training mode, store per-row supervised-logit hashes and scores, and
compare them with the fresh uncached evaluation receipt. Ordinary `evaluate`
switches to eval mode and would cold-miss its separate keys, adding three full
passes; do not silently change mode-key policy. This gap and the bounded probe
proposal were sent to the parent before runtime freeze. If no probe is added,
report that actual reusable-cache parity remains unverified, rather than claiming
that fresh adapter scoring alone proves it.

Judge the aggregate training NLL and every per-case NLL, with added-code token
metrics reported separately where extraction is reliable. An aggregate gain
can conceal a regression on a smaller case. Generated edit validity, application,
compilation, numerical kernel correctness, latency and generalization are
separate tests; a likelihood decrease certifies none of them.

## Follow-up implementation status

The parent authorized the bounded final probe for runtime-v3 after freezing
runtime-v2 for the independent causal experiment. `training.evaluate` now accepts
`preserve_mode=False`; its default behavior is unchanged. Opting in preserves the
current model mode while scoring and restores that entry mode on exceptions.
This keeps the training prefix key and performs no gradient/optimizer operation.
The process-review agent owns the off-by-default `final_fit_examples` workflow,
SFT configuration and warm-versus-fresh receipt check; the quality-review agent
independently audits it.

Eleven numerical regression cases were added: an eight-case mode/exception grid,
invalid-option rejection, and F32/BF16 three-case final warm-score parity with a
separately constructed uncached model carrying the same final adapters. They
also require unchanged parameters/optimizer state, exact per-row targets and
hashes, and absence of additional frozen-prefix calls. Ruff and compilation pass;
these new cases remain pending the parent's coordinated GPU slot. The earlier
21-case cache receipt predates this final-probe feature. A three-row final probe
adds three hits, so expected training-receipt cache totals become three misses
and twelve hits, with no eviction.
