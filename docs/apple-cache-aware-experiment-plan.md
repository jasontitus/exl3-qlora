# Cache-aware experiment strategy

User preference: when expected quality benefit is comparable, choose the experiment that can reuse a verified cache. An uncached trial needs an explicit reason to expect enough additional benefit to justify its measured cost. This prioritizes experiments; it must not substitute stale activations, reduce held-out coverage or turn lower training loss into a functional-quality claim.

The completed layer39 run measured20.6384hours for its first229 presentations and1.55166hours for the repeat229. These are observed epoch totals, not a guaranteed cross-run speed or isolated cache effect. Cold identity/model construction costs are additional to the training timer.

## Immediate order

1. Finish checkpoint458 evaluation against the already completed full16 baseline. Historical rank8/alpha16 provenance stays explicit; four actual rank4 zero-B cases exactly reproduce11077targets and full supervised-logit hashes.
2. Qualify reuse of the existing M5 boundary39 cache across a fresh independent experiment. Check immutable model/implementation/platform/arithmetic binding, exact full token keys and train/recompute mode. Record cache coverage and actual disk/RAM hits before claiming fast cross-run training.
3. Prefer boundary39 trials varying learning rate, rank/alpha, epoch count or sampler order. Each trial resets adapters, optimizer, RNG and sampler independently. Changing suffix configuration must not change frozen-prefix behavior. New texts require new entries; sampling the same exact items does not.
4. Prepare boundary38 cold/warm pilot only as a measured capacity/cost extension. The earlier proposal for layers38–39 remains preparation, not a launched campaign. Broader scope needs a clear expected-benefit argument relative to the available cached trials.
5. Build additional boundary families36/38 only after that comparison. Never use a boundary39 hidden state as input before38 or36.

## Existing mechanism versus required product work

The trainer already has immutable checksummed disk entries and a bounded RAM tier. Keys include complete token IDs, training mode and recompute mode. Disk bindings include full model hashes, implementation, platform/device, quant/activation configuration, chunk size, boundary and state layout. Suffix LoRA rank is not itself a prefix-key field. Compatibility must be established by actual binding/row checks, not directory names. Training/evaluation caches are intentionally distinct; the current platform binding prevents assuming M5 and Studio entries are interchangeable.

Still needed: a cache coverage/compatibility preflight; explicit cache-family inventory; independently qualified fresh-run reuse; a resident multi-trial worker that preserves frozen objects while resetting all learned/optimizer/random/sampler state; trial receipts reporting cold work, warm work, hits/misses/bytes and end-to-end cost. A future multi-boundary producer could capture36/38/39 during one base forward pass, but is not implemented or measured.

A persistent worker must revalidate frozen identity and prevent cross-trial learned-state leakage. Frozen-prefix changes invalidate descendants; suffix changes should not unnecessarily invalidate independently qualified prefix entries. Narrower implementation identities are a future reviewed change, not permission to bypass the current full binding or hash scans. Cache quotas must fit the entire intended working set or report misses/refills explicitly. Do not change sealed runtime-v5 to add these features.

## Decision and quality records

For each proposed trial record expected benefit and its evidence, earliest trainable boundary, exact corpus identity, compatible cached rows/bytes, missing rows, measured cold/warm cost, resource requirements, and quality endpoint. Within a comparable-benefit group, prioritize verified complete-cache coverage, then partial coverage, then cold work. Do not assign false numerical quality probabilities. Optimize useful generated-kernel improvement per wall-clock budget, retaining correctness and compiler/runtime evaluation.

Source collection remains paused. Existing final baselines, held-out data, unrelated workloads and memory reserves remain preserved.
