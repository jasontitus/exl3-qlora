# Time-weighted profile pilot

## Step24 result

Keep original458. The revised pilot generated correct scalar XY16×16 rotation kernels with and without the profile, reproducing the useful mapping original458 already emits. There is no demonstrated replacement benefit on this repeated development task.

| Condition | Baseline/candidate total GPU duration | Evidence |
| --- | ---: | --- |
| original458, profile omitted | 2.34305× | Original reference window |
| original458, profile present | 2.29737× | Original reference window |
| time-weighted step24, profile present | 2.35334× | New measurement |
| time-weighted step24, profile omitted | 2.31538× | Historical measurement of byte-identical kernel and dispatch |

Each ratio sums the 360 measured per-dispatch durations across all12 workloads, with equal invocation weight per case. It is not an average of speedups or measured production traffic. Separate execution windows and the same generated mechanism do not establish a causal adapter advantage. Historical reuse adds no independent measurement.

The new profile-present candidate passed all12 numerical cases plus baseline controls, then840 batches/84,000 dispatches under the existing5warmup/30paired-round/100dispatch policy on Apple M1 Ultra. Three large-case paired median ratios were2.6290×,3.2251× and2.9567× against the supplied baseline. Exact inputs, output padding, canaries and process release were audited. Compilation and numerical correctness precede timing.

The rationale discusses scalar division/modulo and alignment, but does not establish memory/launch bottlenecks through counters or demonstrate a better time-weighted decision than original458. The unused confirmation remains sealed.

## Training and cost

The24 updates used18 distinct examples:15 unchanged replay cases, five exposures to a rewritten positive cost-aware example, three measured-retention exposures and one restricted precision example. These are two focused source lineages, only one speed-qualified. Workload mixes are explicitly designed; proposal prompts contain baseline-only measurements and label hypothetical savings. Retention examples explicitly identify post-test evidence. No evaluation prompt, output or oracle entered training.

A disposable two-item cold probe took481.75seconds and wrote two exact training prefixes. Main24 restarted original458 with fresh Adam and took488.08seconds end-to-end; peak MLX41.023GiB, exact saved-adapter reload. All24 training forwards and both reload forwards hit the frozen-prefix cache;19 disk hits, zero misses/writes. Cold plus main cost16.16minutes. These timings exclude data preparation, reviews and generation.

## Authorized second epoch

The user requested stronger training because cached updates are inexpensive. A separate continuation restores adapter, Adam, RNG and cursor24 and adds24 updates to reach48, preserving step24. It uses the same frozen data, runtime and guards. Completion and generated quality are reported separately after their actual audits; no third epoch is automatically authorized. Source collection remains paused and Qwen parked.

## Reproduction

[Step24 comparison receipt](step24-comparison.json) preserves source identities, all per-case ratios, exact association of reused evidence, and the independent selection decision. Local execution artifacts are under `kernel-library/next-round-v5/time-weighted-pilot-v2`; native generation uses accepted engine800b and the same2128/1263-token profile-present/omitted prompts. Training uses pinned runtime-v5, MLX_ENABLE_TF32=0, layer39 attention rank4/alpha8, LR2.5e-5, chunk2048 and80GiB process/8GiB reserve.
