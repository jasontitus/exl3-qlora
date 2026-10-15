# Profile-guided rotation pilot: completed development screen

Keep original458 as the working adapter. Stop this 24-update pilot branch; do not extend it or open confirmation automatically. This screen did not demonstrate an improvement from the targeted pilot. That is a practical selection decision for the current queue, not a causal claim that tuning harms reasoning or a general ranking of adapters.

The task is clockwise 90-degree strided F32 bit copy, in the same index-permutation family as the earlier transpose task. Four predeclared slots compared original458 and its targeted step24 continuation, each with a real baseline profile present or omitted. Each slot received one greedy attempt with an 8192-token context and 4096-token output allowance, the same native engine, and a fresh private environment. Omitted prompts had 1263 tokens; present prompts had 2128. All four ended with EOS and valid complete files; none was repaired or retried.

The pilot continued original458 weights with a fresh optimizer, using three new training records repeated three times plus fifteen replay rows (24 updates). The new records represent two training lineages. The rotation prompts, profiles and generated outputs were excluded from training. The limited exact/normalized source and marker screen does not establish absence of semantic overlap or base-model pretraining exposure.

| Registered slot | Output | Correctness evidence | Full-mix baseline/candidate GPU-time ratio |
|---|---|---|---:|
| Original458, profile omitted | Scalar xy, 16×16 dispatch | 12/12 exact | 2.343× |
| Pilot, profile present | Exact baseline kernel and dispatch bytes | Reuses qualified baseline identity | Not measured: unchanged output |
| Original458, profile present | Scalar xy, 16×16 dispatch | 12/12 exact | 2.297× |
| Pilot, profile omitted | Scalar xy, 16×16 dispatch | 12/12 exact | 2.315× |

Each changed kernel was compiled unchanged and checked on all twelve cases, including every output bit, padding, signed zero, subnormals, input identity and canaries. Each then completed 840 timing batches / 84,000 dispatches. The pilot without a profile is whitespace-equivalent to the original458 omitted kernel; its exact bytes differ, so it received its own checks and timings. The profile-present pilot returned both baseline files byte-for-byte, so no redundant candidate run or invented 1× measurement is reported.

The full-mix ratios above divide summed baseline GPU durations by summed candidate GPU durations over all twelve cases and all thirty measured pairs. They are not averages of selected speedups. Every run used five warmup pairs and thirty alternating baseline/candidate measured pairs, with 100 dispatches per batch, on Apple M1 Ultra. These warm pooled-buffer GPU measurements are not end-to-end application latency. The three candidate runs occupied separate windows; small differences among their ratios do not establish an adapter or profile advantage.

## All twelve measured workloads

Values are medians of thirty paired baseline/candidate ratios; below 1 would mean a slowdown. The unchanged pilot/profile-present cell has no separate timing column.

| Case | Rows×cols | Original458, omitted | Original458, present | Pilot, omitted |
|---|---|---:|---:|---:|
| case-00 | 1×1 | 1.122× | 1.052× | 1.164× |
| case-01 | 1×47 | 1.118× | 1.055× | 1.149× |
| case-02 | 47×1 | 1.117× | 1.052× | 1.160× |
| case-03 | 15×47 | 1.140× | 1.081× | 1.171× |
| case-04 | 47×15 | 1.118× | 1.063× | 1.153× |
| case-05 | 79×81 | 1.137× | 1.087× | 1.166× |
| case-06 | 80×80 | 1.141× | 1.086× | 1.170× |
| case-07 | 191×193 | 1.405× | 1.359× | 1.432× |
| case-08 | 383×47 | 1.129× | 1.076× | 1.151× |
| case-09 | 384×2048 | 2.596× | 2.596× | 2.588× |
| case-10 | 768×2048 | 3.205× | 3.149× | 3.100× |
| case-11 | 1536×1024 | 3.000× | 3.216× | 3.012× |

[comparison.json](comparison.json) retains all four outcomes, all twelve workloads, all thirty paired samples per measured workload, source hashes and reviewed rationale limitations. [workloads.json](workloads.json) retains exact strides and offsets; its large binary fixtures remain in the original evidence tree.

## Decision explanation and limits

The pilot/profile-present explanation claimed that small and medium grids dominated the workload. Those account for nine cases by count, but the three large cases account for **83.9547% of the measured baseline GPU duration** under the declared equal-case schedule. That specific explanation is contradicted by the supplied profile. No new numerical rationale score is assigned, and this observation is not proof of a general reasoning regression. No occupancy, bandwidth or cache counters were collected.

Original458 selected the same scalar xy mechanism with and without the profile. Its profile-present explanation referred to the large cases, but the separate timing windows do not show that supplying the profile improved the implementation. The pilot without a profile likewise produced that mechanism, with no demonstrated new optimization. The useful result is that the changed kernels beat this scalar baseline on this fixed mix; it is not evidence of a new algorithmic-family generalization.

Confirmation remains unopened. No further training or generation is authorized by this report. Any proposed training repair must use independent train-only evidence, not these exact development prompts, generated answers or measured results. All unsuccessful, unchanged and successful outcomes remain in the four-slot denominator.

## Evidence and reuse

- [protocol.json](protocol.json): exact prospective protocol, including the unused confirmation boundary.
- [baseline-profile.json](baseline-profile.json): actual baseline-only profile supplied before generation; no candidate times or invented counters.
- [input-spec.json](input-spec.json): exact four-cell evidence bindings consumed by the independently reviewed comparison builder.
- [provenance.json](provenance.json): original paths, SHA-256 identities, generator/review bindings and this compact publication inventory.
- [sources/](sources/): exact generated kernel, dispatch and response snapshots, plus the baseline. They are artifacts to inspect; this report does not execute them.

The comparison JSON is a byte-identical export of the reviewed builder's final output. Full audit replay still requires the original evidence paths listed in its inputs; this compact publication does not copy model weights, binary fixtures or native executables. Rerunning generation or timing requires a fresh reviewed owned request and resource admission. No automatic run is embedded here.
