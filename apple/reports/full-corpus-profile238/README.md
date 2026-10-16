# Full-corpus adapter optimization evaluation

The new adapter passes every fixed correctness test and shows a positive RMS performance signal. Rotation and transpose remain effectively comparable to original458. Profile-guided reasoning improvement is not demonstrated.

The endpoint is original458 plus 238 additional updates (696 in this lineage), layer 39 attention LoRA rank 4/alpha 8. This pass used 229 corpus rows plus three measured-profile records repeated 5/3/1, initialized from original458 weights with fresh Adam. The smaller failed pilots are separate branches.

## Results

All four responses ended naturally, parsed as complete files, compiled unchanged, and passed their fixed numerical suites: 224 RMS cases and 12 each for transpose, rotation/profile-present and rotation/profile-omitted. There were no generation retries or source repairs. Existing matching original458 generations were reused. Numerical and timing execution, resources and owned GPU release were independently audited.

These figures are ratios of summed measured baseline/candidate GPU durations across 12 workloads with equal sample counts. They measure speedup over each run's supplied baseline, not an interleaved old/new adapter race or a production traffic distribution.

| Evaluation | Original458 | New full238 |
|---|---:|---:|
| Row RMS statistic |10.908×|12.310×|
| Strided transpose |2.264×|2.260×|
| Rotation, profile supplied |2.297×|2.310×|
| Rotation, profile omitted |2.343×|2.324×|

RMS candidate median latency was lower on 9/12 workloads and higher on 3/12; those three increases were approximately 0.5–1.25%. The new RMS kernel uses an explicit 32-lane shuffle tree; original458 uses simd_sum. Both passed 224 cases. This is a useful positive signal on this RMS task. Separate execution windows, baseline drift and one deterministic response per condition limit causal interpretation. All 12 workloads and dispersion, including regressions, are retained in the linked JSON comparisons.

Rotation and transpose still use guarded scalar XY 16×16 kernels. Rotation chose the same mechanism with and without the actual baseline profile, giving no demonstrated profile-input benefit. The transpose rationale incorrectly claims neighboring x lanes make contiguous output writes; they write with the output stride. The code and false explanation are preserved separately. Correctness and speed do not establish better reasoning.

## Decision and limits

Keep full238 as a promising RMS challenger and original458 as the comparison reference. This panel does not establish a broad replacement winner. It does establish that the extra pass preserved useful generated optimizations and produced a somewhat stronger observed RMS result. Do not automatically add another epoch based on this result.

This is repeated development evaluation over three task families, with four generated conditions, not an unseen benchmark. Confirmation remains sealed. Source collection and recurring automation notices remain paused. No further training was launched.

The common timing policy was 5 warmup pairs, 30 measured pairs and 100 dispatches per batch on all 12 workloads, on the Apple M1 Ultra. Four timing jobs completed 336,000 dispatches. Each candidate was checked for correctness before timing; controls were reused, not rerun.

- [Machine-readable summary](evaluation-result.json)
- [RMS: every workload and original458 comparison](rms-comparison.json)
- [Rotation: every workload and profile comparison](rotation-comparison.md)
- [Transpose: every workload and original458 comparison](transpose-comparison.json)
- [Training completion](training-completion.json)

The run preserved provenance failures and fixed stale source references/condition bindings in the evaluation preparation. Frozen prompts, runtimes, model weights, historical results and generated candidate code were unchanged. These operational repairs do not count as model-quality gains.
