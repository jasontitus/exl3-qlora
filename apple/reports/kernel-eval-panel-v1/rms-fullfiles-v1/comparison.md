# RMS generated-kernel evaluation: four completed conditions

The old458 adapter generated an unchanged SIMD RMS kernel that passed all 224 numerical cases and beat the supplied scalar baseline on all 12 timed workloads on Apple M1 Ultra. The independently audited paired median speedups range from 5.06× to 12.19×. The base model also produced a correct kernel, but its audited improvement over the supplied scalar baseline was only 1.016×–1.193×. The older adapter produced the better optimization on this task. Both newer adapters also passed correctness and completed timing. Old458 is the provisional choice for the next distinct task; neither newer trial improved this generated optimization.

Timing: 5 warmup batches per variant,30 alternating paired rounds,100 dispatches per batch. These are warm amortized GPU timings, not end-to-end engine throughput.

| Workload | Shape (rows × cols) | Baseline µs | Old458 µs | Paired median speedup |
| --- | --- | ---: | ---: | ---: |
| case-09-uniform-seed0 | 32 × 4096 | 500.621 | 41.889 | 11.950× |
| case-09-uniform-seed17 | 32 × 4096 | 501.375 | 42.264 | 11.886× |
| case-09-uniform-seed2026 | 32 × 4096 | 515.372 | 41.734 | 12.194× |
| case-10-uniform-seed0 | 256 × 128 | 33.784 | 6.502 | 5.166× |
| case-10-uniform-seed17 | 256 × 128 | 34.088 | 6.569 | 5.062× |
| case-10-uniform-seed2026 | 256 × 128 | 33.774 | 6.474 | 5.205× |
| case-11-uniform-seed0 | 256 × 1024 | 195.485 | 20.186 | 10.639× |
| case-11-uniform-seed17 | 256 × 1024 | 192.303 | 21.219 | 10.023× |
| case-11-uniform-seed2026 | 256 × 1024 | 197.640 | 20.968 | 9.636× |
| case-12-uniform-seed0 | 256 × 4096 | 550.124 | 46.140 | 11.866× |
| case-12-uniform-seed17 | 256 × 4096 | 563.535 | 46.884 | 10.699× |
| case-12-uniform-seed2026 | 256 × 4096 | 573.174 | 46.872 | 11.664× |

All 840 batches / 84,000 dispatches passed per-batch output, canary and input-immutability checks. Independent review replayed timestamps, paired summaries, source and device bindings, memory observations and owned release. Sampled process peak was 531,709,952 bytes.

All four same-protocol generations and 224-case correctness checks are complete. All four timing runs are independently accepted. Do not regenerate completed samples. One task is provisional; retain earlier affine failures and confirm any chosen adapter on a distinct task family before broader claims.

[Machine-readable evidence](old458-timing.json) records the source paths, hashes, exact workload results and limits.

[Audited base/old458 comparison](base-old458-comparison.json): candidate median times from separate execution windows differ by 4.54×–12.38× in favor of old458; the curator baseline itself drifted 0.913×–1.128× between windows. This is a descriptive comparison, not a directly interleaved old458/base measurement or a generalization claim.

## Completed adapter comparison

| Condition | Exact response | Numerical cases | Paired median speedup over supplied kernel | Generated strategy |
| --- | --- | ---: | ---: | --- |
| Base | Valid complete files | 224/224 | 1.016–1.193× | Scalar Kahan |
| Old458 | Valid complete files | 224/224 | 5.062–12.194× | SIMD32 reduction |
| Final188 | Valid complete files | 224/224 | 1.091–1.194× | Scalar Kahan |
| Code152 | Valid complete files | 224/224 | 1.115–1.286× | Scalar block64 reduction |

Ranges retain all twelve workloads (four shapes × three seeds). Every condition ran one deterministic generation under the same prompt/allowance and the same numerical/timing policy. Candidate-to-candidate timings use separate execution windows; each is paired against the same supplied implementation in its own window. [Full audited comparison](four-condition-comparison.json).

Keep old458 as the provisional practical candidate and test the distinct strided-transpose family next, then an unused confirmation case. Do not automatically extend code152 or final188, or launch rank8 merely because training is idle. Their current outputs are correct but choose much weaker parallelization on this task. This does not prove that code-focused loss or a lower learning rate is generally worse: datasets, masks, exposures and training duration differ, and this is one response per adapter on one task.

In parallel, use independently measured training pairs to improve workload-conditioned decisions. The M5 Q6_K sweep now supplies actual successes and regressions from one source lineage. Packaging and a bounded cold-prefix pilot precede any new profile-rich training. Earlier affine format/numerical failures remain part of the overall assessment. No general kernel-optimization or profile-reasoning benefit is claimed yet.
