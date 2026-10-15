# RMS generated-kernel evaluation: partial result

The old458 adapter generated an unchanged SIMD RMS kernel that passed all 224 numerical cases and beat the supplied scalar baseline on all 12 timed workloads on Apple M1 Ultra. The independently audited paired median speedups range from 5.06× to 12.19×. This is a useful generated kernel; the comparison against the base model and newer adapters is still pending.

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

The same-protocol base, final188 and code152 generations are already complete. Base numerical checks passed; newer candidates still need their numerical/timing results. Do not regenerate completed samples. One task is provisional; retain earlier affine failures and confirm any chosen adapter on a distinct task family before broader claims.

[Machine-readable evidence](old458-timing.json) records the source paths, hashes, exact workload results and limits.
