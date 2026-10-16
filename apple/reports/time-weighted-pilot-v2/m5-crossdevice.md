# M5 Max cross-device validation

The exact step24/profile-present scalarXY16×16 rotation kernel and supplied baseline were rebuilt on Apple M5 Max using its installed Metal/Swift toolchain. This is a kernel/workload validation, not new adapter training or a comparison of model quality. Studio and M5 timing windows remain separate.

All12 cases in both arms matched an independent uint32 oracle, including output padding:8,068,646 words, with unchanged input and intact canaries. The numerical run's observed peak process memory was223,330,304bytes.

Timing used5warmup batches,30 alternating paired rounds and100dispatches perbatch across all12 workloads. All840 batches/84,000dispatches passed reference-hash and safety checks. The observed timing-process peak was230,080,512bytes. Owned process closure and raw global/local release evidence were independently checked.

| Workload | Median of30 paired baseline/candidate duration ratios |
| --- | ---: |
| 384×2048 | 2.352762× |
| 768×2048 | 2.273793× |
| 1536×1024 | 2.158066× |

Across the equal-invocation12-case mix, the ratio of summed measured baseline durations to summed candidate durations was1.99209056×. This is not an average of per-case speedups, measured production traffic, or evidence of a causal advantage for the trained adapter. Original458 already emits the same optimization mechanism. Small-case latency and single-window ratios should be interpreted within this bounded test.

[All workload results](m5-timing-results.json) and the [independent timing review](m5-timing-review.json) preserve the actual source/build/workload identities and evidence. No generated shell was executed and no candidate code was repaired into success. The new runs use separate M5 artifacts and receipts; they do not replace the original Studio measurements.
