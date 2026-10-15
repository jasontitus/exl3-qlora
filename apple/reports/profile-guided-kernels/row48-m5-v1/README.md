# Measured Q6_K decode reuse on M5 Max

One existing source change, fifteen workload variants. Both kernels passed independent packed-weight numerical references and produced bitwise-identical initial outputs (74,144 values checked across both arms).

| Weight shape | Batch 1 | Batch 3 | Batch 4 |
| --- | ---: | ---: | ---: |
| 512 × 4096 | 1.077× | 1.617× | 1.738× |
| 4096 × 4096 | 1.107× | 1.342× | 1.843× |

Values are paired median before/after GPU-time ratios; greater than one favors the change. All nine earlier tiny variants regressed, with ratios 0.790–0.958. All results are retained in [timing-contrast.json](timing-contrast.json).

The separate profiling policy uses four warmups and twenty paired rounds. The larger sweep followed the tiny regressions and also changed synthetic input/scale distributions. It is exploratory, does not isolate size as the cause, and supplies no measured hardware-counter explanation. M5 timing must not be combined with Studio timing or described as end-to-end engine throughput.

This evidence supports preparing apply-change and measured-feedback/retain-baseline examples, grouped in one training lineage. Packaging, source context and target review remain pending; these are not fifteen independent optimizations or an automatically training-ready dataset. [Evidence identities](evidence.json).
