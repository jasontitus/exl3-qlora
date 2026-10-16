# Full238 versus original458: rotation development comparison

All four exact candidates passed numerical checks and completed the same12-workload timing policy. Full238 emits the same scalar guarded `xy16×16` mechanism as original458. These separate timing windows do not establish a fine-tuning or profile-input advantage. Confirmation remains unused.

The table retains every workload. Values are medians of30 paired baseline/candidate duration ratios; greater than1 means faster than the scalar curator baseline in that run.

| Workload (rows × cols) | Original458 present | Original458 omitted | Full238 present | Full238 omitted |
|---|---:|---:|---:|---:|
| case-00 (1 × 1) | 1.0522× | 1.1222× | 1.1136× | 1.0390× |
| case-01 (1 × 47) | 1.0551× | 1.1184× | 1.1109× | 1.0410× |
| case-02 (47 × 1) | 1.0518× | 1.1170× | 1.1167× | 1.0506× |
| case-03 (15 × 47) | 1.0809× | 1.1396× | 1.1305× | 1.0679× |
| case-04 (47 × 15) | 1.0630× | 1.1180× | 1.1085× | 1.0398× |
| case-05 (79 × 81) | 1.0872× | 1.1369× | 1.1260× | 1.0700× |
| case-06 (80 × 80) | 1.0860× | 1.1409× | 1.1297× | 1.0464× |
| case-07 (191 × 193) | 1.3595× | 1.4049× | 1.4114× | 1.3350× |
| case-08 (383 × 47) | 1.0760× | 1.1288× | 1.1302× | 1.0591× |
| case-09 (384 × 2048) | 2.5964× | 2.5965× | 2.5917× | 2.5695× |
| case-10 (768 × 2048) | 3.1485× | 3.2049× | 3.0564× | 3.3214× |
| case-11 (1536 × 1024) | 3.2161× | 3.0003× | 2.9315× | 3.0331× |

The equal-case ratio of summed measured durations (all360 measured pairs per cell) is:

- old458/profile_present: 2.297371×.
- old458/profile_omitted: 2.343053×.
- full238/profile_present: 2.310331×.
- full238/profile_omitted: 2.324445×.

This aggregate is not an average of per-case ratios, a best-case selection, or a direct interleaved adapter race. Original458 and full238 were measured in different windows; generation remains one deterministic attempt per condition. Full data, exact inputs and actual review links are in [comparison.json](rotation-comparison.json). Each linked replay includes all35 pairs per workload including warmups. No timing was fabricated or copied into a new measurement condition.

No additional generation, training, confirmation or automatic winner selection is triggered by this report.
