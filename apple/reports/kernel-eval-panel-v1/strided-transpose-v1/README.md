# Strided-transpose development task

The fixed GPU controls are independently qualified on Studio. Baseline and positive passed all twelve cases. Deliberately wrong packed-stride and tail implementations failed their exact expected nine and eight cases. The audit replayed 14,916,764 exported output words, including padding. All 48 dispatches completed; observed process peak was 209,551,360 bytes and the owned GPU lease was released.

This verifies the numerical harness, not a model improvement or a faster positive control. Separate paired timing and same-prompt base/old458 generations are being prepared. The protocol uses 789 fresh prompt tokens, a 4096-token allowance and exact complete-file parsing. This is a development task; it stays out of training.

The malformed-metadata failure in host-v1 remains preserved. Host-v2 rejects wrong schema/types/IDs/roles and missing or invalid sources before device creation. Reusable CPU geometry and response/replay support are committed as f2b48726 and452326e8. [Control evidence](controls.json).

## Paired control timing

Independent audit accepted 840 batches / 84,000 dispatches on M1 Ultra, using five warmups and 30 alternating paired rounds with 100 dispatches per batch for each of twelve cases. All batches retained exact outputs, padding and input checks. The supplied optimized control achieved paired median speedups of 2.348×, 2.877× and 2.967× on the three larger shapes; smaller cases ranged from 1.056× to 1.114×. See [all twelve outcomes](control-timing.json). Peak observed aggregate RSS was 238,174,208 bytes, and owned release was verified.

This establishes optimization headroom for the task. It is curator control performance, not an adapter result; the generated implementations have their own numerical and timing receipts below. Review SHA: `37ce89a3df30bc639d7bf0cca70aa647e500848035c17905b6bb6e99f7a6c1f4`.

## Generated candidates

Base and old458 each produced a valid complete-file response, compiled without repair and passed all twelve exact numerical cases, including output padding. Independent audits replayed 7,458,382 words per condition across its baseline/candidate outputs and verified owned release. Both use a scalar transpose mapping and 16×16 dispatch. Their numerical success is established; their completed speed comparison follows below. See [candidate numerical receipts](candidate-numerics.json).

## Generated-kernel paired timing

Both timing runs are independently accepted: 840 batches / 84,000 dispatches each, all twelve cases, unchanged 5/30/100 policy. Each candidate ran in a separate window paired against the same supplied baseline; these are not direct interleaved base-versus-adapter measurements. Ratios below are medians of paired baseline/candidate times; below 1 means a slowdown.

| Case | Rows×cols | Base | Old458 |
|---|---|---:|---:|
| case-00 | 1×1 | 0.997× | 0.996× |
| case-01 | 1×33 | 1.142× | 1.100× |
| case-02 | 31×1 | 0.911× | 1.132× |
| case-03 | 17×33 | 1.087× | 1.113× |
| case-04 | 33×17 | 0.946× | 1.171× |
| case-05 | 63×65 | 0.949× | 1.443× |
| case-06 | 64×64 | 1.056× | 1.190× |
| case-07 | 127×129 | 1.112× | 1.069× |
| case-08 | 257×31 | 1.166× | 1.147× |
| case-09 | 512×1024 | 2.496× | 2.455× |
| case-10 | 1024×1536 | 2.934× | 2.919× |
| case-11 | 1536×1024 | 3.187× | 3.253× |

Both kernels deliver similar large-shape improvements. This test does not establish a meaningful large-shape adapter advantage; preserve all small-shape outcomes rather than selecting only wins. Old458 remains provisional based on its stronger RMS result. An unused confirmation task remains before broad claims. See [all paired summaries](candidate-timing-comparison.json).
