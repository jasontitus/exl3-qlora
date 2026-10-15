# Strided-transpose development task

The fixed GPU controls are independently qualified on Studio. Baseline and positive passed all twelve cases. Deliberately wrong packed-stride and tail implementations failed their exact expected nine and eight cases. The audit replayed 14,916,764 exported output words, including padding. All 48 dispatches completed; observed process peak was 209,551,360 bytes and the owned GPU lease was released.

This verifies the numerical harness, not a model improvement or a faster positive control. Separate paired timing and same-prompt base/old458 generations are being prepared. The protocol uses 789 fresh prompt tokens, a 4096-token allowance and exact complete-file parsing. This is a development task; it stays out of training.

The malformed-metadata failure in host-v1 remains preserved. Host-v2 rejects wrong schema/types/IDs/roles and missing or invalid sources before device creation. Reusable CPU geometry and response/replay support are committed as f2b48726 and452326e8. [Control evidence](controls.json).

## Paired control timing

Independent audit accepted 840 batches / 84,000 dispatches on M1 Ultra, using five warmups and 30 alternating paired rounds with 100 dispatches per batch for each of twelve cases. All batches retained exact outputs, padding and input checks. The supplied optimized control achieved paired median speedups of 2.348×, 2.877× and 2.967× on the three larger shapes; smaller cases ranged from 1.056× to 1.114×. See [all twelve outcomes](control-timing.json). Peak observed aggregate RSS was 238,174,208 bytes, and owned release was verified.

This establishes optimization headroom for the task. It is curator control performance, not an adapter result; generated implementations remain separately gated. Review SHA: `37ce89a3df30bc639d7bf0cca70aa647e500848035c17905b6bb6e99f7a6c1f4`.

## Generated candidates

Base and old458 each produced a valid complete-file response, compiled without repair and passed all twelve exact numerical cases, including output padding. Independent audits replayed 7,458,382 words per condition across its baseline/candidate outputs and verified owned release. Both use a scalar transpose mapping and 16×16 dispatch. Their numerical success is established; their speed comparison remains pending. See [candidate numerical receipts](candidate-numerics.json).
