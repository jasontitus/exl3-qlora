# Strided-transpose development task

The fixed GPU controls are independently qualified on Studio. Baseline and positive passed all twelve cases. Deliberately wrong packed-stride and tail implementations failed their exact expected nine and eight cases. The audit replayed 14,916,764 exported output words, including padding. All 48 dispatches completed; observed process peak was 209,551,360 bytes and the owned GPU lease was released.

This verifies the numerical harness, not a model improvement or a faster positive control. Separate paired timing and same-prompt base/old458 generations are being prepared. The protocol uses 789 fresh prompt tokens, a 4096-token allowance and exact complete-file parsing. This is a development task; it stays out of training.

The malformed-metadata failure in host-v1 remains preserved. Host-v2 rejects wrong schema/types/IDs/roles and missing or invalid sources before device creation. Reusable CPU geometry and response/replay support are committed as f2b48726 and452326e8. [Control evidence](controls.json).
