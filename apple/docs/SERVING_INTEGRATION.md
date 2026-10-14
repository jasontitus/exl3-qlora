# Training and serving integration

The Apple trainer produces private EXL3 LoRA checkpoints and evaluates their
supervised loss. Neither training nor loss evaluation requires a serving engine.
The trainer also has a generation path that recomputes the prefix; it is not a
cached serving implementation.

Generated-answer evaluation is being integrated with an isolated build of
`mlx-serve` linked to patched `mlx-stream`. This extends existing inference
software to read the private adapter and apply the learned attention-projection
updates. It is not a replacement inference engine. Loading the base model in
unmodified serving software does not establish support for this adapter format.

As of 2026-10-14, the repaired isolated build passed one absent-adapter
request: a 33-token prefill followed by four generated tokens. The repair
retains the device embedding across the prefill/decode handover, includes it
in decode memory accounting, and emits the six required execution-profile
fields. Independent source, CPU, artifact and actual-run reviews are retained.
Earlier route and report failures are preserved. This qualifies the short
baseline diagnostic; trained-adapter application and general serving
compatibility remain unqualified. Training and loss evaluation were separate
from these serving failures.

Before claiming serving compatibility, verify the absent-adapter route, a
zero-B adapter against that same base, and the trained adapter on the exact
checkpoint. Verify projection names, orientation, scaling and application across
prefill and decode. Compare cached and fresh computation under an explicit
numerical contract. Successful private integration still does not establish
PEFT, unmodified MLX-LM, or general HTTP serving compatibility.

Lower held-out loss is one learning measurement. Useful Metal edits require
generated base/adapter comparisons followed by patch application, compilation,
numerical correctness and representative latency measurements. Compilation or
likelihood alone does not prove faster or more reliable kernels.
