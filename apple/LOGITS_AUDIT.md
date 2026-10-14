# Compare selected logits without running a model

`exl3-mlx-audit logits` checks complete observed vocabulary rows, their exact token prefixes, producer identity assertions, and local file hashes. It uses only Python's standard library. Existing teacher-forced `exl3-mlx-audit --prepared …` usage is unchanged.

```sh
exl3-mlx-audit logits --manifest observations/manifest.json --output logits-audit.json
```

The default is numerical equality (`atol=0`, `rtol=0`); bitwise equality is reported separately, including signed-zero differences. Explicit `--atol` and `--rtol` use `abs(a-b) <= atol + rtol * max(abs(a), abs(b))`. The output records the tolerances. A failed required comparison still publishes the complete audit and exits **1**. Invalid evidence exits **2** without publishing a result. Success exits **0**. Tolerance selection requires a separate experiment decision; this tool does not adjust tolerances to obtain a pass.

Three comparison purposes are supported:

- `zero_b_control`: baseline versus a control adapter whose A tensors are byte-identical to a supplied reference, B values are all zero, and config is unchanged except its payload hash. Equality is required within the declared tolerances.
- `cache_parity`: the same adapter identity over the same exact prefix. Equality is required. The evidence supplier remains responsible for proving that the observations came from cached and fresh execution respectively.
- `adapter_effect`: descriptive numerical differences. A nonzero delta does not fail the audit and does not establish improvement.

The API is `exl3_mlx.logits_audit.audit_logit_files(manifest=…, output=…, atol=0.0, rtol=0.0)`. `compare_logit_values(left_bytes, right_bytes, …)` also supports direct comparison of two complete F32 little-endian rows. The APIs do not launch an engine, import MLX, or read base model tensors.

## Portable manifest contract

Use format `observed-logits-audit-v1`, a `conditions` list, and a `comparisons` list. All file descriptors have exactly `{"path": "relative/file", "sha256": "64 lowercase hex digits"}`. Paths are relative to the manifest directory; absolute paths, traversal and symlink components are rejected. Copy evidence into this directory without rewriting its bytes. Original receipt paths can remain inside the copied receipt as historical data.

Each condition contains:

| Field | Required value |
|---|---|
| `id` | Unique nonempty identifier containing letters, digits, dots, dashes or underscores |
| `role` | `baseline`, `zero_b`, or `adapter` |
| `vocab_size` | Number of values in every complete observed row |
| `identity` | The fields described below |
| `observations` | One to eight rows, ordered by strictly increasing prefix length |
| `adapter` | For adapter/control conditions: `config` and `weights` file descriptors |
| `zero_b_reference` | For controls only: original reference `config` and `weights` descriptors |
| `evidence` | Optional list of original request/construction/result and other receipt descriptors |

`identity` requires `base_payload_manifest_sha256`, `runtime_manifest_sha256`, `tokenizer_sha256`, `arithmetic_policy_sha256`, `request_sha256`, `model_instance` (positive integer), and `adapter_fingerprint`. The first four hashes must agree for each comparison. Request and instance identities must agree with the actual observation metadata. Different processes may have different instance IDs. These hashes are **producer assertions checked for equality**; the auditor does not independently authenticate the process, launch, model manifest or base payload. Optional evidence is hash-bound and retained in the report, not interpreted as proof of successful execution.

Baseline has fingerprint 64 zeros and no adapter files. The current adapter scheme is the private MLX-stream LoRA identity: SHA256 of `mlx-stream-adapter-transfer-stock-v1`, a NUL byte, the 32 raw payload-hash bytes, then the exact config JSON bytes. The parser accepts complete finite F32 `*.lora_a`/`*.lora_b` tensor pairs and checks contiguous spans, paired rank dimensions and `adapter_sha256`. It does not establish compatibility with another adapter format or engine. Zero-B verification checks the actual supplied tensor bytes, not a `zero_b` label alone.

An observation contains `metadata` and `values` descriptors plus the complete `prefix_ids` list. Metadata uses the native `selected_raw_logits_v1` format, including the request/instance/adapter bindings, position, little-endian prefix-ID hash, vocabulary size, original raw filename, F32 byte count/hash, finite flag, source dtype and argmax. The auditor verifies these against every raw value. `values` files may be relocated/renamed; the captured metadata's original raw filename remains `logits-POSITION.f32`.

Each comparison has exactly:

```json
{"left": "base", "right": "zero", "purpose": "zero_b_control", "positions": [33, 34]}
```

Both conditions must contain every requested position, with identical exact prefix IDs and source dtype. Within one condition/request, every longer prefix must extend each earlier observed prefix exactly; nonconsecutive selected positions are allowed. Divergent generated prefixes fail the comparison instead of silently reducing it to an intersection. Extra supplied observations are validated and listed as `uncompared_observations`; they do not become unrequested comparisons. An early EOS may leave later observations unavailable: do not fabricate rows to satisfy the manifest.

## Interpretation

The report separates `observation_evidence_consistent` from `required_comparisons_passed`, and includes whole-row bitwise equality, maximum/mean absolute error, maximum symmetric relative error, tolerance violations and argmax agreement. Adapter changes are descriptive. `device_qualified`, `base_payload_independently_verified`, `quality_certified` and `trainer_forward_parity` remain false.

Actual serving qualification still needs independently bound model/runtime/launch receipts, the intended cache execution paths, and hardware checks. Observation adds synchronization and readback; these rows do not establish serving throughput. Generated optimization quality additionally requires patch application, compilation, independent numerical tests and workload-specific measurements.
