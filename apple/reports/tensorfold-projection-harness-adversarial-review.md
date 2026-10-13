# TensorFold projection harness review — 2026-10-12

Read-only CPU review of `scripts/verify_tensorfold_exlr8.swift` against the actual
before/after `exlr8_projection.metal` translation units. No GPU execution or
numerical result is reported here. The parent owns repairs and eventual execution.

## Findings sent before execution

1. **Qualification identity gate missing (high).** Original lines11–18,42–46
   accept arbitrary libraries while hardcoding commit51426b8e. Passing the same
   library twice can satisfy all comparisons. Require a pinned compile/link
   manifest binding the exact ordered before/after source and metallib digests,
   expected source SHA constants/commit, compiler settings, and link results.
   Load the captured bytes whose digest was checked; hashing `Data` then calling
   `makeLibrary(URL:)` performs two different reads. Record harness source or
   executable SHA for reproducible PRNG/case generation. Parent agreed to repair.

2. **Canary predicate too weak (medium).** Original tail guard tested only
   `isNaN`, so an overwrite by another NaN still looked unchanged. Compare exact
   sentinel bit patterns. The current observed parent edit already uses
   `Float.nan.bitPattern` equality. This checks only the allocated32-float tail;
   it is not a general memory-safety or out-of-bounds-read proof.

3. **Failure receipt can preserve passed=true (low).** The final save runs after
   setting `passed=true`; a thrown save followed by successful catch-save could
   publish contradictory success+error. Catch should reset `passed=false` before
   writing the failure receipt. Parent notified.

## Host layout and dispatch checked

- `TfExlr8Job` is twelve contiguous `uint` fields (48bytes), matching the Swift
  twelve-UInt32 buffer. `TfExlr8Rows` is four uints (16bytes). UInt64 bases, UInt16
  half bit patterns and Int8 signs match MSL `ulong`, `half`, and `char` arrays.
- `blocks *64 *8 *bits` UInt32 words allocate `blocks *64 *32 *bits` packed bytes.
  Base offsets advance by exactly64tiles times32Kbytes. The after-kernel prefetch
  uses the next valid input block or the current run at the final block; the
  selected1/2/3-block cases keep every tile-word read within these allocations.
- Input length `rows *blocks *128` and stride `blocks *128` match zero x_col.
  Output stride128 and zero y_col bound writes to `rows *128` floats.
- Input-scale indices are0..blocks−1; output-channel scales are blocks..blocks+127;
  block scale is blocks+128, the final valid entry of the blocks+129 half buffer.
  Sign indices0..127 are valid. members is unused because count=0.
- Exactly256threads/8SIMD groups of32 are required by source; pipeline limits
  and threadExecutionWidth are checked before dispatch. Capacity16 static xs+zs
  uses16384bytes of threadgroup float storage. Actual dispatch is one job/group.
- Exhaustive codebook dispatch covers256×256=65536states. Case count is162:
  nine(full/tail) row choices across capacities ×3rates ×3block counts ×2modes.
  NaN-initialized output plus finiteness checks rejects uninitialized main output.
  Command completion/error is checked and reports are saved after every case.

## What a passing run would establish

Before/after bitwise regression parity on this GPU for all65536 codebook states
and the162 declared deterministic finite-input projection cases. Raw and transformed
outputs, full/tail capacities, nontrivial input/channel/block scales and alternating
signs are exercised. It is not an independent absolute mathematical oracle: both
versions can share a pre-existing error.

Untested: routed member mapping, BF16 stores, zero/nonzero offsets and padded
strides, multi-job batches, optional-scale/sign sentinel paths, zero blocks,
rotate/combine integration, other translation units/CUDA/prompt path, extreme or
exceptional values, performance, other Apple GPU families, full model quality,
and generated candidate correctness. A passing narrow receipt must not be promoted
to execution verification of the complete multi-file upstream commit.

After the identity/failure repairs, this is a useful bounded before/after
qualification gate. A final review of repaired source and actual GPU receipt is
still needed before claiming the planned result.

## Repaired-source and build-v3 review completed

The current Swift code resolves all three findings: mandatory ordered compile/link
bindings with pinned before/after source hashes, actual runner executable digest,
separate library digests and same-library rejection; the exact captured bytes are
loaded via `DispatchData`; canary comparison uses exact bits with explicit tail
scope; catch resets passed=false.

The builder initially re-read AIR after verification. That race is now closed:
verified AIR bytes are captured once, retained, copied and linked. The builder also
requires successful compile exit/passed/error fields and the fixed
`-std=metal3.1 -O3 -fno-fast-math` compile command layout. The source hashes are
pinned to the reviewed before/after translation units.

Independent CPU inspection of the completed build-v3 passes17 source/artifact
bindings, including compiler-receipt raw and canonical digests, current builder and
captured Swift hashes, executable bytes, ordered source/AIR/metallib bytes,
compile/link exits, and distinct libraries. Manifest SHA-256:
`39afa590299ef4a5f7f80d93d0aa7615cc02715d50acb0bb57f0ec656e867852`.
Receipt: `reports/tensorfold-projection-build-v3-independent-check.json`.

No remaining blocker found for this controlled local regression-parity run.
Checksums are local artifact bindings, not external authentication. GPU parity
and all numerical/performance outcomes remain untested by this review; the scope
and omitted cases listed above still apply.
# Completed GPU receipt: 2026-10-12 local time

Root executed the reviewed build-v3 runner on Apple M1 Ultra. I independently checked the resulting receipt on CPU, without rerunning the GPU workload. All 17 checks passed: manifest, executable, Swift source, builder, compile receipt, ordered and distinct before/after libraries, AIR/source links, argv, declared case set, numerical outcomes, and scope. Evidence is `reports/tensorfold-projection-gpu-independent-check.json`; the raw GPU receipt SHA256 is `432eb859e72dbfbb0909c7827b59a482e00f066cf393f7ca41e6f68902e8c60d`.

The receipt contains every one of the 162 declared projection cases exactly once, totaling 131,328 output values. Before/after outputs are finite and bitwise equal in all cases, and the exact 32-float trailing canaries are unchanged. All 65,536 codebook-state outputs are also finite and bitwise equal. This closes the earlier pending GPU parity gate for this specific qualification scope.

The result establishes before/after regression parity on these finite projection workloads and exhaustive codebook states on this GPU. It is not an independent absolute kernel oracle, full-commit verification, a performance measurement, verification of routed members or BF16 outputs, or evidence that a model-generated kernel is correct. Earlier pre-run statements below describe their historical review stage.

