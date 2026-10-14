# Compile captured local Metal closures

The existing `exl3-mlx-compile --before ... --after ... --output ...` interface is unchanged. For already captured and reviewed source dependencies, use:

```bash
exl3-mlx-compile closure --manifest closure.json \
  --manifest-sha256 <sha256-of-exact-manifest-bytes> --output new-compile-directory
```

Python callers can use `exl3_mlx.metal_compile_closure.compile_closure(manifest, sha256, output)`. This command performs no Git operations, downloads, upstream build scripts, model loads, or GPU execution. It compiles translation units sequentially with the installed Apple compiler. The source itself is not executed.

A small manifest (replace hashes and revision labels with your actual bindings):

```json
{
  "format": "exl3-metal-local-closure-v1",
  "standard": "metal4.0",
  "sdk_mpp": false,
  "units": [{
    "id": "parent",
    "revision": "0123456789012345678901234567890123456789",
    "primary": "kernel.metal",
    "files": {
      "kernel.metal": {"path": "captured/kernel.metal", "sha256": "<64 lowercase hex characters>"},
      "common.h": {"path": "captured/common.h", "sha256": "<64 lowercase hex characters>"}
    }
  }]
}
```

Manifest paths identify local regular files, relative to the manifest directory or absolute. Source keys describe the snapshot layout. Keys must be canonical relative paths; `.metal`, `.h`, `.hpp`, and `.inc` are supported. Symlink inputs, duplicate JSON keys, nonfinite JSON values, unknown fields, case-aliased source keys, and file/directory collisions are rejected. The revision is a supplied 40-hex label, not authenticated repository provenance. A hash-bound manifest is not proof of independent review.

Each of 1–32 units has exactly one `.metal` primary and up to 128 files. UTF-8 source without NULs is limited to 16 MiB per unit and 64 MiB across units (repeated files count each time). Every declared file must be reachable from the primary's quoted includes; dependencies in conditional branches are still required. Quoted includes resolve first relative to the including file, then to the unit root. `../` is allowed only when normalization from the including file stays inside the captured root. There is no external filesystem fallback. Missing, macro, absolute, escaping, alternate-preprocessor, `include_next`, `import`, and `embed` dependencies fail before compilation. Raw C++ strings and module/import/probing syntax are conservatively unsupported.

Angle includes are restricted to the explicit standard Metal header allowlist in `METAL_HEADERS`. Optional `sdk_mpp: true` allows only the installed `MetalPerformancePrimitives/MetalPerformancePrimitives.h` umbrella and requires `metal4.0`. The installed MPP header inventory is bounded to 64 files / 16 MiB, hashed, and rechecked. Other SDK includes are not accepted as user-declared closure dependencies. The compiler and installed standard library remain trusted toolchain inputs, not a hermetic reconstruction of Apple's SDK.

All source snapshots go into a fresh directory before tool invocation. Compilation uses fixed `-O3 -fno-fast-math`, an explicit SDK root, private module cache, private HOME/TMPDIR, and a minimal environment. Custom environment flags and developer-directory overrides are not inherited. Each compiler invocation has a 90-second timeout. Tool discovery/version, compiler hash, SDK path/version, optional MPP hashes, environment, exact argv, include edges, snapshot hashes, diagnostics, and AIR hashes are recorded in `receipt.json`. A successful AIR must be a regular nonempty file no larger than 128 MiB. Source/SDK/compiler continuity and earlier AIR outputs are rechecked before final success.

Input-validation errors exit 2 without launching the compiler. Compiler/toolchain failures exit 1 and retain snapshots and diagnostics; full success exits 0. Existing output directories are never overwritten. An unexpected interruption can leave a partial directory without a final receipt; preserve it and choose a new directory for retry. The compiler is not a sandbox for hostile source or a replacement for reviewing captured inputs.

A successful receipt establishes only compilation with these fixed flags. It does not establish device execution, numerical correctness, speed, host ABI, complete project integration, original benchmark equivalence, licensing permission, or training admission. Existing semantic defects remain defects even when all units compile.
