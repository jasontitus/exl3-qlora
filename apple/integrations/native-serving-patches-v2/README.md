Portable full-attention serving patch candidate

Apply only after the existing native-serving-patches-v1 lifecycle and host report corrections, to an isolated source tree whose targeted files pass check_sources.py --state before. Review mlx-stream-portable.patch, run git apply --check, apply it, then check_sources.py --state after. The checker verifies existing target hashes and absence of three new files; it never fetches, builds or applies code.

The patch adds a strict hardware capability gate, portable stock selected attention with32-row materialized tiles, tiled stock index scoring on every non-fused branch, <=48-assignment expert GEMV batches, and conservative shared memory accounting. Supported fused NAX paths remain selected on eligible hardware. Full logical grouped/BF16 output projection and adapter hooks remain. The last file is dependency-bound TraceOps source tests, not a standalone test executable.

Source, scalar and static linked artifact reviews pass for stated scopes. Actual model-free portable attention is separately validated, but this bundle does not claim full serving, compressed/index hardware parity, adapter equivalence, long-context fit, throughput or generated-kernel quality. The new full-serving regression remains independently gated. No stock mlx-serve compatibility or exact upstream revision is implied by local target fingerprints.

Existing mlx-stream and mlx-serve-host license texts are carried unchanged. This patch is an additive delta to the captured local engine, not a vendored unrelated host tree. Qualification helper inclusion is separately planned; no captured model inputs or weights are included.
