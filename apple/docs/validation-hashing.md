# Bounded digest-only validation

`exl3_mlx.evaluation_receipts.hash_file(path, max_bytes=...)` hashes a canonical absolute regular file using reads of at most 1 MiB. It checks its descriptor and pathname identities before and after reading, rejects aliases and size changes, and closes the descriptor on failure. Every call reads every byte; there is no stat-only shortcut or digest cache. The default limit is 512 MiB; callers should choose the smallest appropriate finite limit and enforce their own path allowlist before invoking it.

Use `capture_receipts` when validation needs immutable report/log bytes or JSON. Use `hash_file` when only a digest is needed, such as validating an already allowlisted runtime artifact. Existing snapshot APIs and callers are unchanged. Hash identity does not validate content semantics, prove provenance, or prevent edits after return; recheck at the required publication/execution boundaries.

This maintained API is available for future digest-only validators. It does not automatically change historical lease scripts, private native frontends, active runtimes, or their execution permissions. Tests use synthetic local files and do not load models, start children, acquire leases, or access a GPU.
