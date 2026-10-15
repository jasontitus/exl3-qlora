# Process identity during training recovery

`exl3_mlx.process_identity.validate_terminal_process_table` validates bounded, caller-supplied `ps` bytes against every recorded PID and birth time. It distinguishes a genuinely surviving worker from a later unrelated process that reused its PID. Unknown, equal, earlier, or ambiguous births and parent/group relationships reject recovery.

The helper is pure Python and never queries, signals, starts, or stops processes. Each invocation validates one finite snapshot. The caller must supply all recorded generations, including the controller, use the same host and time convention, and validate a second fresh snapshot immediately before publishing recovery evidence. It does not prove continuous absence or cover untracked daemonization.

This additive API has 24 CPU tests, including malformed identities, ambiguous relationships and defensive result copies. Three real snapshots were independently replayed against the reviewed transfer implementation. Existing sealed training, transfer and evaluation consumers remain unchanged; callers must adopt it through their own reviewed revisions.
