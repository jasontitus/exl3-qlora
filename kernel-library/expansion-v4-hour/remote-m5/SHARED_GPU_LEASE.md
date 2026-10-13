# Shared GPU ownership

Sushi's M5 runner uses `/private/tmp/sushi-gpu.lock.d`, a symlink to a unique `.holder.<token>` directory with an `owner` record. It serializes mutations with `/usr/bin/lockf -k` on `.mutation` and queues callers with numeric symlink tickets. Read-only captured protocol evidence is in `sushi-lease-protocol/`; the observed PID21341 was another workload and was not interrupted. Parent subsequently observed its natural completion and lease release.

For short immediate controls, the existing pinned Sushi `engine_validate.py` `gpu_session`/`Owner.command` interface already follows this protocol. It owns only children it launches; never register unrelated processes. A free-lock observation alone is not ownership.

`shared_gpu_lease.py` is a separate fail-fast wrapper for an explicitly reviewed local command. It does not change the reviewed model runtime or launchers. CPU fixture tests do not qualify actual M5 execution; the concrete request and implementation require independent review first.

The wrapper refuses occupied, dangling, ownerless or unknown locks, unresolved recovery receipts, and existing waiters. It registers only its own FIFO ticket, uses the same mutation lock and publishes the same atomic symlink claim. It never breaks a lock, prunes another ticket, or signals a process. Only its own exact unchanged claim may be released. If the command fails, source/input identity changes, completion criteria fail, the wrapper is interrupted, or descendant cleanup is uncertain, its own lease and recovery record remain. There is no automatic recovery/break command: inspect the captured child/owner evidence before explicit coordinator recovery.

The wrapper is intended to cover one immediate run or a reviewed serial model campaign. Do not hold it around downloading, editing, compilation, analysis or a queue of unrelated experiments. Its scope is cooperative ownership, not proof that every process on the machine honors the lease.

A request has this shape (paths and hashes must be actual execution-host values):

```json
{
  "schema": "shared-gpu-lease-command-v1",
  "hostname": "<exact M5 hostname>",
  "argv": ["/absolute/reviewed-executable", "/absolute/job.json", "/absolute/new-result.json"],
  "cwd": "/absolute/existing-directory",
  "inputs_sha256": {
    "/absolute/reviewed-executable": "<sha256>",
    "/absolute/job.json": "<sha256>",
    "/absolute/other-bound-source-or-library": "<sha256>"
  },
  "completion_receipt": "/absolute/new-result.json",
  "completion_expect": {"numerical": "completed_fixed_cases", "gpu_executed": true},
  "expected_exit_codes": [0],
  "no_detached_background_children": true
}
```

The executable and every existing literal file argument must be pinned. Include the reviewed job's source/library/manifest inputs as well. A symlink executable requires the resolved target also be pinned. `observed_harness_sha256`, if present, must equal the executable hash. Input hashes are checked before acquisition, immediately before the child starts, and after it completes. Only the exact declared completion output is exempted from input hashing; it must be absent initially and match the declared completion fields afterward. Negative-control requests can explicitly expect exit1 and candidate failure, while still requiring completed reference/guard evidence; the inner reviewed harness/launcher remains responsible for its full result contract. For model launches, bind the reviewed launcher, request/plan and verification receipts rather than adding repeated full model-payload reads; inner launchers retain their full model/runtime guards.

```bash
python shared_gpu_lease.py status
python shared_gpu_lease.py run --request REQUEST.json --sha256 REQUEST_SHA256 --receipt /absolute/new-wrapper-receipt.json
```

No shell is involved. The wrapper captures stdout/stderr, child PID, observed descendant start identities, completion hash and ownership history. Process sampling and successful reviewed child completion provide conservative cleanup evidence for cooperative non-daemonizing commands; this is not hostile-process attestation. `_mutate` is an internal subcommand invoked only under the shared `lockf` mutation lock, not a manual acquisition/release interface.
