# Inspect frozen-prefix cache coverage without loading a model

Run from an installed package or its source checkout:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m exl3_mlx.cache_inspect /absolute/sft-project
PYTHONDONTWRITEBYTECODE=1 python -m exl3_mlx.cache_inspect /absolute/sft-project --cache /absolute/existing-cache --details
```

The project must contain `project.json` and the selected prepared `data/train.jsonl` (or `--split valid` / `--split test`). The inspector verifies the data checksum against the project's manifest. It uses the complete teacher-forcing input, `input_ids[:-1]`, with canonical little-endian **int64** token bytes, batch/sequence shape, and the requested modes. Defaults are `--mode train` and checkpoint blocks enabled; use `--mode eval` or `--no-checkpoint-blocks` to inspect those distinct keys. It never truncates a row.

The compact JSON summary reports coverage, missing/malformed row indices, unique matching payload bytes (duplicate rows do not inflate bytes), recorded frozen boundary, project boundary/arithmetic checks, stored binding identity and recorded platform. `--details` includes per-row status and captured input hashes. Only expected row entries are inspected; this is not a whole-directory garbage or quota audit.

Default inspection reads bounded project/data/binding/entry JSON and stats the state files; it does **not** read tensor payloads or safetensors headers. `metadata_consistent` means the expected key, binding hash, declared tensor geometry and file size agree. It is not a cache hit, trusted tensor result, or compatibility certificate. Missing entries and malformed entries are reported independently. Unsafe or changing project/binding inputs refuse the inspection. An active producer may cause a directory/metadata identity check to refuse; retry after it is idle. No lock is acquired and no cache is created, repaired or modified.

An explicitly supplied `--expected-binding /absolute/binding.json` compares the complete recorded binding object to that file. Equality is still not proof that a running worker has those model/source/platform identities. The offline inspector does not open paths listed in model payload metadata, import MLX, query a GPU, or load model/cache tensors. Its local OS/platform comparison does not verify the GPU device.

**Inspector and worker identities are different.** The worker's current `implementation_identity()` hashes every package `.py`, `.metal` and `.html` file. Adding this inspector to a package changes that package identity; do not imply the changed package can consume an old cache unchanged. Keep an existing frozen worker/runtime unchanged when inspecting its caches. External orchestration may use this offline tool, but actual cache access must still run the worker's original complete model/source/platform binding, payload checksum, header and tensor validation. This tool does not weaken or replace any worker check. `compatibility_qualified` and `worker_implementation_verified` remain false even with complete coverage and matching supplied binding.

Full state-file hashing is a separate explicit option:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m exl3_mlx.cache_inspect /absolute/sft-project --full-payload-checksums
```

This can read tens of GiB and take substantial time. It streams every expected unique payload with a bounded descriptor reader; it does not load tensors. A valid checksum still does not establish numerical correctness or worker compatibility. Limits: 64 MiB prepared data, 2 MiB per metadata document, 128 MiB aggregate captured metadata (failed read attempts consume allowance), 10,000 rows, 32,768 serialized tokens per row, 2 GiB per payload and 256 GiB aggregate optional payload hashing. Limits bound I/O/retained raw bytes, not total Python memory or runtime. Files are rechecked before reporting, but finite unlocked observations cannot prevent later changes.

This command prints to stdout only. Redirect output yourself if a saved report is wanted. It sends no signals, starts no processes, and does not write project, cache or model files.
