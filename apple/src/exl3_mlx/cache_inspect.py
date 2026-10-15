"""Read-only SFT frozen-prefix cache coverage; no MLX, tensor loads or writes.

Coverage is metadata consistency, never permission to reuse a cache. The loaded
model must still verify its complete binding, payload checksums and tensor state.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import stat
import struct
from pathlib import Path

from .evaluation_audit import _json
from .evaluation_receipts import FileIdentity, capture_receipts, hash_file
from .prefix_disk import canonical

FORMAT = "exl3-frozen-prefix-disk-v1"
HEX = re.compile(r"[0-9a-f]{64}\Z")
MAX_DATA = 64 << 20
MAX_METADATA = 2 << 20
MAX_TOTAL_METADATA = 128 << 20
MAX_ROWS = 10000
MAX_TOKENS = 32768
MAX_PAYLOAD = 2 << 30
MAX_TOTAL_PAYLOAD = 256 << 30
LEAVES = {"h", "pre", "kv0", "kv1", "kv2", "kv3"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_directory(value):
    path = Path(os.path.abspath(value))
    require(path.resolve(strict=True) == path, "Directory contains a symlink alias")
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode), "Expected existing directory")
    return path, (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns)


def key_for_row(row, *, training, checkpoint_blocks):
    require(
        type(training) is bool and type(checkpoint_blocks) is bool,
        "Boolean modes required",
    )
    require(isinstance(row, dict), "Prepared row must be an object")
    ids, start = row.get("input_ids"), row.get("loss_start")
    require(
        isinstance(ids, list) and 2 <= len(ids) <= MAX_TOKENS,
        "Invalid complete token length",
    )
    require(
        all(type(x) is int and 0 <= x < 2**63 for x in ids), "Invalid integer token IDs"
    )
    require(type(start) is int and 1 <= start < len(ids), "Invalid supervised boundary")
    # Teacher forcing consumes every token except the final target; key uses i64,
    # even though training's input tensor is i32. Never hash native-width IDs.
    tokens = ids[:-1]
    metadata = {
        "shape": [1, len(tokens)],
        "training": training,
        "checkpoint_blocks": checkpoint_blocks,
        "tokens_hex": struct.pack("<" + "q" * len(tokens), *tokens).hex(),
    }
    return hashlib.sha256(canonical(metadata)).hexdigest(), metadata


class Reader:
    def __init__(self):
        self.snapshots = []
        self.remaining = MAX_TOTAL_METADATA

    def read(self, path, limit=MAX_METADATA):
        limit = min(limit, self.remaining)
        require(limit > 0, "Aggregate metadata byte limit exceeded")
        try:
            snapshot = capture_receipts(
                {"input": path}, max_file_bytes=limit, max_total_bytes=limit
            )
        except (OSError, ValueError):
            self.remaining -= limit  # Bound repeated malformed read attempts too.
            raise
        self.remaining -= len(snapshot.files[0].data)
        self.snapshots.append(snapshot)
        return snapshot.files[0].data

    def recheck(self):
        for snapshot in self.snapshots:
            snapshot.recheck()

    def hashes(self):
        return {str(s.files[0].path): s.files[0].sha256 for s in self.snapshots}


def inspect_project(
    project,
    *,
    cache=None,
    split="train",
    training=True,
    checkpoint_blocks=True,
    expected_binding=None,
    full_payload_checksums=False,
):
    """Inspect expected row entries only. Does not discover or open model files."""
    require(split in {"train", "valid", "test"}, "Unknown prepared split")
    require(type(full_payload_checksums) is bool, "Checksum flag must be boolean")
    require(
        type(training) is bool and type(checkpoint_blocks) is bool,
        "Boolean modes required",
    )
    root, root_identity = canonical_directory(project)
    reader = Reader()
    inspector_sha = hashlib.sha256(reader.read(Path(__file__).resolve())).hexdigest()
    project_record = _json(reader.read(root / "project.json"))
    require(
        project_record.get("format") == "exl3-sft-project-v1", "Unsupported project"
    )
    config = project_record.get("config")
    require(isinstance(config, dict), "Missing project configuration")
    cache_value = cache if cache is not None else config.get("prefix_disk_cache")
    require(
        isinstance(cache_value, (str, Path)) and str(cache_value),
        "Explicit existing cache required",
    )
    directory, directory_identity = canonical_directory(cache_value)
    binding_record = _json(reader.read(directory / "binding.json"))
    require(
        binding_record.get("format") == FORMAT
        and isinstance(binding_record.get("binding"), dict),
        "Malformed cache binding",
    )
    binding = binding_record["binding"]
    require(
        all(
            isinstance(binding.get(k), dict)
            for k in ("implementation", "platform", "model_payloads", "state_layout")
        ),
        "Incomplete binding structure",
    )
    binding_hash = hashlib.sha256(canonical(binding)).hexdigest()
    boundary = binding.get("frozen_blocks")
    require(type(boundary) is int and boundary > 0, "Invalid frozen boundary")
    selected = config.get("adapter_layers")
    experts = config.get("expert_adapter_layers", [])
    require(
        isinstance(experts, list) and all(type(x) is int and x >= 0 for x in experts),
        "Malformed expert adapter layers",
    )
    require(
        selected is None
        or (
            isinstance(selected, list)
            and selected
            and all(type(x) is int and x >= 0 for x in selected)
        ),
        "Malformed adapter layers",
    )
    project_boundary = 0 if selected is None else min(selected + experts)
    checks = {
        "frozen_boundary": boundary == project_boundary,
        "activation_mode": binding.get("activation_mode")
        == config.get("activation_mode"),
        "chunk_size": binding.get("chunk_size") == config.get("chunk_size"),
    }
    base_metadata = project_record.get("capabilities", {}).get("base_metadata", {})
    model_metadata_checks = {
        name: isinstance(binding["model_payloads"].get(name), dict)
        and binding["model_payloads"][name].get("sha256") == digest
        for name, digest in base_metadata.items()
    }
    expected_match = None
    if expected_binding is not None:
        expected = _json(reader.read(Path(os.path.abspath(expected_binding))))
        require(
            expected.get("format") == FORMAT
            and isinstance(expected.get("binding"), dict),
            "Expected binding must use the complete binding.json schema",
        )
        expected_match = expected == binding_record
    local_platform = {
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
        "version": platform.version(),
    }
    platform_match = isinstance(binding.get("platform"), dict) and all(
        binding["platform"].get(k) == v for k, v in local_platform.items()
    )
    data_path = root / "data" / (split + ".jsonl")
    data = reader.read(data_path, MAX_DATA)
    data_hash = hashlib.sha256(data).hexdigest()
    expected_data = (
        project_record.get("data", {}).get("files", {}).get(split + ".jsonl")
    )
    require(
        isinstance(expected_data, str)
        and HEX.fullmatch(expected_data)
        and data_hash == expected_data,
        "Prepared data does not match project manifest",
    )
    lines = data.splitlines()
    require(
        len(lines) <= MAX_ROWS and all(line.strip() for line in lines),
        "Invalid/bounded row count",
    )
    rows = [_json(line) for line in lines]
    results, directories, payloads, unique_bytes = [], {}, {}, {}
    checksum_bytes = 0
    for index, row in enumerate(rows):
        key, metadata = key_for_row(
            row, training=training, checkpoint_blocks=checkpoint_blocks
        )
        entry = directory / key
        record = {"row": index, "key": key, "tokens": len(row["input_ids"]) - 1}
        if not os.path.lexists(entry):
            record["status"] = "missing"
            results.append(record)
            continue
        try:
            _, entry_identity = canonical_directory(entry)
            directories[entry] = entry_identity
            with os.scandir(entry) as entries:
                names = []
                for item in entries:
                    names.append(item.name)
                    require(len(names) <= 2, "Surplus cache entry files")
            require(
                set(names) == {"entry.json", "state.safetensors"}, "Incomplete entry"
            )
            receipt = _json(reader.read(entry / "entry.json"))
            require(isinstance(receipt, dict), "Entry metadata must be an object")
            payload = entry / "state.safetensors"
            require(payload.resolve(strict=True) == payload, "Payload alias")
            identity = FileIdentity.from_stat(payload.lstat())
            require(
                receipt.get("format") == FORMAT
                and canonical(receipt.get("key")) == canonical(metadata)
                and receipt.get("binding_sha256") == binding_hash,
                "Entry key/binding mismatch",
            )
            require(
                type(receipt.get("file_bytes")) is int
                and receipt["file_bytes"] == identity.size
                and 0 < identity.size <= MAX_PAYLOAD,
                "Payload byte count/limit mismatch",
            )
            require(
                isinstance(receipt.get("sha256"), str)
                and HEX.fullmatch(receipt["sha256"]),
                "Malformed payload checksum",
            )
            tensors = receipt.get("tensors")
            require(
                isinstance(tensors, dict) and set(tensors) == LEAVES,
                "Malformed tensor metadata",
            )
            widths = {"float16": 2, "bfloat16": 2, "float32": 4, "int32": 4, "uint8": 1}
            for name, value in tensors.items():
                require(isinstance(value, dict), "Malformed tensor specification")
                shape, dtype, size = (
                    value.get("shape"),
                    value.get("dtype"),
                    value.get("bytes"),
                )
                require(
                    isinstance(shape, list)
                    and len(shape) == (4 if name == "h" else 3)
                    and all(type(x) is int and 0 <= x <= MAX_PAYLOAD for x in shape)
                    and isinstance(dtype, str)
                    and dtype.removeprefix("mlx.core.") in widths
                    and type(size) is int
                    and 0 <= size <= MAX_PAYLOAD,
                    "Malformed tensor specification",
                )
                dtype = dtype.removeprefix("mlx.core.")
                require(
                    math.prod(shape) * widths[dtype] == size,
                    "Tensor byte geometry mismatch",
                )
                require(
                    shape[0] == 1
                    and (
                        shape[1] == metadata["shape"][1]
                        if name in {"h", "pre", "kv2", "kv3"}
                        else shape[1] <= metadata["shape"][1]
                    ),
                    "Tensor token geometry mismatch",
                )
                require(
                    dtype == ({"kv2": "int32", "kv3": "uint8"}[name])
                    if name in {"kv2", "kv3"}
                    else dtype in {"float16", "bfloat16", "float32"},
                    "Tensor dtype mismatch",
                )
            require(
                sum(value["bytes"] for value in tensors.values()) < identity.size,
                "Tensor bytes do not fit payload",
            )
            if full_payload_checksums and payload not in payloads:
                require(
                    checksum_bytes + identity.size <= MAX_TOTAL_PAYLOAD,
                    "Aggregate payload limit",
                )
                checksum_bytes += identity.size
                require(
                    hash_file(payload, max_bytes=MAX_PAYLOAD) == receipt["sha256"],
                    "Payload checksum mismatch",
                )
            if payload in payloads:
                require(
                    payloads[payload] == identity, "Repeated payload identity changed"
                )
            else:
                payloads[payload] = identity
            unique_bytes[key] = identity.size
            record.update(
                status="metadata_consistent",
                payload_bytes=identity.size,
                payload_checksum_verified=full_payload_checksums,
            )
        except (OSError, ValueError) as error:
            record.update(status="malformed", error=str(error))
        results.append(record)
    reader.recheck()
    require(canonical_directory(root)[1] == root_identity, "Project directory changed")
    require(
        canonical_directory(directory)[1] == directory_identity,
        "Cache directory changed",
    )
    for path, identity in directories.items():
        require(canonical_directory(path)[1] == identity, "Entry directory changed")
    for path, identity in payloads.items():
        require(
            path.resolve(strict=True) == path
            and FileIdentity.from_stat(path.lstat()) == identity,
            "Payload changed during inspection",
        )
    return {
        "schema": "exl3-cache-coverage-v1",
        "inspector_source_sha256": inspector_sha,
        "worker_implementation_verified": False,
        "inspector_is_worker": False,
        "project": str(root),
        "cache": str(directory),
        "split": split,
        "rows": len(rows),
        "unique_expected_keys": len({r["key"] for r in results}),
        "training": training,
        "checkpoint_blocks": checkpoint_blocks,
        "frozen_blocks": boundary,
        "project_frozen_blocks": project_boundary,
        "project_binding_checks": checks,
        "binding_sha256": binding_hash,
        "binding_summary": {
            "platform": binding["platform"],
            "activation_mode": binding.get("activation_mode"),
            "chunk_size": binding.get("chunk_size"),
            "model_payload_count": len(binding["model_payloads"]),
            "implementation_sha256": hashlib.sha256(
                canonical(binding["implementation"])
            ).hexdigest(),
        },
        "project_model_metadata_checks": model_metadata_checks,
        "expected_binding_equal": expected_match,
        "local_platform_fields_equal": platform_match,
        "device_verified": False,
        "metadata_consistent_rows": sum(
            r["status"] == "metadata_consistent" for r in results
        ),
        "missing_rows": [r["row"] for r in results if r["status"] == "missing"],
        "malformed_rows": [r["row"] for r in results if r["status"] == "malformed"],
        "unique_matching_payload_bytes": sum(unique_bytes.values()),
        "payload_bytes_hashed": checksum_bytes,
        "entries": results,
        "input_hashes": reader.hashes(),
        "compatibility_qualified": False,
        "scope": "Expected-row metadata coverage only; other cache entries uninspected. No model/source/device compatibility attestation, numerical validation or hit permission. Even matching supplied binding or payload hashes require loaded-model binding/header/tensor checks. Read-only finite observation; no locks or writes.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--no-checkpoint-blocks", action="store_true")
    parser.add_argument("--expected-binding", type=Path)
    parser.add_argument("--full-payload-checksums", action="store_true")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include per-row records and input hashes",
    )
    args = parser.parse_args(argv)
    result = inspect_project(
        args.project,
        cache=args.cache,
        split=args.split,
        training=args.mode == "train",
        checkpoint_blocks=not args.no_checkpoint_blocks,
        expected_binding=args.expected_binding,
        full_payload_checksums=args.full_payload_checksums,
    )
    if not args.details:
        result = {
            k: v for k, v in result.items() if k not in {"entries", "input_hashes"}
        }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
