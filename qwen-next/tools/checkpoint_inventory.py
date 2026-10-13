"""Bounded, stdlib-only local safetensors inventory; never reads tensor payloads.

This validates a storage container, not EXL3 decoding or model compatibility.
Only flat, regular shards are supported. Duplicate names across physical shards
are rejected even if a legacy loader might choose an index-selected owner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import struct
import sys
from pathlib import Path

MAX_HEADER_BYTES = 16 << 20
MAX_METADATA_BYTES = 32 << 20
MAX_FILES = 4096
MAX_TENSORS_PER_FILE = 200_000
MAX_RANK = 32
DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "U16": 16,
    "I16": 16,
    "U32": 32,
    "I32": 32,
    "U64": 64,
    "I64": 64,
    "F16": 16,
    "BF16": 16,
    "F32": 32,
    "F64": 64,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0": 8,
}
CATEGORIES = ("expert", "trunk", "PLE", "head", "vision", "MTP", "unknown")


class InventoryError(ValueError):
    """Malformed, ambiguous, unsupported or changing local inputs."""


def require(condition, message):
    if not condition:
        raise InventoryError(message)


def positive_limit(value):
    require(type(value) is int and value > 0, "Read limits must be positive integers")
    return value


def strict_json(raw, label):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(value):
        raise InventoryError(f"{label}: nonfinite JSON number {value}")

    try:
        result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise InventoryError(f"{label}: invalid JSON: {error}") from error
    require(isinstance(result, dict), f"{label}: expected a JSON object")
    return result


def identity(value):
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size_bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def regular_identity(path):
    try:
        value = path.lstat()
    except OSError as error:
        raise InventoryError(f"Cannot stat {path}: {error}") from error
    require(stat.S_ISREG(value.st_mode), f"Expected regular non-symlink file: {path}")
    return identity(value)


def read_prefix(path, count, offset=0):
    """Read at most count bytes and bind the opened regular file's identity."""
    before = regular_identity(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        require(
            identity(os.fstat(stream.fileno())) == before,
            f"File changed opening {path}",
        )
        stream.seek(offset)
        raw = stream.read(count)
        require(
            identity(os.fstat(stream.fileno())) == before,
            f"File changed reading {path}",
        )
    require(regular_identity(path) == before, f"File replaced reading {path}")
    require(len(raw) == count, f"Truncated read: {path}")
    return raw, before


def read_metadata(path, limit):
    before = regular_identity(path)
    require(before["size_bytes"] <= limit, f"Metadata exceeds byte limit: {path}")
    raw, opened = read_prefix(path, before["size_bytes"])
    require(opened == before, f"Metadata changed before reading: {path}")
    return strict_json(raw, path.name), {
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "stat": before,
    }


def classify(name):
    """Ordered name heuristics, deliberately not architecture validation."""
    pieces = re.split(r"[./]", name.lower())
    if any(x in pieces for x in ("vision", "visual", "vision_tower", "vision_model")):
        return "vision"
    if any(
        x in pieces for x in ("mtp", "multi_token_predictor", "multi_token_prediction")
    ):
        return "MTP"
    if any(
        x in pieces for x in ("ple", "ple_embedding", "ngram_embedding", "ngram_table")
    ):
        return "PLE"
    if any(
        x in pieces
        for x in ("experts", "expert", "switch_mlp", "shared_expert", "shared_experts")
    ):
        return "expert"
    if any(x in pieces for x in ("lm_head", "output_head")):
        return "head"
    if pieces[0] in ("model", "language_model", "transformer"):
        return "trunk"
    return "unknown"


def inspect_shard(path, *, max_header_bytes=MAX_HEADER_BYTES):
    positive_limit(max_header_bytes)
    prefix, before = read_prefix(path, 8)
    length = struct.unpack("<Q", prefix)[0]
    require(
        0 < length <= max_header_bytes, f"{path.name}: invalid/bounded header length"
    )
    require(length <= before["size_bytes"] - 8, f"{path.name}: truncated header")
    raw, opened = read_prefix(path, length, 8)
    require(opened == before, f"{path.name}: changed between header reads")
    require(
        raw.startswith(b"{"), f"{path.name}: safetensors header must begin with '{{'"
    )
    header = strict_json(raw, path.name)
    metadata = header.pop("__metadata__", {})
    require(
        isinstance(metadata, dict)
        and all(isinstance(v, str) for v in metadata.values()),
        f"{path.name}: metadata values must be strings",
    )
    require(
        len(header) <= MAX_TENSORS_PER_FILE, f"{path.name}: too many tensor entries"
    )
    payload_bytes = before["size_bytes"] - 8 - length
    tensors, ranges = [], []
    for name, spec in header.items():
        require(
            bool(name) and isinstance(spec, dict), f"{path.name}: invalid tensor entry"
        )
        require(
            set(spec) == {"dtype", "shape", "data_offsets"},
            f"{name}: invalid tensor fields",
        )
        dtype, shape, offsets = spec["dtype"], spec["shape"], spec["data_offsets"]
        require(
            isinstance(dtype, str) and dtype in DTYPE_BITS,
            f"{name}: unsupported dtype {dtype!r}",
        )
        require(
            isinstance(shape, list)
            and len(shape) <= MAX_RANK
            and all(type(d) is int and 0 <= d <= 2**63 - 1 for d in shape),
            f"{name}: invalid shape",
        )
        require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(type(x) is int for x in offsets),
            f"{name}: invalid offsets",
        )
        lo, hi = offsets
        require(
            0 <= lo <= hi <= payload_bytes, f"{name}: offsets outside payload bounds"
        )
        elements = 0 if 0 in shape else 1
        for dim in shape:
            elements *= dim
            require(
                elements * DTYPE_BITS[dtype] <= payload_bytes * 8,
                f"{name}: shape exceeds payload size",
            )
        size = elements * DTYPE_BITS[dtype] // 8
        require(hi - lo == size, f"{name}: dtype/shape size differs from offsets")
        if size:
            ranges.append((lo, hi, name))
        tensors.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "data_offsets": offsets,
                "physical_bytes": size,
                "category_heuristic": classify(name),
            }
        )
    cursor = 0
    for lo, hi, name in sorted(ranges):
        require(lo >= cursor, f"{path.name}: overlapping tensor ranges at {name}")
        require(lo == cursor, f"{path.name}: unindexed gap in tensor payload")
        cursor = hi
    require(cursor == payload_bytes, f"{path.name}: trailing unindexed payload bytes")
    require(
        regular_identity(path) == before,
        f"{path.name}: changed while validating header",
    )
    return {
        "path": path.name,
        "stat": before,
        "file_bytes": before["size_bytes"],
        "header_bytes": length,
        "header_sha256": hashlib.sha256(raw).hexdigest(),
        "length_and_header_sha256": hashlib.sha256(prefix + raw).hexdigest(),
        "physical_tensor_bytes": payload_bytes,
        "tensor_count": len(tensors),
        "metadata": metadata,
        "tensors": tensors,
    }


def basename(value, label):
    require(
        isinstance(value, str)
        and bool(value)
        and value not in (".", "..")
        and Path(value).name == value
        and "/" not in value
        and "\\" not in value
        and "\0" not in value,
        f"{label}: only local flat basenames are supported",
    )
    return value


def totals(tensors):
    result = {name: {"tensor_count": 0, "physical_bytes": 0} for name in CATEGORIES}
    for tensor in tensors:
        cell = result[tensor["category_heuristic"]]
        cell["tensor_count"] += 1
        cell["physical_bytes"] += tensor["physical_bytes"]
    return result


def inventory(
    directory,
    *,
    index=None,
    max_header_bytes=MAX_HEADER_BYTES,
    max_metadata_bytes=MAX_METADATA_BYTES,
):
    positive_limit(max_header_bytes)
    positive_limit(max_metadata_bytes)
    root = Path(directory).absolute()
    require(
        root.is_dir() and not root.is_symlink(),
        "Checkpoint must be a local non-symlink directory",
    )
    names = sorted(p.name for p in root.iterdir() if p.name.endswith(".safetensors"))
    require(
        0 < len(names) <= MAX_FILES,
        "Expected 1..4096 local safetensors shards; metadata alone is insufficient",
    )
    config, config_binding = read_metadata(root / "config.json", max_metadata_bytes)
    if index is None:
        found = sorted(
            p.name for p in root.iterdir() if p.name.endswith(".safetensors.index.json")
        )
        require(
            len(found) <= 1, "Multiple safetensors indices; choose --index explicitly"
        )
        index = found[0] if found else None
    index_data = index_binding = None
    if index is not None:
        index_data, index_binding = read_metadata(
            root / basename(index, "index"), max_metadata_bytes
        )
        mapping = index_data.get("weight_map")
        require(
            isinstance(mapping, dict) and bool(mapping),
            "Index requires a nonempty weight_map",
        )
        for name, owner in mapping.items():
            require(bool(name), "Index has an empty tensor name")
            basename(owner, "index owner")
            require(owner in names, f"Index owner shard missing: {owner}")
    shards = [
        inspect_shard(root / name, max_header_bytes=max_header_bytes) for name in names
    ]
    owners, all_tensors = {}, []
    for shard in shards:
        shard["categories"] = totals(shard["tensors"])
        for tensor in shard["tensors"]:
            name = tensor["name"]
            require(
                name not in owners,
                f"Duplicate physical tensor name across shards: {name}",
            )
            owners[name] = shard["path"]
            all_tensors.append(tensor)
    tensor_bytes = sum(x["physical_tensor_bytes"] for x in shards)
    if index_data is not None:
        require(
            index_data["weight_map"] == owners,
            "Index ownership mismatch or incomplete tensor mapping",
        )
        metadata = index_data.get("metadata", {})
        require(isinstance(metadata, dict), "Index metadata must be an object")
        if "total_size" in metadata:
            require(
                type(metadata["total_size"]) is int
                and metadata["total_size"] == tensor_bytes,
                "Index metadata total_size differs from physical tensor bytes",
            )
    external = []
    if "ngram_table" in config:
        table = config["ngram_table"]
        require(
            isinstance(table, dict) and "file" in table,
            "Invalid config ngram_table declaration",
        )
        name = basename(table["file"], "ngram_table file")
        require(
            name not in names, "External ngram table cannot alias a safetensors shard"
        )
        path = root / name
        exists = path.exists() or path.is_symlink()
        external.append(
            {
                "config_key": "ngram_table",
                "path": name,
                "declaration": table,
                "category_heuristic": "PLE",
                "status": "declared_not_inventoried",
                "present": exists,
                "stat": regular_identity(path) if exists else None,
                "payload_read": False,
                "payload_sha256": None,
                "included_in_tensor_totals": False,
            }
        )
    bindings = [config_binding] + ([index_binding] if index_binding else [])
    for binding in [*bindings, *shards, *[x for x in external if x["present"]]]:
        require(
            regular_identity(root / binding["path"]) == binding["stat"],
            f"Input changed during inventory: {binding['path']}",
        )
    require(
        sorted(p.name for p in root.iterdir() if p.name.endswith(".safetensors"))
        == names,
        "Shard set changed during inventory",
    )
    return {
        "format": "local-safetensors-header-inventory-v1",
        "checkpoint_directory": str(root),
        "header_validation_passed": True,
        "supported_training": False,
        "whole_checkpoint_complete": False,
        "weight_payloads_read": False,
        "payload_integrity_verified": False,
        "config": config_binding,
        "index": index_binding,
        "index_ownership_verified": index_data is not None,
        "limits": {
            "header_bytes_per_file": max_header_bytes,
            "metadata_bytes_per_file": max_metadata_bytes,
            "shards": MAX_FILES,
            "tensors_per_shard": MAX_TENSORS_PER_FILE,
            "rank": MAX_RANK,
        },
        "shard_count": len(shards),
        "tensor_count": len(all_tensors),
        "physical_tensor_bytes": tensor_bytes,
        "physical_safetensors_file_bytes": sum(x["file_bytes"] for x in shards),
        "categories": totals(all_tensors),
        "files": shards,
        "external_payloads": external,
        "category_policy": "Ordered name heuristics: vision, MTP, PLE, expert, head, trunk, unknown. Names are not semantic proof; MTP/vision expert tensors remain in their component category.",
        "limitations": [
            "Stored tensor bytes include quantization codes/scales, not decoded parameters or runtime memory estimates.",
            "Header and config/index hashes do not authenticate weight payloads or detect same-stat payload corruption.",
            "Declared ngram_table payloads are listed but not inventoried. Other external payload declarations are not exhaustively recognized; whole-checkpoint completeness is not asserted.",
            "No EXL3 decoder, tensor geometry, shard-stamp compatibility, numerical correctness or training support is validated.",
            "Only flat regular files and listed byte-aligned dtypes are supported; duplicate tensor names across shards fail closed.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--index", help="Index basename; auto-detects one *.safetensors.index.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON report; existing files are never overwritten",
    )
    parser.add_argument("--max-header-bytes", type=int, default=MAX_HEADER_BYTES)
    parser.add_argument("--max-metadata-bytes", type=int, default=MAX_METADATA_BYTES)
    args = parser.parse_args(argv)
    try:
        result = inventory(
            args.checkpoint,
            index=args.index,
            max_header_bytes=args.max_header_bytes,
            max_metadata_bytes=args.max_metadata_bytes,
        )
        if args.output:
            with args.output.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
    except (InventoryError, OSError) as error:
        print(f"Inventory refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
