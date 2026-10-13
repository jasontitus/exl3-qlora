"""Bounded CPU affine row payload reader; no decoding, MLX or model support.

A binding hashes only explicitly selected rows, plus index/config/header metadata.
It never scans a complete weight shard. The separately pinned canonical identity
is local consistency evidence, not authenticated provenance or whole-file integrity.
Weight, scales and biases must have one exact physical/index owner. Other shards
are not inspected for unindexed duplicate copies. Returned payloads are raw bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

MAX_ROWS = 4096
MAX_READ_BYTES = 8 << 20  # Selected payload bytes, including binding; not metadata.
MAX_FILE_BYTES = 64 << 30  # Size guard only, never a full-shard read.
MAX_IDENTITY_BYTES = 16 << 20
KINDS = ("weight", "scales", "biases")
FORMAT = "affine-selected-row-payload-identity-v1"


def _helper(name):
    spec = importlib.util.spec_from_file_location(
        "affine_reader_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inventory = _helper("checkpoint_inventory")
planner = _helper("affine_row_views")


def require(value, message):
    if not value:
        raise ValueError(message)


def _limit(value, maximum, name):
    require(
        type(value) is int and 0 < value <= maximum, "Invalid bounded limit: " + name
    )
    return value


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def identity_sha256(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _digest(value):
    require(
        isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value), "Invalid SHA256"
    )
    return value


def implementation_identity():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            Path(__file__).resolve(),
            Path(inventory.__file__).resolve(),
            Path(planner.__file__).resolve(),
        )
    }


_LOADED_IMPLEMENTATION = implementation_identity()


def _root(directory):
    p = Path(directory).absolute()
    require(
        p.is_dir() and not p.is_symlink(), "Expected non-symlink checkpoint directory"
    )
    return p.resolve(strict=True)


def _metadata(
    root,
    prefix,
    rows,
    *,
    index_name,
    shard_name=None,
    max_rows=MAX_ROWS,
    max_read_bytes=MAX_READ_BYTES,
    max_header_bytes=inventory.MAX_HEADER_BYTES,
    max_file_bytes=MAX_FILE_BYTES,
):
    _limit(max_rows, MAX_ROWS, "max_rows")
    _limit(max_read_bytes, MAX_READ_BYTES, "max_read_bytes")
    _limit(max_header_bytes, inventory.MAX_HEADER_BYTES, "max_header_bytes")
    _limit(max_file_bytes, MAX_FILE_BYTES, "max_file_bytes")
    require(isinstance(prefix, str) and prefix, "Expected nonempty affine prefix")
    inventory.basename(index_name, "index")
    config, cb = inventory.read_metadata(
        root / "config.json", inventory.MAX_METADATA_BYTES
    )
    index, ib = inventory.read_metadata(root / index_name, inventory.MAX_METADATA_BYTES)
    weight_map = index.get("weight_map")
    require(
        isinstance(weight_map, dict) and bool(weight_map), "Missing index weight_map"
    )
    for name, owner in weight_map.items():
        require(isinstance(name, str) and name, "Invalid indexed tensor name")
        inventory.basename(owner, "index owner")
    names = [prefix + "." + kind for kind in KINDS]
    require(
        all(name in weight_map for name in names), "Affine tensor missing from index"
    )
    owners = {weight_map[name] for name in names}
    require(len(owners) == 1, "Affine companion tensors must share one shard owner")
    actual_shard = next(iter(owners))
    require(actual_shard.endswith(".safetensors"), "Expected safetensors owner")
    if shard_name is not None:
        inventory.basename(shard_name, "shard")
        require(shard_name == actual_shard, "Requested shard differs from index owner")
    before = inventory.regular_identity(root / actual_shard)
    require(before["size_bytes"] <= max_file_bytes, "Shard exceeds file-size bound")
    report = inventory.inspect_shard(
        root / actual_shard, max_header_bytes=max_header_bytes
    )
    require(report["stat"] == before, "Shard changed while inspecting metadata")
    header = {
        t["name"]: {k: t[k] for k in ("dtype", "shape", "data_offsets")}
        for t in report["tensors"]
    }
    require(
        {name for name, owner in weight_map.items() if owner == actual_shard}
        == set(header),
        "Selected shard physical/index ownership mismatch",
    )
    plan = planner.plan_rows(
        header,
        config,
        prefix,
        rows,
        data_start=8 + report["header_bytes"],
        file_bytes=report["file_bytes"],
        max_rows=max_rows,
        max_read_bytes=max_read_bytes,
    )
    metadata = {
        "config": cb,
        "index": ib,
        "shard": {
            k: report[k]
            for k in (
                "path",
                "stat",
                "file_bytes",
                "header_bytes",
                "header_sha256",
                "length_and_header_sha256",
            )
        },
    }
    return metadata, plan, header, config


def _unchanged(root, metadata, implementation):
    for label in ("config", "index", "shard"):
        binding = metadata[label]
        require(
            inventory.regular_identity(root / binding["path"]) == binding["stat"],
            "Bound " + label + " changed or was replaced",
        )
    require(
        implementation_identity() == implementation == _LOADED_IMPLEMENTATION,
        "Reader/helper implementation changed",
    )


def _open_bound(root, shard):
    path = root / shard["path"]
    require(
        inventory.regular_identity(path) == shard["stat"], "Shard changed before open"
    )
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        require(
            inventory.identity(os.fstat(fd)) == shard["stat"],
            "Shard changed while opening",
        )
        length = 8 + shard["header_bytes"]
        raw = os.pread(fd, length, 0)
        require(
            len(raw) == length
            and hashlib.sha256(raw).hexdigest() == shard["length_and_header_sha256"],
            "Opened shard header checksum mismatch",
        )
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read(root, metadata, plan, implementation, expected=None):
    """One header verification and exactly one pread per unique row/component."""
    _unchanged(root, metadata, implementation)
    payloads, spans = {}, []
    fd = _open_bound(root, metadata["shard"])
    try:
        for span in plan["spans"]:
            raw = os.pread(fd, span["bytes"], span["offset"])
            require(len(raw) == span["bytes"], "Truncated affine row read")
            digest = hashlib.sha256(raw).hexdigest()
            if expected is not None:
                require(
                    digest == expected[(span["row"], span["kind"])],
                    "Selected affine payload checksum mismatch",
                )
            payloads.setdefault(span["row"], {})[span["kind"]] = raw
            spans.append({**span, "sha256": digest})
        require(
            inventory.identity(os.fstat(fd)) == metadata["shard"]["stat"],
            "Shard changed during row read",
        )
    finally:
        os.close(fd)
    _unchanged(root, metadata, implementation)
    return payloads, spans


def bind_rows(
    directory,
    *,
    prefix,
    rows,
    shard_name=None,
    index_name="model.safetensors.index.json",
    max_rows=MAX_ROWS,
    max_read_bytes=MAX_READ_BYTES,
    max_header_bytes=inventory.MAX_HEADER_BYTES,
    max_file_bytes=MAX_FILE_BYTES,
):
    """Explicitly read/hash a bounded selected row set; never scan whole weights.

    max_read_bytes bounds all selected tensor payload bytes read by this call.
    Header and config/index JSON have separate fixed bounds and are not tensor
    payload. Bind does not read rows twice merely to establish their identities.
    """
    root = _root(directory)
    implementation = implementation_identity()
    require(
        implementation == _LOADED_IMPLEMENTATION, "Implementation changed after import"
    )
    metadata, plan, _, _ = _metadata(
        root,
        prefix,
        rows,
        shard_name=shard_name,
        index_name=index_name,
        max_rows=max_rows,
        max_read_bytes=max_read_bytes,
        max_header_bytes=max_header_bytes,
        max_file_bytes=max_file_bytes,
    )
    _, spans = _read(root, metadata, plan, implementation)
    result = {
        "format": FORMAT,
        "checkpoint_directory": str(root),
        "prefix": prefix,
        "implementation_sha256": implementation,
        **metadata,
        "plan": {**plan, "spans": spans},
        "binding_payload_bytes_read": plan["total_read_bytes"],
        "opened_header_verification_bytes": 8 + metadata["shard"]["header_bytes"],
        "full_file_hashed": False,
        "scope": "Only selected affine rows and config/index/header are bound. Raw bytes; no full-file integrity, other-shard duplicate detection, decode, numerical or model/training qualification.",
    }
    require(len(canonical(result)) <= MAX_IDENTITY_BYTES, "Identity exceeds size bound")
    return result


def load_identity(path, *, expected_sha256):
    value, _ = inventory.read_metadata(Path(path), MAX_IDENTITY_BYTES)
    require(
        identity_sha256(value) == _digest(expected_sha256),
        "Pinned identity digest mismatch",
    )
    return value


class AffineRowReader:
    def __init__(
        self,
        directory,
        identity,
        *,
        expected_sha256,
        max_rows=MAX_ROWS,
        max_read_bytes=MAX_READ_BYTES,
        max_header_bytes=inventory.MAX_HEADER_BYTES,
        max_file_bytes=MAX_FILE_BYTES,
    ):
        require(isinstance(identity, dict), "Expected identity mapping")
        raw = canonical(identity)
        require(len(raw) <= MAX_IDENTITY_BYTES, "Identity exceeds size bound")
        require(
            hashlib.sha256(raw).hexdigest() == _digest(expected_sha256),
            "Pinned identity digest mismatch",
        )
        bound = self._identity = json.loads(raw)
        require(
            bound.get("format") == FORMAT and bound.get("full_file_hashed") is False,
            "Unsupported row identity",
        )
        self._root = _root(directory)
        require(
            bound.get("checkpoint_directory") == str(self._root),
            "Identity belongs to another directory",
        )
        self._implementation = implementation_identity()
        require(
            bound.get("implementation_sha256")
            == self._implementation
            == _LOADED_IMPLEMENTATION,
            "Identity implementation differs",
        )
        require(
            isinstance(bound.get("plan"), dict)
            and isinstance(bound.get("index"), dict)
            and isinstance(bound.get("shard"), dict),
            "Malformed identity",
        )
        metadata, plan, header, config = _metadata(
            self._root,
            bound.get("prefix"),
            bound["plan"].get("rows"),
            shard_name=bound["shard"].get("path"),
            index_name=bound["index"].get("path"),
            max_rows=max_rows,
            max_read_bytes=max_read_bytes,
            max_header_bytes=max_header_bytes,
            max_file_bytes=max_file_bytes,
        )
        require(
            all(bound.get(k) == v for k, v in metadata.items()),
            "Bound metadata/header/stat differs",
        )
        unhashed = json.loads(json.dumps(bound["plan"]))
        require(isinstance(unhashed.get("spans"), list), "Missing row spans")
        self._expected = {}
        for span in unhashed["spans"]:
            require(isinstance(span, dict), "Malformed row span")
            digest = _digest(span.pop("sha256", None))
            self._expected[(span.get("row"), span.get("kind"))] = digest
        # Binding may have duplicate requests; the actual row set is deduplicated.
        original_mapping = unhashed.get("request_to_unique")
        require(
            isinstance(original_mapping, list)
            and original_mapping
            and len(original_mapping) <= max_rows
            and list(dict.fromkeys(original_mapping)) == list(range(len(plan["rows"])))
            and all(
                type(i) is int and 0 <= i < len(plan["rows"]) for i in original_mapping
            ),
            "Invalid original request mapping",
        )
        plan["request_to_unique"] = original_mapping
        require(unhashed == plan, "Bound row geometry differs")
        require(
            bound.get("binding_payload_bytes_read") == plan["total_read_bytes"]
            and bound.get("opened_header_verification_bytes")
            == 8 + metadata["shard"]["header_bytes"],
            "Bound byte accounting differs",
        )
        self._metadata, self._header, self._config = metadata, header, config
        self._pin = expected_sha256
        self._max_rows, self._max_read_bytes = max_rows, max_read_bytes
        _unchanged(self._root, self._metadata, self._implementation)

    def read(self, rows):
        """Return unique-row immutable bytes with a request-to-unique mapping.

        Refuses any unbound row. Every selected row/component hash is checked even
        if metadata stat guards miss corruption. No cache, decode or full scan.
        """
        plan = planner.plan_rows(
            self._header,
            self._config,
            self._identity["prefix"],
            rows,
            data_start=8 + self._metadata["shard"]["header_bytes"],
            file_bytes=self._metadata["shard"]["file_bytes"],
            max_rows=self._max_rows,
            max_read_bytes=self._max_read_bytes,
        )
        require(
            all(
                (span["row"], span["kind"]) in self._expected for span in plan["spans"]
            ),
            "Requested row is not bound",
        )
        payloads, spans = _read(
            self._root, self._metadata, plan, self._implementation, self._expected
        )
        receipt = {
            "format": "affine-selected-row-read-v1",
            "identity_sha256": self._pin,
            "prefix": self._identity["prefix"],
            "shard": self._metadata["shard"]["path"],
            "plan": {**plan, "spans": spans},
            "payload_bytes_read": plan["total_read_bytes"],
            "opened_header_verification_bytes": 8
            + self._metadata["shard"]["header_bytes"],
            "selected_payload_checksums_verified": True,
            "full_file_rehashed": False,
            "decoded": False,
            "scope": self._identity["scope"],
        }
        return {
            "payloads": tuple(payloads[row] for row in plan["rows"]),
            "rows": tuple(plan["rows"]),
            "request_to_unique": tuple(plan["request_to_unique"]),
            "receipt": receipt,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("bind", "read"):
        p = sub.add_parser(name)
        p.add_argument("checkpoint", type=Path)
        p.add_argument("--rows", nargs="+", required=True, type=int)
        if name == "bind":
            p.add_argument("--prefix", required=True)
            p.add_argument("--output", type=Path, required=True)
        else:
            p.add_argument("--identity", type=Path, required=True)
            p.add_argument("--identity-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "bind":
            value = bind_rows(args.checkpoint, prefix=args.prefix, rows=args.rows)
            with args.output.open("x") as stream:
                json.dump(value, stream, indent=2, allow_nan=False)
                stream.write("\n")
            print(
                json.dumps(
                    {
                        "identity": str(args.output),
                        "canonical_sha256": identity_sha256(value),
                    }
                )
            )
        else:
            value = load_identity(args.identity, expected_sha256=args.identity_sha256)
            result = AffineRowReader(
                args.checkpoint, value, expected_sha256=args.identity_sha256
            ).read(args.rows)
            print(json.dumps(result["receipt"], indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("Affine row read refused: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
