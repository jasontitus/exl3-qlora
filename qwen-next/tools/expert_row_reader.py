"""Bounded local Sushi expert payload reads with explicitly pinned identities.

`bind` scans ONE selected projection shard in chunks and records full-file and
per-expert span hashes. Reading then verifies only the selected expert's exact
raw trellis/suh/svh bytes against that identity. Neither operation decodes or
normalizes weights. Window14 decoder/model/training qualification is unchanged.
No network, framework imports, mmap, caching or GPU work is performed.
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

MAX_SHARD_BYTES = 512 << 20
MAX_EXPERT_BYTES = 8 << 20
MAX_IDENTITY_BYTES = 16 << 20
SCAN_CHUNK_BYTES = 1 << 20
MAX_EXPERTS = 4096
KINDS = ("trellis", "suh", "svh")
FORMAT = "sushi-uniform-expert-payload-identity-v1"


def _helper(name):
    path = Path(__file__).resolve().with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("expert_reader_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inventory = _helper("checkpoint_inventory")
sushi = _helper("sushi_expert_views")


class ReaderError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReaderError(message)


def _positive(value, name):
    require(type(value) is int and value > 0, f"{name} must be a positive integer")
    return value


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def identity_sha256(value):
    """Canonical manifest hash to pin separately from the manifest file itself."""
    return hashlib.sha256(canonical(value)).hexdigest()


def _digest(value, name):
    require(
        isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value),
        f"Invalid SHA256: {name}",
    )
    return value


def implementation_identity():
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__).resolve(),
            Path(inventory.__file__).resolve(),
            Path(sushi.__file__).resolve(),
        )
    }


_LOADED_IMPLEMENTATION = implementation_identity()


def _root(directory):
    root = Path(directory).absolute()
    require(
        root.is_dir() and not root.is_symlink(),
        "Expected non-symlink checkpoint directory",
    )
    return root.resolve(strict=True)


def _current_metadata(
    root,
    shard_name,
    prefix,
    *,
    index_name,
    max_shard_bytes,
    max_header_bytes,
    max_expert_bytes,
):
    for name, value in (
        ("max_shard_bytes", max_shard_bytes),
        ("max_header_bytes", max_header_bytes),
        ("max_expert_bytes", max_expert_bytes),
    ):
        _positive(value, name)
    inventory.basename(shard_name, "shard")
    inventory.basename(index_name, "index")
    require(shard_name.endswith(".safetensors"), "Expected a safetensors shard")
    require(isinstance(prefix, str), "Projection prefix must be a string")
    stat = inventory.regular_identity(root / shard_name)
    require(
        stat["size_bytes"] <= max_shard_bytes,
        "Selected shard exceeds bind/read file-size limit",
    )
    config, config_binding = inventory.read_metadata(
        root / "config.json", inventory.MAX_METADATA_BYTES
    )
    index, index_binding = inventory.read_metadata(
        root / index_name, inventory.MAX_METADATA_BYTES
    )
    weight_map = index.get("weight_map")
    require(
        isinstance(weight_map, dict) and bool(weight_map), "Missing index weight_map"
    )
    for owner in weight_map.values():
        inventory.basename(owner, "index owner")
    report = inventory.inspect_shard(
        root / shard_name, max_header_bytes=max_header_bytes
    )
    require(report["stat"] == stat, "Selected shard changed during metadata inspection")
    header = {
        t["name"]: {k: t[k] for k in ("dtype", "shape", "data_offsets")}
        for t in report["tensors"]
    }
    require(
        {name for name, owner in weight_map.items() if owner == shard_name}
        == set(header),
        "Selected shard index ownership mismatch",
    )
    header["__metadata__"] = report["metadata"]
    text = config.get("text_config", {})
    experts = text.get("num_experts") if isinstance(text, dict) else None
    require(
        type(experts) is int and 0 < experts <= MAX_EXPERTS, "Unsupported expert count"
    )
    plans = [
        sushi.expert_views(
            header,
            prefix=prefix,
            expert=i,
            config=config,
            data_start=8 + report["header_bytes"],
            file_bytes=stat["size_bytes"],
        )
        for i in range(experts)
    ]
    require(
        all(p["total_read_bytes"] <= max_expert_bytes for p in plans),
        "One expert exceeds bounded read limit",
    )
    return {
        "config": config_binding,
        "index": index_binding,
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
        "plans": plans,
    }


def _unchanged(root, metadata, implementation):
    for name in ("config", "index", "shard"):
        binding = metadata[name]
        require(
            inventory.regular_identity(root / binding["path"]) == binding["stat"],
            f"Bound {name} changed or was replaced",
        )
    require(
        implementation_identity() == implementation,
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
    except BaseException:
        os.close(fd)
        raise
    return fd


def bind_projection(
    directory,
    *,
    shard_name,
    prefix,
    index_name="model.safetensors.index.json",
    max_shard_bytes=MAX_SHARD_BYTES,
    max_header_bytes=inventory.MAX_HEADER_BYTES,
    max_expert_bytes=MAX_EXPERT_BYTES,
    scan_chunk_bytes=SCAN_CHUNK_BYTES,
):
    """Explicit streaming identity capture; scans one file, never a full model.

    The resulting manifest is local and inode/stat-bound. Persist its canonical
    SHA separately. It is consistency evidence, not authenticated provenance.
    """
    _positive(scan_chunk_bytes, "scan_chunk_bytes")
    require(scan_chunk_bytes <= SCAN_CHUNK_BYTES, "Scan chunks may not exceed1MiB")
    root, implementation = _root(directory), implementation_identity()
    require(
        implementation == _LOADED_IMPLEMENTATION, "Reader/helper changed after import"
    )
    metadata = _current_metadata(
        root,
        shard_name,
        prefix,
        index_name=index_name,
        max_shard_bytes=max_shard_bytes,
        max_header_bytes=max_header_bytes,
        max_expert_bytes=max_expert_bytes,
    )
    rows = json.loads(json.dumps(metadata["plans"]))
    regions = sorted(
        (view["offset_bytes"], view["length_bytes"], i, kind)
        for i, plan in enumerate(rows)
        for kind, view in plan["views"].items()
    )
    full = hashlib.sha256()
    cursor = 0
    with os.fdopen(_open_bound(root, metadata["shard"]), "rb") as stream:
        # Header participates in full-file identity, but is never tensor data.
        boundaries = [(0, 8 + metadata["shard"]["header_bytes"], None, None), *regions]
        for offset, length, expert, kind in boundaries:
            require(
                offset == cursor, "Projection spans do not cover shard contiguously"
            )
            segment = hashlib.sha256()
            remaining = length
            while remaining:
                raw = stream.read(min(remaining, scan_chunk_bytes))
                require(bool(raw), "Truncated shard during identity capture")
                full.update(raw)
                segment.update(raw)
                remaining -= len(raw)
                cursor += len(raw)
            if expert is not None:
                rows[expert]["views"][kind]["sha256"] = segment.hexdigest()
            else:
                require(
                    segment.hexdigest()
                    == metadata["shard"]["length_and_header_sha256"],
                    "Header changed during identity capture",
                )
        require(cursor == metadata["shard"]["file_bytes"], "Shard coverage mismatch")
        require(
            inventory.identity(os.fstat(stream.fileno())) == metadata["shard"]["stat"],
            "Shard changed during payload identity capture",
        )
    _unchanged(root, metadata, implementation)
    return {
        "format": FORMAT,
        "checkpoint_directory": str(root),
        "prefix": prefix,
        "implementation_sha256": implementation,
        "config": metadata["config"],
        "index": metadata["index"],
        "shard": {**metadata["shard"], "file_sha256": full.hexdigest()},
        "experts": rows,
        "binding_scan_bytes": cursor,
        "limits": {
            "max_shard_bytes": max_shard_bytes,
            "max_header_bytes": max_header_bytes,
            "max_expert_bytes": max_expert_bytes,
            "scan_chunk_bytes": scan_chunk_bytes,
            "max_experts": MAX_EXPERTS,
        },
        "scope": "Exact local payload identity for one uniform routed projection bank; raw bytes only. No decode, normalization, full-model integrity, numerical parity or training qualification.",
    }


def load_identity(path, *, expected_sha256):
    """Read bounded strict JSON and verify its separately supplied canonical pin."""
    value, _ = inventory.read_metadata(Path(path), MAX_IDENTITY_BYTES)
    _digest(expected_sha256, "identity")
    require(
        identity_sha256(value) == expected_sha256, "Pinned identity digest mismatch"
    )
    return value


class ExpertRowReader:
    def __init__(
        self,
        directory,
        identity,
        *,
        expected_sha256,
        max_shard_bytes=MAX_SHARD_BYTES,
        max_header_bytes=inventory.MAX_HEADER_BYTES,
        max_expert_bytes=MAX_EXPERT_BYTES,
    ):
        _digest(expected_sha256, "identity")
        require(
            isinstance(identity, dict) and identity_sha256(identity) == expected_sha256,
            "Pinned identity digest mismatch",
        )
        self._identity = json.loads(canonical(identity))
        bound = self._identity
        require(bound.get("format") == FORMAT, "Unsupported payload identity format")
        self._root = _root(directory)
        require(
            bound.get("checkpoint_directory") == str(self._root),
            "Identity belongs to another checkpoint directory",
        )
        self._implementation = implementation_identity()
        require(
            self._implementation == _LOADED_IMPLEMENTATION,
            "Reader/helper changed after import",
        )
        require(
            bound.get("implementation_sha256") == self._implementation,
            "Identity implementation differs",
        )
        require(
            isinstance(bound.get("shard"), dict)
            and isinstance(bound.get("index"), dict),
            "Malformed shard/index identity",
        )
        self._metadata = _current_metadata(
            self._root,
            bound["shard"].get("path"),
            bound.get("prefix"),
            index_name=bound["index"].get("path"),
            max_shard_bytes=max_shard_bytes,
            max_header_bytes=max_header_bytes,
            max_expert_bytes=max_expert_bytes,
        )
        for name in ("config", "index"):
            require(
                bound.get(name) == self._metadata[name],
                f"Bound {name} metadata differs",
            )
        expected_shard = {
            **self._metadata["shard"],
            "file_sha256": _digest(bound["shard"].get("file_sha256"), "full file"),
        }
        require(bound["shard"] == expected_shard, "Bound shard header/stat differs")
        experts = bound.get("experts")
        require(
            isinstance(experts, list) and len(experts) == len(self._metadata["plans"]),
            "Bound expert table differs",
        )
        for declared, plan in zip(experts, self._metadata["plans"]):
            require(
                isinstance(declared, dict) and isinstance(declared.get("views"), dict),
                "Malformed expert identity",
            )
            unhashed = json.loads(json.dumps(declared))
            require(
                set(unhashed["views"]) == set(KINDS),
                "Incomplete expert payload identity",
            )
            for kind in KINDS:
                require(
                    isinstance(unhashed["views"][kind], dict), "Malformed view identity"
                )
                _digest(unhashed["views"][kind].pop("sha256", None), kind)
            require(unhashed == plan, "Bound expert geometry differs")
        self._expected_sha256 = expected_sha256
        self._max_expert_bytes = max_expert_bytes
        _unchanged(self._root, self._metadata, self._implementation)

    def read(self, expert):
        """Return three raw immutable byte strings and a serializable receipt.

        The full-file hash was checked during bind, not rescanned here. Selected
        span hashes are rechecked even if a stat guard were to miss corruption.
        No retained cache or full-bank decoded allocation is made.
        """
        require(
            type(expert) is int and 0 <= expert < len(self._identity["experts"]),
            "Expert index out of bounds",
        )
        _unchanged(self._root, self._metadata, self._implementation)
        plan = self._identity["experts"][expert]
        require(
            plan["total_read_bytes"] <= self._max_expert_bytes,
            "Expert read byte cap exceeded",
        )
        payloads = {}
        fd = _open_bound(self._root, self._metadata["shard"])
        try:
            for kind in KINDS:
                view = plan["views"][kind]
                raw = os.pread(fd, view["length_bytes"], view["offset_bytes"])
                require(
                    len(raw) == view["length_bytes"], "Truncated expert payload read"
                )
                require(
                    hashlib.sha256(raw).hexdigest() == view["sha256"],
                    f"Expert {kind} payload checksum mismatch",
                )
                payloads[kind] = raw
            require(
                inventory.identity(os.fstat(fd)) == self._metadata["shard"]["stat"],
                "Shard changed during expert read",
            )
        finally:
            os.close(fd)
        _unchanged(self._root, self._metadata, self._implementation)
        receipt = {
            "format": "sushi-uniform-expert-read-v1",
            "identity_sha256": self._expected_sha256,
            "expert": expert,
            "prefix": self._identity["prefix"],
            "shard": self._identity["shard"]["path"],
            "bound_file_sha256": self._identity["shard"]["file_sha256"],
            "full_file_rehashed_on_read": False,
            "payload_bytes_read": sum(map(len, payloads.values())),
            "views": json.loads(json.dumps(plan["views"])),
            "selected_payload_checksums_verified": True,
            "decoded": False,
            "scope": self._identity["scope"],
        }
        return {"payloads": payloads, "receipt": receipt}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    bind = modes.add_parser(
        "bind", help="Explicit bounded full scan of one projection shard"
    )
    bind.add_argument("checkpoint", type=Path)
    bind.add_argument("--shard", required=True)
    bind.add_argument("--prefix", required=True)
    bind.add_argument("--output", required=True, type=Path)
    read = modes.add_parser(
        "read", help="Verify one expert; print receipt, not payload bytes"
    )
    read.add_argument("checkpoint", type=Path)
    read.add_argument("--identity", type=Path, required=True)
    read.add_argument("--identity-sha256", required=True)
    read.add_argument("--expert", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "bind":
            value = bind_projection(
                args.checkpoint, shard_name=args.shard, prefix=args.prefix
            )
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
            result = ExpertRowReader(
                args.checkpoint, value, expected_sha256=args.identity_sha256
            ).read(args.expert)
            print(json.dumps(result["receipt"], indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Expert payload read refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
