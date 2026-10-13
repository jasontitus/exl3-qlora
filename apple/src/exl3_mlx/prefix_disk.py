"""Opt-in checksummed, immutable frozen-prefix disk entries (CPU-importable).

This is a cache of trusted local model computation, not a security sandbox or an
attestation. A directory has one exact model/implementation/arithmetic binding.
Different train/eval/recompute modes never share entries. Quota exhaustion and
corruption fail closed; entries are never silently evicted or overwritten.
"""

import contextlib
import errno
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

_NAMES = ("h", "pre", "kv0", "kv1", "kv2", "kv3")
_FORMAT = "exl3-frozen-prefix-disk-v1"


def validate_disk_options(directory, quota_gib):
    try:
        valid = (
            type(quota_gib) in (int, float) and math.isfinite(quota_gib) and 0 <= quota_gib <= 1024
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("prefix_disk_cache_gib must be finite and in 0..1024")
    if directory is not None and (
        not isinstance(directory, (str, Path)) or not str(directory).strip()
    ):
        raise ValueError("prefix_disk_cache must be a nonempty local path")
    if (directory is None) != (quota_gib == 0):
        raise ValueError("Set both prefix_disk_cache and positive prefix_disk_cache_gib")
    if directory is not None and int(quota_gib * 2**30) < 1:
        raise ValueError("Prefix disk cache quota must be at least one byte")
    return int(quota_gib * 2**30)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash_read_policy(fd):
    """Use descriptor-local uncached reads on Darwin, with explicit fallback.

    Apple's SDK sys/fcntl.h defines F_NOCACHE=48. Its fcntl(2) manual specifies
    that a nonzero argument disables data caching for this descriptor. This is
    not a global cache purge or a promise of lower process/whole-system memory.
    No read-ahead policy is changed. Other systems keep normal buffered I/O.
    """
    if sys.platform != "darwin":
        return {"mode": "buffered", "reason": "unsupported-platform", "platform": sys.platform}
    command = getattr(fcntl, "F_NOCACHE", 48)
    try:
        fcntl.fcntl(fd, command, 1)
    except OSError as error:
        unsupported = {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.ENOTTY}
        if error.errno not in unsupported:
            raise
        return {"mode": "buffered", "reason": "unsupported-fcntl", "errno": error.errno}
    return {"mode": "uncached", "mechanism": "darwin-F_NOCACHE", "command": command}


def file_hash(path, *, policy_callback=None):
    """Full SHA256, in bounded sequential reads; I/O policy does not affect bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        policy = _hash_read_policy(source.fileno())
        if policy_callback:
            policy_callback(policy)
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def hash_payloads(root, paths, *, progress=None):
    """Hash full immutable input bytes once per model construction, not per row.

    Never reuse a persisted stat-only hash memo. Existing model stat guards must
    also be checked after this scan and on each cache access. Hashing detects
    prior in-place edits even if an mtime was restored between processes.
    """
    root = Path(root).resolve()
    records, observed = {}, {}
    policies = set()

    def report_policy(policy):
        key = tuple(sorted(policy.items()))
        if progress and key not in policies:
            progress("Prefix disk identity read policy: " + json.dumps(policy, sort_keys=True))
        policies.add(key)

    paths = sorted(set(Path(p).resolve() for p in paths))
    for index, path in enumerate(paths):
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Prefix payload must be a regular checkpoint file")
        before = fingerprint(path)
        digest = file_hash(path, policy_callback=report_policy)
        if fingerprint(path) != before:
            raise RuntimeError(f"Frozen prefix payload changed during hashing: {path}")
        observed[path] = before
        records[str(path.relative_to(root))] = {"sha256": digest, "bytes": before[2]}
        if progress:
            progress(f"Prefix disk identity: hashed payload {index + 1}/{len(paths)}")
    if any(fingerprint(path) != expected for path, expected in observed.items()):
        raise RuntimeError("Frozen prefix payload changed during identity scan")
    return records, observed


def _save(path, arrays):
    import mlx.core as mx

    mx.save_safetensors(str(path), arrays)


def _load(path):
    import mlx.core as mx

    arrays = mx.load(str(path))
    mx.eval(arrays)
    return arrays


def _same_leaf_bytes(left, right):
    """Exact duplicate-producer check; bounded byte comparison including BF16."""
    import numpy as np

    if left.shape != right.shape or str(left.dtype) != str(right.dtype):
        return False

    def buffer(value):
        if str(value.dtype) == "mlx.core.bfloat16":
            import mlx.core as mx

            value = value.view(mx.uint16)
        return memoryview(np.ascontiguousarray(np.asarray(value)).reshape(-1)).cast("B")

    a, b = buffer(left), buffer(right)
    return all(a[i : i + (1 << 20)] == b[i : i + (1 << 20)] for i in range(0, len(a), 1 << 20))


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _specs(arrays):
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.nbytes}
        for name, value in arrays.items()
    }


def _validate_arrays(arrays, key, layout):
    if set(arrays) != set(_NAMES):
        raise ValueError("Prefix disk entry must contain exactly six state leaves")
    h, pre, *kv = [arrays[name] for name in _NAMES]
    if (
        len(h.shape) != 4
        or len(pre.shape) != 3
        or tuple(h.shape[:2]) != key.shape
        or tuple(pre.shape[:2]) != key.shape
        or h.shape[2] != pre.shape[2]
        or any(len(v.shape) != 3 or v.shape[0] != key.shape[0] for v in kv)
        or any(tuple(v.shape[:2]) != key.shape for v in kv[2:])
        or any(v.shape[1] > key.shape[1] for v in kv[:2])
    ):
        raise ValueError("Prefix disk tensor shapes disagree with complete token shape")
    dtypes = [str(v.dtype).removeprefix("mlx.core.") for v in (h, pre, *kv)]
    if any(d not in {"float16", "float32", "bfloat16"} for d in dtypes[:4]):
        raise ValueError("Invalid floating prefix disk tensor dtype")
    if dtypes[4:] != ["int32", "uint8"]:
        raise ValueError("Invalid discrete prefix disk tensor dtype")
    if layout and (
        h.shape[2:] != (layout["hc_mult"], layout["hidden_size"])
        or pre.shape[-1] != layout["hc_mult"]
        or kv[0].shape[-1] != layout["head_dim"]
        or kv[1].shape[-1] != layout["index_head_dim"]
    ):
        raise ValueError("Prefix disk shapes disagree with frozen model layout")


def _validate_header(path, expected):
    """Check bounded safetensors layout before a loader can allocate tensors."""
    widths = {"F16": 2, "BF16": 2, "F32": 4, "I32": 4, "U8": 1}
    names = {"F16": "float16", "BF16": "bfloat16", "F32": "float32", "I32": "int32", "U8": "uint8"}
    with path.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 1 <= length <= 65536 or length + 8 > path.stat().st_size:
            raise ValueError("Invalid prefix safetensors header size")
        header = json.loads(stream.read(length))
    header.pop("__metadata__", None)
    if set(header) != set(_NAMES) or not isinstance(expected, dict) or set(expected) != set(_NAMES):
        raise ValueError("Invalid prefix safetensors tensor names")
    spans = []
    for name, spec in header.items():
        shape, dtype, offsets = spec.get("shape"), spec.get("dtype"), spec.get("data_offsets")
        if (
            not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)
            or dtype not in widths
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int or n < 0 for n in offsets)
        ):
            raise ValueError("Invalid prefix safetensors tensor layout")
        lo, hi = offsets
        size = math.prod(shape) * widths[dtype]
        actual = {"shape": shape, "dtype": names[dtype], "bytes": size}
        declared = dict(expected[name])
        declared["dtype"] = declared["dtype"].removeprefix("mlx.core.")
        if hi - lo != size or hi + length + 8 > path.stat().st_size or actual != declared:
            raise ValueError("Prefix safetensors layout disagrees with tensor receipt")
        spans.append((lo, hi))
    end = 0
    for lo, hi in sorted(spans):
        if lo != end:
            raise ValueError("Prefix safetensors ranges overlap or contain gaps")
        end = hi
    if end + length + 8 != path.stat().st_size:
        raise ValueError("Prefix safetensors payload has trailing or missing bytes")


class FrozenPrefixDisk:
    def __init__(self, directory, *, quota_bytes, binding, save=_save, load=_load):
        if type(quota_bytes) is not int or quota_bytes < 1:
            raise ValueError("Prefix disk quota must be a positive integer byte count")
        directory = Path(directory).expanduser()
        if directory.is_symlink():
            raise ValueError("Prefix disk directory must not be a symlink")
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory.resolve()
        stat = self.directory.stat()
        self._directory_identity = stat.st_dev, stat.st_ino
        self.quota_bytes = quota_bytes
        self.binding = json.loads(canonical(binding))
        self.binding_sha256 = hashlib.sha256(canonical(self.binding)).hexdigest()
        self._save, self._load = save, load
        self.stats = {"hits": 0, "misses": 0, "writes": 0, "read_bytes": 0, "written_bytes": 0}
        with self._lock():
            manifest = self.directory / "binding.json"
            if manifest.exists():
                self._verify_binding()
            else:
                if any(p.name != ".lock" for p in self.directory.iterdir()):
                    raise ValueError("Unbound nonempty prefix disk cache; use a fresh directory")
                initial = {"format": _FORMAT, "binding": self.binding}
                if len(canonical(initial)) > self.quota_bytes:
                    raise ValueError("Prefix disk quota cannot fit binding")
                pending = self.directory / ".binding-pending"
                try:
                    self._write_json(pending, initial)
                    os.rename(pending, manifest)
                    _sync_dir(self.directory)
                finally:
                    pending.unlink(missing_ok=True)
            self._usage()

    @contextlib.contextmanager
    def _lock(self):
        if self.directory.is_symlink():
            raise ValueError("Prefix disk directory was replaced with a symlink")
        stat = self.directory.stat()
        if (stat.st_dev, stat.st_ino) != self._directory_identity:
            raise ValueError("Prefix disk directory identity changed")
        path = self.directory / ".lock"
        if path.is_symlink():
            raise ValueError("Prefix disk lock must not be a symlink")
        with path.open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _write_json(self, path, value):
        with path.open("xb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        _sync_dir(path.parent)

    def _verify_binding(self):
        path = self.directory / "binding.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or unsafe prefix disk binding")
        if json.loads(path.read_bytes()) != {"format": _FORMAT, "binding": self.binding}:
            raise ValueError("Prefix disk binding mismatch (model/source/arithmetic changed)")

    def _usage(self):
        total = 0
        for path in self.directory.rglob("*"):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError("Unsafe file in prefix disk cache")
            if path.is_file():
                total += path.stat().st_size
        if total > self.quota_bytes:
            raise ValueError(
                "Prefix disk cache exceeds quota; raise quota or use a fresh directory"
            )
        return total

    @staticmethod
    def _key(key):
        metadata = {
            "shape": list(key.shape),
            "training": key.training,
            "checkpoint_blocks": key.checkpoint_blocks,
            "tokens_hex": key.tokens.hex(),
        }
        return hashlib.sha256(canonical(metadata)).hexdigest(), metadata

    def _read(self, path, metadata, key):
        if (
            path.is_symlink()
            or not path.is_dir()
            or {p.name for p in path.iterdir()} != {"state.safetensors", "entry.json"}
        ):
            raise ValueError("Incomplete or unsafe prefix disk entry")
        state, receipt_path = path / "state.safetensors", path / "entry.json"
        if any(p.is_symlink() or not p.is_file() for p in (state, receipt_path)):
            raise ValueError("Unsafe prefix disk entry payload")
        receipt = json.loads(receipt_path.read_bytes())
        before = fingerprint(state)
        if (
            receipt.get("format") != _FORMAT
            or receipt.get("binding_sha256") != self.binding_sha256
            or receipt.get("key") != metadata
            or receipt.get("file_bytes") != state.stat().st_size
            or receipt.get("sha256") != file_hash(state)
        ):
            raise ValueError("Prefix disk checksum/key/binding mismatch")
        if fingerprint(state) != before:
            raise RuntimeError("Prefix disk entry changed during checksum verification")
        _validate_header(state, receipt.get("tensors"))
        arrays = self._load(state)
        if fingerprint(state) != before or receipt_path.read_bytes() != canonical(receipt):
            raise RuntimeError("Prefix disk entry changed during load")
        _validate_arrays(arrays, key, self.binding.get("state_layout"))
        if _specs(arrays) != receipt.get("tensors"):
            raise ValueError("Prefix disk tensor shape/dtype metadata mismatch")
        self.stats["read_bytes"] += state.stat().st_size
        return arrays["h"], arrays["pre"], tuple(arrays[f"kv{i}"] for i in range(4))

    def get(self, key):
        name, metadata = self._key(key)
        with self._lock():
            self._verify_binding()
            self._usage()
            path = self.directory / name
            if not path.exists():
                self.stats["misses"] += 1
                return None
            result = self._read(path, metadata, key)
            self.stats["hits"] += 1
            return result

    def put(self, key, state):
        name, metadata = self._key(key)
        arrays = dict(zip(_NAMES, (state[0], state[1], *state[2]), strict=True))
        _validate_arrays(arrays, key, self.binding.get("state_layout"))
        with self._lock():
            self._verify_binding()
            used = self._usage()
            target = self.directory / name
            if target.exists():
                # Entries are immutable. A duplicate producer verifies the saved
                # entry and never overwrites it or silently repairs corruption.
                saved = self._read(target, metadata, key)
                old = dict(zip(_NAMES, (saved[0], saved[1], *saved[2]), strict=True))
                if any(not _same_leaf_bytes(arrays[name], old[name]) for name in _NAMES):
                    raise ValueError(
                        "Different computed prefix bytes for an existing exact cache key"
                    )
                return
            reservation = sum(v.nbytes for v in arrays.values()) + len(key.tokens) * 2 + 65536
            if used + reservation > self.quota_bytes:
                raise ValueError("Prefix disk quota cannot fit entry plus atomic-write scratch")
            stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.directory))
            try:
                payload = stage / "state.safetensors"
                self._save(payload, arrays)
                with payload.open("rb") as stream:
                    os.fsync(stream.fileno())
                receipt = {
                    "format": _FORMAT,
                    "binding_sha256": self.binding_sha256,
                    "key": metadata,
                    "sha256": file_hash(payload),
                    "file_bytes": payload.stat().st_size,
                    "tensors": _specs(arrays),
                }
                self._write_json(stage / "entry.json", receipt)
                size = sum(p.stat().st_size for p in stage.iterdir())
                if size > reservation:
                    raise ValueError("Serialized prefix entry exceeded reserved quota")
                self._usage()
                # Validate the exact serialized tensors before publication.
                self._read(stage, metadata, key)
                os.rename(stage, target)
                _sync_dir(self.directory)
                self.stats["writes"] += 1
                self.stats["written_bytes"] += size
            finally:
                if stage.exists():
                    shutil.rmtree(stage)

    def report(self):
        with self._lock():
            self._verify_binding()
            used = self._usage()
            return {
                **self.stats,
                "directory": str(self.directory),
                "quota_bytes": self.quota_bytes,
                "stored_bytes": used,
                "binding_sha256": self.binding_sha256,
                "scope": "full payload hashes; exact complete-token and separate mode keys",
            }
