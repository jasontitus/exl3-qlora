"""Bounded local receipt snapshots; no process or model-provenance attestation.

Callers choose allowed files and supply semantic validators. Capture after the
producer is terminal, validate captured JSON, and recheck after downstream work
immediately before publication. Finite checks cannot defeat arbitrary same-user
tampering after the last check. This module never imports MLX or starts children.
"""

import hashlib
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation_audit import _canonical, _json

DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_FILES = 32
DEFAULT_MAX_HASH_BYTES = 512 * 1024 * 1024


def _require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, info):
        _require(stat.S_ISREG(info.st_mode), "Receipt must be a regular file")
        return cls(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _canonical_path(path):
    path = Path(path)
    _require(path.is_absolute(), "Receipt path must be absolute and canonical")
    _require(path.resolve(strict=True) == path, "Receipt path must not contain symlink aliases")
    return path


def _read(path, max_bytes):
    """Bound bytes through the descriptor, then verify its current pathname."""
    path = _canonical_path(path)
    # NONBLOCK prevents a raced FIFO replacement from blocking before fstat.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        before = FileIdentity.from_stat(os.fstat(fd))
        _require(before.size <= max_bytes, "Receipt exceeds byte limit")
        parts = []
        count = 0
        while True:
            part = os.read(fd, min(1024 * 1024, max_bytes + 1 - count))
            if not part:
                break
            count += len(part)
            _require(count <= max_bytes, "Receipt exceeds byte limit")
            parts.append(part)
        data = b"".join(parts)
        after = FileIdentity.from_stat(os.fstat(fd))
        _require(before == after and len(data) == before.size, "Receipt changed during read")
        current = FileIdentity.from_stat(path.lstat())
        _require(current == before, "Receipt pathname changed during read")
        _canonical_path(path)
        return data, before
    finally:
        os.close(fd)


def hash_file(path: str | Path, *, max_bytes: int = DEFAULT_MAX_HASH_BYTES) -> str:
    """Hash all bytes without retaining them, checking descriptor/path identity.

    The caller must restrict allowed paths and choose a finite byte limit. This
    does not parse or validate file contents, establish provenance, or prevent
    changes after return. No hashes are cached; every call reads the whole file.
    """
    _require(type(max_bytes) is int and max_bytes > 0, "Invalid hash byte limit")
    path = _canonical_path(path)
    initial = path.lstat()
    before = FileIdentity.from_stat(initial)
    _require(before.size <= max_bytes, "File exceeds hash byte limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        _require(
            FileIdentity.from_stat(opened) == before and opened.st_mode == initial.st_mode,
            "File changed before hash read",
        )
        digest = hashlib.sha256()
        count = 0
        while True:
            part = os.read(fd, min(1024 * 1024, max_bytes + 1 - count))
            if not part:
                break
            count += len(part)
            _require(count <= max_bytes and count <= before.size, "File grew during hash read")
            digest.update(part)
        after = os.fstat(fd)
        _require(
            count == before.size
            and FileIdentity.from_stat(after) == before
            and after.st_mode == initial.st_mode,
            "File changed during hash read",
        )
        current = path.lstat()
        _require(
            FileIdentity.from_stat(current) == before and current.st_mode == initial.st_mode,
            "File pathname changed during hash read",
        )
        _canonical_path(path)
        return digest.hexdigest()
    finally:
        os.close(fd)


@dataclass(frozen=True)
class CapturedReceipt:
    role: str
    path: Path
    data: bytes
    sha256: str
    identity: FileIdentity


@dataclass(frozen=True)
class ReceiptSnapshot:
    """Immutable captured bytes; identity evidence is local, hashes are portable."""

    files: tuple[CapturedReceipt, ...]
    max_file_bytes: int

    @property
    def hashes(self):
        return {item.role: item.sha256 for item in self.files}

    def recheck(self):
        for item in self.files:
            data, identity = _read(item.path, self.max_file_bytes)
            _require(
                identity == item.identity and data == item.data,
                f"Receipt changed after capture: {item.role}",
            )

    def validate_json(self, role: str, validator: Callable[[Any], Any]):
        """Validate captured JSON, reject semantic callback edits, and recheck.

        The validator's return value is deliberately ignored. Returning a fresh
        decode prevents a callback retaining the returned mutable object.
        """
        matches = [item for item in self.files if item.role == role]
        _require(len(matches) == 1, f"Unknown receipt role: {role}")
        raw = matches[0].data
        value = _json(raw)
        original = _canonical(value)
        validator(value)
        _require(_canonical(value) == original, "Receipt validator mutated captured JSON")
        self.recheck()
        return _json(raw)


def capture_receipts(
    paths: Mapping[str, str | Path],
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> ReceiptSnapshot:
    """Capture a small explicit report/log set, never recursively discover files.

    The caller must restrict allowed paths (in particular model-payload paths).
    Limits apply to captured raw bytes, not JSON decoder/process peak memory.
    """
    _require(isinstance(paths, Mapping) and 0 < len(paths) <= MAX_FILES, "Invalid receipt set")
    for name, value in (("file", max_file_bytes), ("total", max_total_bytes)):
        _require(type(value) is int and value > 0, f"Invalid {name} byte limit")
    _require(
        all(isinstance(role, str) and role and len(role) <= 128 for role in paths),
        "Invalid receipt role",
    )
    canonical = {role: _canonical_path(path) for role, path in paths.items()}
    _require(len(set(canonical.values())) == len(canonical), "Duplicate receipt paths")
    files = []
    remaining = max_total_bytes
    for role, path in canonical.items():
        data, identity = _read(path, min(max_file_bytes, remaining))
        remaining -= len(data)
        files.append(CapturedReceipt(role, path, data, hashlib.sha256(data).hexdigest(), identity))
    snapshot = ReceiptSnapshot(tuple(files), max_file_bytes)
    snapshot.recheck()
    return snapshot
