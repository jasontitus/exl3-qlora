"""Synthetic descriptor tests only; no device, model, lease or child process."""
import hashlib
import os
from pathlib import Path

import pytest
from exl3_mlx import evaluation_receipts as receipts


def sample(tmp_path, data=b"bounded digest\x00\xff"):
    path = tmp_path / "input"
    path.write_bytes(data)
    return path.resolve()


@pytest.mark.parametrize("data", [b"", b"x", b"a" * (2 * 1024 * 1024 + 19)])
def test_exact_digest_and_chunk_bound(tmp_path, monkeypatch, data):
    path = sample(tmp_path, data)
    original = os.read
    reads = []

    def read(fd, size):
        reads.append(size)
        return original(fd, size)

    monkeypatch.setattr(receipts.os, "read", read)
    assert receipts.hash_file(path, max_bytes=max(1, len(data))) == hashlib.sha256(data).hexdigest()
    assert reads and max(reads) <= 1024 * 1024
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("Whole-file read"))
    assert receipts.hash_file(path, max_bytes=max(1, len(data))) == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("limit", [True, False, 1.0, 0, -1, None])
def test_invalid_limit_before_open(tmp_path, monkeypatch, limit):
    path = sample(tmp_path)
    monkeypatch.setattr(receipts.os, "open", lambda *_: pytest.fail("Must reject before open"))
    with pytest.raises(ValueError, match="Invalid hash byte limit"):
        receipts.hash_file(path, max_bytes=limit)


@pytest.mark.parametrize("kind", ["symlink", "parent", "relative", "fifo", "directory", "oversize"])
def test_unsafe_inputs_before_open(tmp_path, monkeypatch, kind):
    path = sample(tmp_path)
    limit = 100
    if kind == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(path)
        path = alias
    elif kind == "parent":
        alias = tmp_path / "parent"
        alias.symlink_to(tmp_path, target_is_directory=True)
        path = alias / path.name
    elif kind == "relative":
        path = Path("relative")
    elif kind == "fifo":
        path = tmp_path / "fifo"
        os.mkfifo(path)
    elif kind == "directory":
        path = tmp_path
    else:
        limit = 1
    monkeypatch.setattr(receipts.os, "open", lambda *_: pytest.fail("Must reject before open"))
    with pytest.raises((ValueError, FileNotFoundError)):
        receipts.hash_file(path, max_bytes=limit)


def mutate_during_read(path, monkeypatch, action):
    original = os.read
    changed = False

    def read(fd, size):
        nonlocal changed
        data = original(fd, size)
        if data and not changed:
            changed = True
            action()
        return data

    monkeypatch.setattr(receipts.os, "read", read)
    with pytest.raises(ValueError):
        receipts.hash_file(path, max_bytes=path.stat().st_size)


@pytest.mark.parametrize("kind", ["grow", "truncate", "rewrite", "replace", "mode"])
def test_read_mutations(tmp_path, monkeypatch, kind):
    raw = b"a" * (1024 * 1024 + 11)
    path = sample(tmp_path, raw)

    def action():
        if kind == "grow":
            with path.open("ab") as stream:
                stream.write(b"x")
        elif kind == "truncate":
            path.write_bytes(b"")
        elif kind == "rewrite":
            path.write_bytes(b"b" * len(raw))
        elif kind == "replace":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(raw)
            replacement.replace(path)
        else:
            path.chmod(0o400)

    mutate_during_read(path, monkeypatch, action)


def test_replace_between_stat_and_open(tmp_path, monkeypatch):
    path = sample(tmp_path)
    original = os.open

    def open_file(name, flags):
        other = tmp_path / "other"
        other.write_bytes(path.read_bytes())
        other.replace(path)
        return original(name, flags)

    monkeypatch.setattr(receipts.os, "open", open_file)
    with pytest.raises(ValueError, match="before hash"):
        receipts.hash_file(path)


def test_fifo_swap_at_open_does_not_block(tmp_path, monkeypatch):
    path = sample(tmp_path)
    original = os.open

    def open_file(name, flags):
        assert flags & os.O_NONBLOCK
        path.unlink()
        os.mkfifo(path)
        return original(name, flags)

    monkeypatch.setattr(receipts.os, "open", open_file)
    with pytest.raises(ValueError, match="regular file"):
        receipts.hash_file(path)


def test_exception_closes_descriptor(tmp_path, monkeypatch):
    path = sample(tmp_path)
    original = os.close
    closed = []

    def close(fd):
        closed.append(fd)
        return original(fd)

    monkeypatch.setattr(receipts.os, "close", close)
    monkeypatch.setattr(receipts.os, "read", lambda *_: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(OSError, match="injected"):
        receipts.hash_file(path)
    assert len(closed) == 1
    with pytest.raises(OSError):
        os.fstat(closed[0])


def test_no_identity_cache(tmp_path):
    path = sample(tmp_path, b"first")
    before = receipts.hash_file(path)
    path.write_bytes(b"other")
    assert receipts.hash_file(path) != before


def test_parent_alias_introduced_during_read(tmp_path, monkeypatch):
    directory = tmp_path / "source"
    directory.mkdir()
    path = sample(directory, b"a" * (1024 * 1024 + 11))

    def action():
        moved = tmp_path / "moved"
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)

    mutate_during_read(path, monkeypatch, action)
