"""CPU-only hash I/O policy checks; no MLX import, GPU or checkpoint scan."""

import errno
import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from exl3_mlx import prefix_disk as disk


def test_real_local_file_hash_and_darwin_fcntl(tmp_path):
    path = tmp_path / "small.bin"
    raw = bytes(range(256)) * 37 + b"unaligned-tail"
    path.write_bytes(raw)
    policies = []
    actual_fcntl = disk.fcntl.fcntl
    with mock.patch.object(disk.fcntl, "fcntl", wraps=actual_fcntl) as call:
        actual = disk.file_hash(path, policy_callback=policies.append)
    assert actual == hashlib.sha256(raw).hexdigest()
    assert len(policies) == 1
    if sys.platform == "darwin":
        # This is an actual local-filesystem call, not a mocked acceptance.
        assert policies[0]["mode"] == "uncached"
        assert call.call_args.args[1:] == (48, 1)
    else:
        assert policies[0]["mode"] == "buffered"
        call.assert_not_called()


def test_empty_file_preserves_sha(tmp_path):
    path = tmp_path / "empty"
    path.touch()
    assert disk.file_hash(path) == hashlib.sha256(b"").hexdigest()


def test_non_darwin_uses_buffered_hash_without_fcntl(tmp_path, monkeypatch):
    path = tmp_path / "weights"
    path.write_bytes(b"same mathematical identity")
    monkeypatch.setattr(disk.sys, "platform", "linux")
    policies = []
    with mock.patch.object(disk.fcntl, "fcntl") as call:
        result = disk.file_hash(path, policy_callback=policies.append)
    call.assert_not_called()
    assert policies == [{"mode": "buffered", "reason": "unsupported-platform", "platform": "linux"}]
    assert result == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "code", sorted({errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.ENOTTY})
)
def test_unsupported_fcntl_explicit_fallback_keeps_full_hash(tmp_path, monkeypatch, code):
    path = tmp_path / "weights"
    raw = b"unaligned" * 513
    path.write_bytes(raw)
    monkeypatch.setattr(disk.sys, "platform", "darwin")
    policies = []
    with mock.patch.object(disk.fcntl, "fcntl", side_effect=OSError(code, "unsupported")):
        result = disk.file_hash(path, policy_callback=policies.append)
    assert policies == [{"mode": "buffered", "reason": "unsupported-fcntl", "errno": code}]
    assert result == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("code", [errno.EBADF, errno.EIO, errno.EPERM])
def test_unexpected_fcntl_error_propagates_without_silent_fallback(tmp_path, monkeypatch, code):
    path = tmp_path / "weights"
    path.write_bytes(b"abc")
    monkeypatch.setattr(disk.sys, "platform", "darwin")
    policies = []
    with (
        mock.patch.object(disk.fcntl, "fcntl", side_effect=OSError(code, "unexpected")),
        pytest.raises(OSError) as caught,
    ):
        disk.file_hash(path, policy_callback=policies.append)
    assert caught.value.errno == code
    assert policies == []


def test_darwin_abi_constant_fallback_when_python_does_not_export_it(monkeypatch):
    monkeypatch.setattr(disk.sys, "platform", "darwin")
    monkeypatch.delattr(disk.fcntl, "F_NOCACHE", raising=False)
    with mock.patch.object(disk.fcntl, "fcntl", return_value=0) as call:
        assert disk._hash_read_policy(17) == {
            "mode": "uncached",
            "mechanism": "darwin-F_NOCACHE",
            "command": 48,
        }
    call.assert_called_once_with(17, 48, 1)


def test_policy_is_configured_before_first_read_and_reads_stay_bounded(tmp_path):
    path = tmp_path / "multi-block.bin"
    block = bytes(range(256)) * 1024
    expected = hashlib.sha256()
    with path.open("wb") as out:
        for _ in range(65):  # 16.25 MiB; three8 MiB-bounded payload reads.
            out.write(block)
            expected.update(block)
    events = []
    real_open = Path.open

    class Source:
        def __init__(self, raw):
            self.raw = raw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.raw.close()

        def fileno(self):
            return self.raw.fileno()

        def read(self, size):
            events.append(("read", size))
            assert size == 8 << 20
            return self.raw.read(size)

    def opening(self, *args, **kwargs):
        return Source(real_open(self, *args, **kwargs))

    def policy(fd):
        assert fd >= 0
        events.append(("policy", fd))
        return {"mode": "uncached"}

    with (
        mock.patch.object(Path, "open", opening),
        mock.patch.object(disk, "_hash_read_policy", policy),
    ):
        result = disk.file_hash(path)
    assert events[0][0] == "policy"
    assert len([event for event in events if event[0] == "read"]) == 4  # Includes EOF.
    assert result == expected.hexdigest()


def test_progress_reports_policy_once_and_identity_records_unchanged(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    progress = []
    records, observed = disk.hash_payloads(tmp_path, [second, first], progress=progress.append)
    assert records == {
        "a": {"sha256": hashlib.sha256(b"alpha").hexdigest(), "bytes": 5},
        "b": {"sha256": hashlib.sha256(b"beta").hexdigest(), "bytes": 4},
    }
    assert len(observed) == 2
    policies = [p for p in progress if "read policy:" in p]
    assert len(policies) == 1
    assert json.loads(policies[0].split("read policy: ", 1)[1])["mode"] in {"uncached", "buffered"}
    assert progress[-1] == "Prefix disk identity: hashed payload 2/2"


def test_mutation_during_hash_still_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "weights"
    path.write_bytes(b"abc")
    original = disk.file_hash

    def changed(path, **kwargs):
        result = original(path, **kwargs)
        with path.open("ab") as out:
            out.write(b"changed")
        return result

    monkeypatch.setattr(disk, "file_hash", changed)
    with pytest.raises(RuntimeError, match="changed during hashing"):
        disk.hash_payloads(tmp_path, [path])


def test_later_mutation_still_rejected_at_scan_end(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")

    def progress(message):
        if message.endswith("hashed payload 1/2"):
            first.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="changed during identity scan"):
        disk.hash_payloads(tmp_path, [first, second], progress=progress)
