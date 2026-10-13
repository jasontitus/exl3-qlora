"""CPU tests of disk persistence, integrity, quota and bounded RAM integration."""

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from test_prefix_cache_cpu import key, state, state_bytes

from exl3_mlx.prefix_cache import FrozenPrefixCache
from exl3_mlx.prefix_disk import (
    FrozenPrefixDisk,
    canonical,
    file_hash,
    hash_payloads,
    validate_disk_options,
)


def disk(path, *, quota=1 << 20, binding=None, **kwargs):
    return FrozenPrefixDisk(
        path,
        quota_bytes=quota,
        binding=binding or {"implementation": "test", "frozen_blocks": 2},
        save=lambda p, a: save_file(a, str(p)),
        load=lambda p: load_file(str(p)),
        **kwargs,
    )


def entry(root, token_key=None):
    name, _ = FrozenPrefixDisk._key(token_key or key())
    return root / name


def assert_equal(left, right):
    for a, b in zip((left[0], left[1], *left[2]), (right[0], right[1], *right[2]), strict=True):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype


def test_complete_state_reloads_in_new_process_equivalent_instance(tmp_path):
    first = disk(tmp_path)
    assert first.get(key()) is None
    first.put(key(), state())
    second = disk(tmp_path)
    assert_equal(second.get(key()), state())
    assert second.report()["hits"] == 1
    assert second.get(key(training=False)) is None
    assert second.get(key(checkpoint_blocks=False)) is None
    assert second.get(key(((1, 2, 4),))) is None
    assert second.get(key(((1,), (2,), (3,)))) is None
    assert second.report()["misses"] == 4


def test_disk_hit_survives_ram_eviction_and_disk_only_is_enabled(tmp_path):
    store = disk(tmp_path)
    cache = FrozenPrefixCache(
        state_bytes(state(), key()), frozen_blocks=2, materializer=np.copy, disk=store
    )
    cache.put(key(), state())
    other = key(((4, 5, 6),))
    cache.put(other, state())
    assert cache.report()["evictions"] == 1
    assert_equal(cache.get(key()), state())
    assert cache.report()["entries"] == 1
    assert store.stats["hits"] == 1
    disk_only = FrozenPrefixCache(0, frozen_blocks=2, materializer=np.copy, disk=disk(tmp_path))
    assert disk_only.enabled
    assert_equal(disk_only.get(key()), state())
    assert disk_only.report()["cached_bytes"] == 0
    disk_only.put(key(((7, 8, 9),)), state())
    assert disk_only.report()["entries"] == 0


@pytest.mark.parametrize("change", ["implementation", "frozen_blocks", "payload", "chunk", "mode"])
def test_changed_binding_refused(tmp_path, change):
    original = {
        "implementation": "A",
        "frozen_blocks": 2,
        "payload": "P",
        "chunk": 512,
        "mode": "checkpoint",
    }
    disk(tmp_path, binding=original)
    with pytest.raises(ValueError, match="binding mismatch"):
        disk(tmp_path, binding={**original, change: "changed"})


@pytest.mark.parametrize("kind", ["bytes", "receipt", "missing", "symlink", "binding"])
def test_corruption_is_error_never_silent_cache_miss(tmp_path, kind):
    store = disk(tmp_path)
    store.put(key(), state())
    folder = entry(tmp_path)
    payload = folder / "state.safetensors"
    if kind == "bytes":
        data = bytearray(payload.read_bytes())
        data[-1] ^= 1
        payload.write_bytes(data)
    elif kind == "receipt":
        receipt = json.loads((folder / "entry.json").read_bytes())
        receipt["key"]["tokens_hex"] = "0000"
        (folder / "entry.json").write_bytes(canonical(receipt))
    elif kind == "missing":
        payload.unlink()
    elif kind == "symlink":
        saved = tmp_path.parent / (tmp_path.name + "-state")
        payload.rename(saved)
        payload.symlink_to(saved)
    else:
        (tmp_path / "binding.json").write_text("{}")
    with pytest.raises((ValueError, RuntimeError)):
        store.get(key())


def test_shape_dtype_header_checked_before_loading(tmp_path):
    store = disk(tmp_path)
    store.put(key(), state())
    receipt_path = entry(tmp_path) / "entry.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["tensors"]["h"]["shape"][-1] += 1
    receipt_path.write_bytes(canonical(receipt))
    store._load = lambda _: pytest.fail("malformed header should fail before tensor loading")
    with pytest.raises(ValueError, match="layout disagrees"):
        store.get(key())


def test_invalid_state_and_layout_refused(tmp_path):
    store = disk(
        tmp_path,
        binding={
            "state_layout": {"hc_mult": 4, "hidden_size": 9, "head_dim": 8, "index_head_dim": 4}
        },
    )
    with pytest.raises(ValueError, match="model layout"):
        store.put(key(), state())
    ordinary = disk(tmp_path / "other")
    bad = state()
    bad = (bad[0], bad[1], (*bad[2][:2], bad[2][2].astype(np.float32), bad[2][3]))
    with pytest.raises(ValueError, match="discrete"):
        ordinary.put(key(), bad)


def test_quota_includes_atomic_scratch_before_serializer(tmp_path):
    store = disk(tmp_path, quota=70000)
    store.put(key(), state())
    original = file_hash(entry(tmp_path) / "state.safetensors")
    store.quota_bytes = store.report()["stored_bytes"] + state_bytes(state(), key())
    store._save = lambda *_: pytest.fail("quota must be checked before allocating disk scratch")
    with pytest.raises(ValueError, match="atomic-write scratch"):
        store.put(key(((4, 5, 6),)), state())
    assert file_hash(entry(tmp_path) / "state.safetensors") == original
    assert not list(tmp_path.glob(".pending-*"))


def test_interrupted_entry_never_reused_and_partial_bytes_count(tmp_path):
    store = disk(tmp_path)
    incomplete = tmp_path / ".pending-interrupted"
    incomplete.mkdir()
    (incomplete / "partial").write_bytes(b"x" * 1000)
    before = store.report()["stored_bytes"]
    assert before >= 1000
    assert store.get(key()) is None
    assert incomplete.exists()
    store.quota_bytes = before - 1
    with pytest.raises(ValueError, match="exceeds quota"):
        store.get(key())


def test_serializer_failure_preserves_other_entries_and_removes_owned_temp(tmp_path):
    store = disk(tmp_path)
    store.put(key(), state())

    def fail(path, arrays):
        path.write_bytes(b"partial")
        raise RuntimeError("injected write failure")

    store._save = fail
    with pytest.raises(RuntimeError, match="injected"):
        store.put(key(((4, 5, 6),)), state())
    assert_equal(store.get(key()), state())
    assert not list(tmp_path.glob(".pending-*"))


def test_payload_full_hash_changes_even_after_mtime_restoration(tmp_path):
    import os

    path = tmp_path / "weights.bin"
    path.write_bytes(b"abc")
    first, _ = hash_payloads(tmp_path, [path])
    old = path.stat()
    path.write_bytes(b"abd")
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    second, _ = hash_payloads(tmp_path, [path])
    assert first != second


@pytest.mark.parametrize(
    "directory,quota",
    [
        (None, 1),
        ("path", 0),
        ("", 1),
        ("path", float("nan")),
        ("path", float("inf")),
        ("path", True),
        ("path", -1),
        ("path", 10**1000),
        ("path", 1025),
        ("  ", 1),
    ],
)
def test_options_fail_closed(directory, quota):
    with pytest.raises(ValueError):
        validate_disk_options(directory, quota)


def test_disabled_default_and_valid_options():
    assert validate_disk_options(None, 0) == 0
    assert validate_disk_options(Path("cache"), 0.5) == 1 << 29


def test_cache_directory_symlink_substitution_is_rejected(tmp_path):
    original = tmp_path / "cache"
    store = disk(original)
    store.put(key(), state())
    moved = tmp_path / "moved"
    original.rename(moved)
    original.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        store.get(key())


def test_cache_directory_replacement_is_rejected(tmp_path):
    original = tmp_path / "cache"
    store = disk(original)
    original.rename(tmp_path / "old")
    original.mkdir()
    with pytest.raises(ValueError, match="identity changed"):
        store.get(key())


def test_duplicate_producer_must_match_every_saved_leaf_byte(tmp_path):
    store = disk(tmp_path)
    store.put(key(), state())
    second = disk(tmp_path)
    second.put(key(), state())
    before = file_hash(entry(tmp_path) / "state.safetensors")
    changed = state()
    changed[2][3][0, 0, 0] += 1
    with pytest.raises(ValueError, match="Different computed prefix bytes"):
        second.put(key(), changed)
    assert file_hash(entry(tmp_path) / "state.safetensors") == before
    assert_equal(second.get(key()), state())


def test_duplicate_comparison_is_bitwise_for_signed_zero(tmp_path):
    store = disk(tmp_path)
    original = state()
    original[0][0, 0, 0, 0] = 0.0
    store.put(key(), original)
    changed = state()
    changed[0][0, 0, 0, 0] = -0.0
    with pytest.raises(ValueError, match="Different computed prefix bytes"):
        store.put(key(), changed)
