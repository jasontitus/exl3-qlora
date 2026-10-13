"""CPU-only fail-closed receipt tests; importing the script never imports MLX."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "frozen_prefix_diagnostic",
    Path(__file__).parents[1] / "scripts" / "diagnose_frozen_prefix.py",
)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


@pytest.fixture
def cache(tmp_path):
    files = (
        "row-0001.safetensors",
        "initial-adapter/adapter.safetensors",
        "initial-adapter/adapter_config.json",
    )
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
    binding = {"data": "source", "model": "metadata", "source": "frozen"}
    manifest = {
        "schema": "frozen-prefix-cache-v1",
        "status": "complete",
        "binding": binding,
        "rows": {
            "1": {
                "base_parity_exact": True,
                "uncached_base": {"loss": 0.5},
                "cached_base": {"loss": 0.5},
            }
        },
        "files": {name: diagnostic.digest(tmp_path / name) for name in files},
    }
    return tmp_path, binding, manifest


def test_valid_cache_and_rows(cache):
    path, binding, manifest = cache
    diagnostic.validate_cache_manifest(manifest, binding, [1], path)
    data = path / "data.jsonl"
    data.write_text(json.dumps({"input_ids": [0, 5, 1], "loss_start": 1}) + "\n")
    assert diagnostic.selected_rows(data, [0])[0]["input_ids"] == [0, 5, 1]
    with pytest.raises(ValueError, match="unique"):
        diagnostic.selected_rows(data, [0, 0])


@pytest.mark.parametrize("field", ["source", "model", "data"])
def test_changed_binding_rejected(cache, field):
    path, binding, manifest = cache
    altered = {**binding, field: "changed"}
    with pytest.raises(ValueError, match="differs"):
        diagnostic.validate_cache_manifest(manifest, altered, [1], path)


@pytest.mark.parametrize("kind", ["incomplete", "missing_row", "parity", "hash", "missing_hash"])
def test_incomplete_or_tampered_cache_rejected(cache, kind):
    path, binding, original = cache
    manifest = copy.deepcopy(original)
    if kind == "incomplete":
        manifest["status"] = "capturing"
    elif kind == "missing_row":
        del manifest["rows"]["1"]
    elif kind == "parity":
        manifest["rows"]["1"]["cached_base"]["loss"] = 0.51
    elif kind == "hash":
        (path / "row-0001.safetensors").write_bytes(b"changed")
    else:
        del manifest["files"]["row-0001.safetensors"]
    with pytest.raises(ValueError):
        diagnostic.validate_cache_manifest(manifest, binding, [1], path)


def test_prefix_symlink_rejected(cache):
    path, binding, manifest = cache
    payload = path / "row-0001.safetensors"
    payload.unlink()
    payload.symlink_to(path / "initial-adapter/adapter.safetensors")
    with pytest.raises(ValueError, match="Unsafe"):
        diagnostic.validate_cache_manifest(manifest, binding, [1], path)
