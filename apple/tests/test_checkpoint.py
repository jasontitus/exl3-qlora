import json

import mlx.core as mx
import numpy as np
import pytest
from exl3_mlx.checkpoint import MARKERS, Checkpoint
from exl3_mlx.cli import train_smoke
from exl3_mlx.remote import fetch_linear
from exl3_mlx.validation import validate_linear
from safetensors.numpy import save_file


def tensors(rate=3):
    rng = np.random.default_rng(5)
    return {
        "layer.trellis": rng.integers(0, 65536, (8, 8, int(rate * 16)), dtype=np.uint16).view(
            np.int16
        ),
        "layer.suh": rng.uniform(-0.1, 0.1, 128).astype(np.float16),
        "layer.svh": np.ones(128, dtype=np.float16),
        "layer.mul1": np.array(MARKERS["mul1"], dtype=np.uint32).view(np.int32),
    }


def test_cross_shard_loading_and_legacy_signs(tmp_path):
    data = tensors(2.5)
    del data["layer.suh"]
    data["layer.su"] = np.full(8, -21846, dtype=np.int16)  # 0xaaaa: +1,-1 alternating
    save_file({"layer.trellis": data.pop("layer.trellis")}, tmp_path / "a.safetensors")
    save_file(data, tmp_path / "b.safetensors")
    (tmp_path / "config.json").write_text('{"model_type":"deepseek_v41"}')
    ckpt = Checkpoint(tmp_path)
    assert ckpt.summary()["full_model_training_supported"] is False
    assert ckpt.summary()["matrices_by_bpw"] == {"2.5": 1}
    base = ckpt.load_linear("layer")
    np.testing.assert_array_equal(np.array(base.suh), np.tile([1.0, -1.0], 64))
    assert validate_linear(base)["decode_bit_exact"]


@pytest.mark.parametrize("error", ["marker", "both", "shape", "scale", "nan"])
def test_reject_malformed_checkpoint(tmp_path, error):
    data = tensors()
    if error == "marker":
        data["layer.mul1"] = np.array(123, dtype=np.int32)
    elif error == "both":
        data["layer.mcg"] = np.array(MARKERS["mcg"], dtype=np.uint32).view(np.int32)
    elif error == "shape":
        data["layer.trellis"] = data["layer.trellis"].reshape(-1)
    elif error == "scale":
        data["layer.scale"] = np.array(1.0, dtype=np.float16)
    else:
        data["layer.suh"][0] = np.nan
    path = tmp_path / "model.safetensors"
    save_file(data, path)
    with pytest.raises(ValueError):
        Checkpoint(path).load_linear("layer")


def test_duplicate_tensor_and_load_budget(tmp_path):
    save_file(tensors(), tmp_path / "a.safetensors")
    with pytest.raises(ValueError, match="budget"):
        Checkpoint(tmp_path).load_linear("layer", max_packed_bytes=1)
    save_file(tensors(), tmp_path / "b.safetensors")
    with pytest.raises(ValueError, match="Duplicate"):
        Checkpoint(tmp_path)


def test_train_artifacts_contain_only_adapter_and_reload(tmp_path):
    path = tmp_path / "base.safetensors"
    save_file(tensors(), path)
    report = train_smoke(Checkpoint(path).load_linear("layer"), output=tmp_path / "adapter")
    assert report["adapter_reload_exact"] and report["base_unchanged"]
    assert set(mx.load(str(tmp_path / "adapter/adapter.safetensors"))) == {"lora_a", "lora_b"}


def test_range_extraction_and_payload_budget(tmp_path, monkeypatch):
    import exl3_mlx.remote as remote
    from safetensors.numpy import save

    blob = save(tensors())
    calls = []

    def fake_range(url, start, length):
        calls.append((start, length))
        return blob[start : start + length]

    monkeypatch.setattr(remote, "_range", fake_range)
    kwargs = dict(repo="test/repo", revision="a" * 40, shard="layer.safetensors", key="layer")
    with pytest.raises(ValueError, match="budget"):
        fetch_linear(**kwargs, output=tmp_path / "too-big.safetensors", max_bytes=1)
    assert len(calls) == 2  # header only; no payload was fetched
    out = tmp_path / "extracted.safetensors"
    report = fetch_linear(**kwargs, output=out)
    assert report["payload_bytes"] == sum(v.nbytes for v in tensors().values())
    assert Checkpoint(out).load_linear("layer").bpw == 3
    assert json.loads(out.with_suffix(".provenance.json").read_text())["revision"] == "a" * 40
    with pytest.raises(FileExistsError):
        fetch_linear(**kwargs, output=out)
    with pytest.raises(ValueError, match="commit"):
        fetch_linear("test/repo", "main", "layer.safetensors", "layer", tmp_path / "bad")


def test_server_ignoring_range_is_not_read(monkeypatch):
    import exl3_mlx.remote as remote

    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, length):
            pytest.fail("Must reject a full-shard response before reading its body")

    monkeypatch.setattr(remote.urllib.request, "urlopen", lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match="byte range"):
        remote._range("https://example.com/file", 0, 8)
