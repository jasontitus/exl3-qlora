import hashlib
import json

import mlx.core as mx
import mlx.nn as nn
import pytest
from test_block_training import Dense

from exl3_mlx.native import LoRALinear
from exl3_mlx.train import load_adapter, load_examples, save_adapter


class TokenizerStub:
    def encode(self, text, add_special_tokens=False):
        return type("Encoding", (), {"ids": [3, 4, 5]})()


def test_dataset_boundaries_truncation_and_invalid_schema(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"text":"hello"}\n{"input_ids":[6,7,8]}\n')
    assert load_examples(p, TokenizerStub(), seq_length=2, vocab_size=9) == [
        [0, 3, 4],
        [6, 7, 8],
    ]
    p.write_text('{"messages":[]}\n')
    with pytest.raises(ValueError, match="exactly text or input_ids"):
        load_examples(p, TokenizerStub(), seq_length=2, vocab_size=9)
    p.write_text('{"input_ids":[true,3]}\n')
    with pytest.raises(ValueError, match="invalid text token"):
        load_examples(p, TokenizerStub(), seq_length=2, vocab_size=9)


def test_adapter_save_load_identity_and_prediction_parity(tmp_path):
    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = LoRALinear(Dense(128, 128), rank=2, alpha=4.0)
            self._root = tmp_path / "base"
            self.adapter_config = {
                "rank": 2,
                "alpha": 4.0,
                "adapter_layers": [0],
                "expert_adapter_layers": [],
            }

        def __call__(self, x):
            return self.linear(x)

    base = tmp_path / "base"
    (base / "engram").mkdir(parents=True)
    for name in (
        "config.json",
        "expert-manifest-v2.json",
        "engram/engram-manifest.json",
        "engram-token-map.u32.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    ):
        (base / name).write_text("{}")
    model = SmallModel()
    model.linear.lora_b = mx.random.normal((2, 128)) * 0.01
    x = mx.random.normal((3, 128))
    expected = model(x)
    mx.eval(expected)
    save_adapter(model, tmp_path / "adapter", {"step": 1})
    model.linear.lora_b = mx.zeros((2, 128))
    load_adapter(model, tmp_path / "adapter")
    assert mx.array_equal(expected, model(x)).item()
    (base / "config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="identity mismatch"):
        load_adapter(model, tmp_path / "adapter")


@pytest.fixture
def saved_adapter(tmp_path):
    from mlx.utils import tree_flatten

    class SmallModel(nn.Module):
        def __init__(self, root):
            super().__init__()
            self.linear = LoRALinear(Dense(8, 12), rank=2, alpha=4.0)
            self._root = root
            self.adapter_config = dict(
                rank=2, alpha=4.0, adapter_layers=[0], expert_adapter_layers=[]
            )

    root = tmp_path / "base"
    (root / "engram").mkdir(parents=True)
    for name in (
        "config.json",
        "expert-manifest-v2.json",
        "engram/engram-manifest.json",
        "engram-token-map.u32.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    ):
        (root / name).write_text("{}")
    model = SmallModel(root)
    model.linear.lora_b = mx.full((2, 12), 0.125)
    directory = tmp_path / "adapter"
    save_adapter(model, directory, {"step": 1})
    original = dict(tree_flatten(model.trainable_parameters()))
    mx.eval(original)
    return model, directory, original


def assert_adapter_unchanged(model, original):
    from mlx.utils import tree_flatten

    current = dict(tree_flatten(model.trainable_parameters()))
    assert current.keys() == original.keys()
    assert all(mx.array_equal(current[key], value).item() for key, value in original.items())


def write_integrity_receipt(directory):
    # load_adapter needs only these two fields of a training checkpoint receipt;
    # optimizer/RNG validation belongs to load_checkpoint.
    state = {
        "format": "exl3-mlx-training-state-v1",
        "files": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("adapter.safetensors", "adapter_config.json")
        },
    }
    (directory / "trainer_state.json").write_text(json.dumps(state))


def test_adapter_rejects_changed_finite_payload_without_mutation(saved_adapter):
    model, directory, original = saved_adapter
    path = directory / "adapter.safetensors"
    weights = mx.load(str(path))
    key = next(iter(weights))
    weights[key] = weights[key].at[0, 0].add(1.0)
    mx.save_safetensors(str(path), weights)
    with pytest.raises(ValueError, match="weight checksum mismatch"):
        load_adapter(model, directory)
    assert_adapter_unchanged(model, original)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_adapter_rejects_nonfinite_even_with_matching_checksums(saved_adapter, bad):
    model, directory, original = saved_adapter
    path = directory / "adapter.safetensors"
    weights = mx.load(str(path))
    key = next(iter(weights))
    weights[key] = weights[key].at[0, 0].add(bad)
    mx.save_safetensors(str(path), weights)
    config_path = directory / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["adapter_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config))
    write_integrity_receipt(directory)
    with pytest.raises(ValueError, match="nonfinite"):
        load_adapter(model, directory)
    assert_adapter_unchanged(model, original)


def test_adapter_rejects_changed_config_using_training_receipt(saved_adapter):
    model, directory, original = saved_adapter
    write_integrity_receipt(directory)
    path = directory / "adapter_config.json"
    config = json.loads(path.read_text())
    config["training"] = {"changed": True}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration checksum mismatch"):
        load_adapter(model, directory)
    assert_adapter_unchanged(model, original)


def test_legacy_training_receipt_verifies_adapter_and_rejects_corruption(saved_adapter):
    model, directory, original = saved_adapter
    config_path = directory / "adapter_config.json"
    config = json.loads(config_path.read_text())
    del config["adapter_sha256"]
    config_path.write_text(json.dumps(config))
    write_integrity_receipt(directory)
    model.linear.lora_b = mx.zeros_like(model.linear.lora_b)
    load_adapter(model, directory)
    assert_adapter_unchanged(model, original)
    path = directory / "adapter.safetensors"
    weights = mx.load(str(path))
    key = next(iter(weights))
    weights[key] = weights[key].at[0, 0].add(1.0)
    mx.save_safetensors(str(path), weights)
    with pytest.raises(ValueError, match="weight checksum mismatch"):
        load_adapter(model, directory)
    assert_adapter_unchanged(model, original)


def test_adapter_rejects_missing_checksum_without_mutation(saved_adapter):
    model, directory, original = saved_adapter
    config_path = directory / "adapter_config.json"
    config = json.loads(config_path.read_text())
    del config["adapter_sha256"]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="no saved checksum"):
        load_adapter(model, directory)
    assert_adapter_unchanged(model, original)


@pytest.mark.parametrize("option", ["adapter", "resume", "resume_adapter"])
def test_model_settings_rejects_config_tamper_before_model_allocation(saved_adapter, option):
    from types import SimpleNamespace

    from exl3_mlx.workflow import model_settings

    model, directory, original = saved_adapter
    write_integrity_receipt(directory)
    path = directory / "adapter_config.json"
    config = json.loads(path.read_text())
    config["rank"] = 1_000_000
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration checksum mismatch"):
        model_settings(SimpleNamespace(**{option: str(directory)}))
    assert_adapter_unchanged(model, original)


def test_adapter_accepts_large_finite_values_with_matching_checksum(saved_adapter):
    from mlx.utils import tree_flatten

    model, directory, _ = saved_adapter
    path = directory / "adapter.safetensors"
    weights = mx.load(str(path))
    key = next(iter(weights))
    weights[key] = mx.full(weights[key].shape, 1e30, dtype=weights[key].dtype)
    mx.save_safetensors(str(path), weights)
    config_path = directory / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["adapter_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config))
    load_adapter(model, directory)
    actual = dict(tree_flatten(model.trainable_parameters()))
    assert mx.array_equal(actual[key], weights[key]).item()
