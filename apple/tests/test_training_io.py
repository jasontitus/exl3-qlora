import mlx.core as mx
import mlx.nn as nn
import pytest
from exl3_mlx.native import LoRALinear
from exl3_mlx.train import load_adapter, load_examples, save_adapter
from test_block_training import Dense


class TokenizerStub:
    def encode(self, text, add_special_tokens=False):
        return type("Encoding", (), {"ids": [3, 4, 5]})()


def test_dataset_boundaries_truncation_and_invalid_schema(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"text":"hello"}\n{"input_ids":[6,7,8]}\n')
    assert load_examples(p, TokenizerStub(), seq_length=2, vocab_size=9) == [[0, 3, 4], [6, 7, 8]]
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
