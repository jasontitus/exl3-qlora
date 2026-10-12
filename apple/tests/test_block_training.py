import mlx.core as mx
import mlx.nn as nn
import numpy as np
from test_streaming_moe import make_checkpoint, make_expert

from exl3_mlx.attention import ShortAttention
from exl3_mlx.deepseek import TrainingBlock
from exl3_mlx.hyperconnections import HyperConnection, expand_streams
from exl3_mlx.moe import DeepSeekGate, DeepSeekMoE
from exl3_mlx.native import LoRALinear
from exl3_mlx.recompute import recompute
from exl3_mlx.streaming import MatrixStore


class Dense(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.in_features, self.out_features = k, n
        self.weight = mx.random.normal((n, k)) * 0.03
        self.freeze()

    def __call__(self, x):
        return x @ self.weight.T


def make_block(store, layer):
    c = {
        "num_attention_heads": 2,
        "head_dim": 128,
        "qk_rope_head_dim": 32,
        "rope_theta": 10000,
        "compress_rope_theta": 160000,
        "index_topk": 512,
        "rms_norm_eps": 1e-20,
        "compress_ratios": [2, 2],
        "o_groups": 2,
        "sliding_window": 2,
        "rope_scaling": {
            "original_max_position_embeddings": 65536,
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    }
    linears = {
        "wq_a": LoRALinear(Dense(128, 128), rank=2),
        "wq_b": Dense(128, 256),
        "wkv": Dense(128, 128),
        "wo_b": LoRALinear(Dense(128, 128), rank=2),
    }
    norms = {k: mx.ones(128) for k in ("q", "kv", "compressor")}
    compressor = {"wkv": Dense(128, 128), "wgate": Dense(128, 128)} if layer == 0 else None
    attn = ShortAttention(
        layer, c, linears, norms, mx.zeros(2), [Dense(128, 64), Dense(128, 64)], compressor
    )
    gate = DeepSeekGate(mx.random.normal((5, 128)) * 0.1, mx.arange(5) * 0.1, top_k=2)
    ffn = DeepSeekMoE(gate, [make_expert(store, i) for i in range(5)], make_expert(store, 5))
    hcs = [
        HyperConnection(mx.random.normal((24, 512)) * 0.02, mx.ones(3), mx.zeros(24))
        for _ in range(2)
    ]
    return TrainingBlock(attn, ffn, *hcs, mx.ones(128), mx.ones(128))


def test_checkpoint_recomputation_preserves_shared_kv_and_adapter_gradients(tmp_path):
    mx.random.seed(19)
    store = MatrixStore(
        make_checkpoint(tmp_path / "experts.safetensors"), cache_bytes=0, chunk_size=128
    )
    model = nn.Sequential(make_block(store, 0), make_block(store, 1))
    # Use nonzero B so all adapter inputs and shared-KV paths carry gradients.
    for b in model.layers:
        for name in ("wq_a", "wo_b"):
            b.attn.linears[name].lora_b = mx.random.normal((2, 128)) * 0.01
    x = mx.random.normal((1, 3, 128))
    h, pre = expand_streams(x)

    def loss(m, checkpoint):
        state = (h, pre, mx.zeros((1, 0, 128)), mx.zeros((0,)))
        for block in m.layers:
            a, b, c = recompute(block, *state) if checkpoint else block(*state)
            mx.eval(a, b, c)
            state = (a, b, c, state[-1])
        return mx.mean(a * a) + mx.mean(c * c)

    a, ga = nn.value_and_grad(model, lambda m: loss(m, False))(model)
    b, gb = nn.value_and_grad(model, lambda m: loss(m, True))(model)
    from mlx.utils import tree_flatten

    np.testing.assert_allclose(a.item(), b.item(), rtol=1e-6)
    for (ka, va), (kb, vb) in zip(tree_flatten(ga), tree_flatten(gb), strict=True):
        assert ka == kb and ka.endswith(("lora_a", "lora_b"))
        assert mx.linalg.norm(va).item() > 0
        np.testing.assert_allclose(np.array(va), np.array(vb), rtol=3e-5, atol=1e-7)
    assert store.stats.backward_calls > 0
