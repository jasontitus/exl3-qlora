"""Small actual hidden_states/recompute/Adam cache parity; requires coordinated GPU."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from exl3_mlx.deepseek import DeepSeekTextModel
from exl3_mlx.hyperconnections import hc_pre
from exl3_mlx.prefix_cache import FrozenPrefixCache, frozen_structure_signature
from exl3_mlx.training import evaluate, optimizer_step


class Block(nn.Module):
    def __init__(self, index, *, adapter):
        super().__init__()
        rng = np.random.default_rng(760 + index)
        self.weight = mx.array(rng.normal(0, 0.15, (8, 8)).astype(np.float32))
        self.freeze()
        if adapter:
            self.lora_a = mx.array(rng.normal(0, 0.1, (8, 2)).astype(np.float32))
            self.lora_b = mx.zeros((2, 8))
        self._calls = 0
        self._index = index

    def __call__(self, h, pre, state, embedding):
        self._calls += 1
        value = hc_pre(h, pre)
        out = mx.tanh(value @ self.weight.astype(value.dtype))
        if "lora_a" in self:
            out = out + ((value @ self.lora_a) @ self.lora_b).astype(value.dtype)
        if self._index == 0:
            pooled = out[:, ::2]
            keys = pooled[..., :4]
            selected = mx.broadcast_to(
                mx.arange(2, dtype=mx.int32)[None, None], (h.shape[0], h.shape[1], 2)
            )
            candidates = mx.ones((h.shape[0], h.shape[1], 1), dtype=mx.uint8)
            state = pooled, keys, selected, candidates
        else:
            # Exercise every cached KV leaf in suffix output, including discrete
            # state. The real engine freezes index keys/indices but retains them.
            extra = mx.mean(state[0].astype(mx.float32), axis=(1, 2))
            extra += mx.mean(state[1].astype(mx.float32), axis=(1, 2))
            extra += mx.mean(state[2].astype(mx.float32), axis=(1, 2))
            extra += mx.mean(state[3].astype(mx.float32), axis=(1, 2))
            out = (out.astype(mx.float32) + extra[:, None, None] * 0.01).astype(value.dtype)
        h = (h.astype(mx.float32) + out.astype(mx.float32)[..., None, :] * 0.1).astype(h.dtype)
        pre = mx.sigmoid(mx.mean(h.astype(mx.float32), axis=-1))
        return h, pre, state


def make_model(*, cache_bytes, dtype, attention_layers=(2,), expert_layers=()):
    """Use real model forward with a deterministic small frozen trunk fixture."""
    result = DeepSeekTextModel.__new__(DeepSeekTextModel)
    nn.Module.__init__(result)
    result.config = dict(
        hidden_size=8,
        hc_mult=4,
        head_dim=8,
        index_head_dim=4,
        engram_layer_ids=[],
        max_position_embeddings=32768,
    )
    selected = set(attention_layers) | set(expert_layers)
    result.blocks = [Block(i, adapter=i in selected) for i in range(3)]
    embedding = mx.array(np.random.default_rng(52).normal(size=(12, 8)).astype(np.float32)).astype(
        dtype
    )
    result._archive = SimpleNamespace(array=lambda name, rows: embedding[rows[0] : rows[1]])
    result._engrams = SimpleNamespace(hashes=lambda ids: np.zeros((*ids.shape, 0, 0), np.int64))
    result._store = SimpleNamespace(chunk_size=512)
    result._progress = None
    result.checkpoint_blocks = True
    result.norm = mx.ones(8)
    result.freeze(keys=["norm"], recurse=False)
    result._prefix_cache = FrozenPrefixCache(cache_bytes, frozen_blocks=min(selected))
    result._prefix_sources = {}
    result._prefix_signature = (
        result._frozen_prefix_signature() if result._prefix_cache.enabled else None
    )
    return result


def assert_bits_equal(a, b):
    aa, bb = tree_flatten(a), tree_flatten(b)
    assert [key for key, _ in aa] == [key for key, _ in bb]
    for (key, left), (_, right) in zip(aa, bb, strict=True):
        assert left.dtype == right.dtype and left.shape == right.shape, key
        left = left.view(mx.uint16) if left.dtype == mx.bfloat16 else left
        right = right.view(mx.uint16) if right.dtype == mx.bfloat16 else right
        np.testing.assert_array_equal(np.array(left), np.array(right), err_msg=key)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_frozen_storage_signature_survives_autograd_wrapper_replacement(dtype):
    model = make_model(cache_bytes=0, dtype=dtype)
    model.blocks[0].weight = model.blocks[0].weight.astype(dtype)
    mx.eval(model.parameters())
    original = model.blocks[0].weight
    signature = frozen_structure_signature(original)

    def loss(model):
        # MLX wraps even frozen arrays while tracing. Their storage is unchanged.
        assert model.blocks[0].weight is not original
        assert frozen_structure_signature(model.blocks[0].weight) == signature
        return mx.sum(model.blocks[2].lora_b)

    nn.value_and_grad(model, loss)(model)
    assert model.blocks[0].weight is original
    model.blocks[0].weight = model.blocks[0].weight + 1
    assert frozen_structure_signature(model.blocks[0].weight) != signature


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_cache_miss_hit_and_suffix_updates_are_bitwise_equal_to_uncached(dtype):
    base = make_model(cache_bytes=0, dtype=dtype)
    cached = make_model(cache_bytes=1 << 20, dtype=dtype)
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    base_optim, cached_optim = optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)
    initial_keys = [key for key, _ in tree_flatten(cached.parameters())]

    def loss(model):
        values = model.hidden_states(tokens)
        target = mx.arange(8, dtype=mx.float32) * 0.1
        return mx.mean((values.astype(mx.float32) - target) ** 2)

    for step in range(3):
        base_loss, base_grads = nn.value_and_grad(base, loss)(base)
        cached_loss, cached_grads = nn.value_and_grad(cached, loss)(cached)
        mx.eval(base_loss, cached_loss, base_grads, cached_grads)
        assert_bits_equal((base_loss, base_grads), (cached_loss, cached_grads))
        base_optim.update(base, base_grads)
        cached_optim.update(cached, cached_grads)
        mx.eval(base.parameters(), cached.parameters(), base_optim.state, cached_optim.state)
        assert_bits_equal(base.trainable_parameters(), cached.trainable_parameters())
        assert_bits_equal(base_optim.state, cached_optim.state)
        assert cached._prefix_cache.stats.hits == step
        assert cached._prefix_cache.stats.misses == 1
        # Only frozen prefix forward calls are skipped. The trainable suffix
        # recomputes on every backward, including a cold first cache insertion.
        assert cached.blocks[0]._calls == cached.blocks[1]._calls == 1
        assert cached.blocks[2]._calls == (step + 1) * 2

    assert [key for key, _ in tree_flatten(cached.parameters())] == initial_keys
    assert all("prefix_cache" not in key for key in initial_keys)
    assert_bits_equal(base.hidden_states(tokens), cached.hidden_states(tokens))
    # Different complete IDs miss even with the same first four tokens.
    changed = mx.array([[1, 2, 3, 4, 6]], dtype=mx.int32)
    assert_bits_equal(base.hidden_states(changed), cached.hidden_states(changed))
    assert cached._prefix_cache.stats.misses == 2
    # Eval is deliberately a separate key. Subsequent eval reuses its own entry.
    base.eval()
    cached.eval()
    assert_bits_equal(base.hidden_states(tokens), cached.hidden_states(tokens))
    assert cached._prefix_cache.stats.misses == 3
    calls = cached.blocks[0]._calls
    assert_bits_equal(base.hidden_states(tokens), cached.hidden_states(tokens))
    assert cached.blocks[0]._calls == calls


def test_earlier_expert_adapter_boundary_and_suffix_load_preserve_gradients():
    cached = make_model(
        cache_bytes=1 << 20, dtype=mx.float32, attention_layers=(2,), expert_layers=(1,)
    )
    uncached = make_model(
        cache_bytes=0, dtype=mx.float32, attention_layers=(2,), expert_layers=(1,)
    )
    assert cached._prefix_cache.frozen_blocks == 1
    tokens = mx.array([[1, 2, 3, 4, 5]])
    cached.hidden_states(tokens)
    updates = [(key, value + 0.01) for key, value in tree_flatten(cached.trainable_parameters())]
    cached.load_weights(updates, strict=False)
    uncached.load_weights(updates, strict=False)

    def loss(model):
        return mx.sum(model.hidden_states(tokens).astype(mx.float32) ** 2)

    left = nn.value_and_grad(cached, loss)(cached)
    right = nn.value_and_grad(uncached, loss)(uncached)
    mx.eval(left, right)
    assert_bits_equal(left, right)
    gradients = dict(tree_flatten(left[1]))
    assert mx.linalg.norm(gradients["blocks.1.lora_b"]).item() > 0
    assert mx.linalg.norm(gradients["blocks.2.lora_b"]).item() > 0
    assert cached._prefix_cache.stats.hits == 1


@pytest.mark.parametrize("mutation", ["weight", "trainable", "config", "file"])
def test_frozen_prefix_change_clears_cache_and_fails(mutation, tmp_path):
    model = make_model(cache_bytes=1 << 20, dtype=mx.float32)
    tokens = mx.array([[1, 2, 3, 4, 5]])
    file = tmp_path / "source.bin"
    file.write_bytes(b"old")
    stat = file.stat()
    model._prefix_sources = {file: (stat.st_ino, stat.st_size, stat.st_mtime_ns)}
    model.hidden_states(tokens)
    assert model._prefix_cache.report()["entries"] == 1
    if mutation == "weight":
        model.blocks[0].weight = model.blocks[0].weight + 1
    elif mutation == "trainable":
        model.blocks[0].unfreeze(keys=["weight"], recurse=False)
    elif mutation == "config":
        model.config["hidden_size"] = 9
    else:
        file.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="prefix"):
        model.hidden_states(tokens)
    assert model._prefix_cache.report()["entries"] == 0


def test_all_layer_adapters_leave_cache_inactive():
    model = make_model(cache_bytes=1 << 20, dtype=mx.float32, attention_layers=(0, 1, 2))
    tokens = mx.array([[1, 2, 3]])
    model.hidden_states(tokens)
    model.hidden_states(tokens)
    assert not model._prefix_cache.enabled
    assert model._prefix_cache.report()["entries"] == 0
    assert model.blocks[0]._calls == 2


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_final_training_mode_fit_reuses_cache_without_updates_and_matches_fresh_eval(dtype):
    cached = make_model(cache_bytes=1 << 20, dtype=dtype)
    uncached = make_model(cache_bytes=0, dtype=dtype)
    for model in (cached, uncached):
        model.head = nn.Linear(8, 12, bias=False)
        model.head.weight = (mx.arange(96).reshape(12, 8) / 50).astype(dtype)
        model.head.freeze()
    examples = [
        {"input_ids": [1, 2, 3, 4, 5], "loss_start": 2},
        {"input_ids": [4, 2, 8, 9], "loss_start": 1},
        {"input_ids": [1, 6, 7, 2, 3, 10], "loss_start": 3},
    ]
    optimizer = optim.Adam(learning_rate=1e-3)
    for example in examples:
        optimizer_step(cached, optimizer, [example])
    uncached.load_weights(tree_flatten(cached.trainable_parameters()), strict=False)
    weights = [(key, mx.array(value)) for key, value in tree_flatten(cached.parameters())]
    state = [(key, mx.array(value)) for key, value in tree_flatten(optimizer.state)]
    mx.eval(weights, state)
    calls = [block._calls for block in cached.blocks]
    assert cached.training
    assert cached._prefix_cache.stats.misses == 3

    warm = evaluate(cached, examples, include_token_metrics=True, preserve_mode=True)
    assert cached.training
    assert cached._prefix_cache.stats.hits == 3
    assert cached._prefix_cache.stats.misses == 3
    assert cached.blocks[0]._calls == calls[0] and cached.blocks[1]._calls == calls[1]
    assert cached.blocks[2]._calls == calls[2] + 3  # Forward scoring only, no backward.
    assert_bits_equal(cached.parameters(), tree_unflatten(weights))
    assert_bits_equal(optimizer.state, tree_unflatten(state))
    fresh = evaluate(uncached, examples, include_token_metrics=True)
    assert fresh == warm
    assert fresh["supervised_tokens"] == 9
    assert [row["target_ids"] for row in warm["rows"]] == [
        row["input_ids"][row["loss_start"] :] for row in examples
    ]
