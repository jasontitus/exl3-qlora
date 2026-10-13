"""Actual MLX disk-tier parity. Run only in the parent's exclusive GPU test slot."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from test_prefix_cache_model import assert_bits_equal, make_model

from exl3_mlx.prefix_cache import FrozenPrefixCache
from exl3_mlx.prefix_disk import FrozenPrefixDisk
from exl3_mlx.training import evaluate, optimizer_step


def attach(model, directory, ram_bytes=0):
    store = FrozenPrefixDisk(
        directory,
        quota_bytes=1 << 20,
        binding={"test": "fixed-three-block-fixture", "frozen_blocks": 2},
    )
    model._prefix_cache = FrozenPrefixCache(ram_bytes, frozen_blocks=2, disk=store)
    model._prefix_signature = model._frozen_prefix_signature()
    return store


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_disk_roundtrip_actual_hidden_gradient_optimizer_and_mode_separation(tmp_path, dtype):
    plain = make_model(cache_bytes=0, dtype=dtype)
    writer = make_model(cache_bytes=0, dtype=dtype)
    first = attach(writer, tmp_path)
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    plain.train()
    writer.train()
    expected = plain.hidden_states(tokens)
    produced = writer.hidden_states(tokens)
    mx.eval(expected, produced)
    assert_bits_equal(expected, produced)
    assert first.report()["writes"] == 1
    exact_key = FrozenPrefixCache.key(tokens, training=True, checkpoint_blocks=True)
    saved = first.get(exact_key)
    # Duplicate writers must compare exact evaluated bytes, including BF16.
    first.put(exact_key, saved)
    changed = (saved[0], saved[1], (*saved[2][:3], saved[2][3] + mx.array(1, dtype=mx.uint8)))
    mx.eval(changed)
    with pytest.raises(ValueError, match="Different computed prefix bytes"):
        first.put(exact_key, changed)
    assert first.report()["writes"] == 1
    # A fresh model instance must load the evaluated BF16/F32 state from disk.
    reader = make_model(cache_bytes=0, dtype=dtype)
    second = attach(reader, tmp_path)
    reader.train()
    actual = reader.hidden_states(tokens)
    mx.eval(actual)
    assert_bits_equal(expected, actual)
    assert second.report()["hits"] == 1
    assert reader.blocks[0]._calls == reader.blocks[1]._calls == 0

    def loss(model):
        return mx.mean(model.hidden_states(tokens).astype(mx.float32) ** 2)

    plain_opt, reader_opt = optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)
    for _ in range(2):
        p_value, p_grads = nn.value_and_grad(plain, loss)(plain)
        r_value, r_grads = nn.value_and_grad(reader, loss)(reader)
        mx.eval(p_value, p_grads, r_value, r_grads)
        assert_bits_equal((p_value, p_grads), (r_value, r_grads))
        plain_opt.update(plain, p_grads)
        reader_opt.update(reader, r_grads)
        mx.eval(plain.parameters(), reader.parameters(), plain_opt.state, reader_opt.state)
        assert_bits_equal(plain.trainable_parameters(), reader.trainable_parameters())
        assert_bits_equal(plain_opt.state, reader_opt.state)
    reader.eval()
    plain.eval()
    assert_bits_equal(plain.hidden_states(tokens), reader.hidden_states(tokens))
    assert second.report()["writes"] == 1  # independent eval-mode entry
    assert second.report()["misses"] == 1


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_paired_evaluation_reuses_frozen_disk_entries_with_changed_suffix(tmp_path, dtype):
    plain = make_model(cache_bytes=0, dtype=dtype)
    cached = make_model(cache_bytes=0, dtype=dtype)
    # Small fixture's projection is deterministic and common to both models.
    for model in (plain, cached):
        model.project_logits = lambda hidden: hidden.astype(mx.float32)
    store = attach(cached, tmp_path)
    rows = [
        {"input_ids": [1, 2, 3, 4], "loss_start": 2},
        {"input_ids": [2, 3, 4, 5, 6], "loss_start": 2},
        {"input_ids": [3, 4, 5, 6, 7, 1], "loss_start": 3},
    ]
    assert evaluate(plain, rows, include_token_metrics=True) == evaluate(
        cached, rows, include_token_metrics=True
    )
    assert store.report()["writes"] == 3
    p_opt, c_opt = optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)
    for row in rows:
        assert optimizer_step(plain, p_opt, [row]) == optimizer_step(cached, c_opt, [row])
    assert_bits_equal(plain.trainable_parameters(), cached.trainable_parameters())
    before = store.report()
    assert evaluate(plain, rows, include_token_metrics=True) == evaluate(
        cached, rows, include_token_metrics=True
    )
    after = store.report()
    assert after["hits"] - before["hits"] == 3
    assert after["writes"] == before["writes"]
