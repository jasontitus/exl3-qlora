"""Long-context loss matches MLX-LM while bounding vocabulary projection rows."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.tuner.trainer import default_loss

from exl3_mlx.training import evaluate, masked_loss, projected_cross_entropy


@pytest.mark.parametrize("size", [1, 7, 32, 200])
def test_projected_loss_and_vjp_match_dense(size):
    mx.random.seed(19)
    h = mx.random.normal((69, 16))
    weight = mx.random.normal((83, 16))
    targets = mx.arange(69) % 83
    calls = []

    def project(x):
        calls.append(x.shape[0])
        return x @ weight.T

    def dense(x):
        return nn.losses.cross_entropy(x @ weight.T, targets, reduction="mean")

    value, grad = mx.value_and_grad(
        lambda x: projected_cross_entropy(project, x, targets, chunk_size=size)
    )(h)
    expected, expected_grad = mx.value_and_grad(dense)(h)
    np.testing.assert_allclose(np.array(value), np.array(expected), rtol=1e-6)
    np.testing.assert_allclose(np.array(grad), np.array(expected_grad), atol=2e-7)
    assert max(calls) <= size


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(101, 16)
        self.head = nn.Linear(16, 101, bias=False)
        self.head.freeze()
        self._head_rows = []

    def hidden_states(self, tokens):
        # Causal mixing makes gradients through unsupervised prompt tokens matter.
        return mx.cumsum(self.embedding(tokens), axis=1) / 10

    def project_logits(self, values):
        self._head_rows.append(values.size // values.shape[-1])
        return self.head(values)

    def __call__(self, tokens):
        return self.head(self.hidden_states(tokens))


def test_long_masked_loss_preserves_prompt_gradients_and_evaluation():
    mx.random.seed(21)
    model = Tiny()
    tokens = (mx.arange(1031)[None] % 101).astype(mx.int32)
    lengths = mx.array([[777, 1030]])
    value, grads = nn.value_and_grad(model, masked_loss)(model, tokens, lengths)
    expected, reference = nn.value_and_grad(
        model, lambda m, t, spans: default_loss(m, t, spans)[0]
    )(model, tokens, lengths)
    np.testing.assert_allclose(np.array(value), np.array(expected), rtol=2e-6)
    np.testing.assert_allclose(
        np.array(grads["embedding"]["weight"]),
        np.array(reference["embedding"]["weight"]),
        atol=2e-6,
    )
    assert not grads.get("head")
    report = evaluate(model, [{"input_ids": tokens.tolist()[0], "loss_start": 777}])
    assert report["supervised_tokens"] == 254
    assert report["logits_scope"] == "supervised_rows"
    assert max(model._head_rows) <= 32
    assert report["loss"] == pytest.approx(value.item(), rel=2e-6)
