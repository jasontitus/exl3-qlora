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


def test_token_metrics_match_dense_reference_and_preserve_aggregate():
    mx.random.seed(22)
    model = Tiny()
    examples = [
        {"input_ids": (np.arange(71) % 101).tolist(), "loss_start": 17},
        {"input_ids": [0, 4, 12, 20], "loss_start": 2},
    ]
    plain = evaluate(model, examples)
    detailed = evaluate(model, examples, include_token_metrics=True)
    assert {k: v for k, v in detailed.items() if k != "rows"} == plain
    for example, row in zip(examples, detailed["rows"]):
        tokens = mx.array([example["input_ids"]])
        start = example["loss_start"]
        logits = model(tokens[:, :-1])[0, start - 1 :]
        gold = tokens[0, start:]
        np.testing.assert_allclose(
            row["token_nll"], nn.losses.cross_entropy(logits, gold).tolist(), atol=1e-6
        )
        assert row["target_ids"] == gold.tolist()
        assert row["predicted_ids"] == mx.argmax(logits, axis=-1).tolist()
        assert row["supervised_tokens"] == len(example["input_ids"]) - start
        single = evaluate(model, [example], include_token_metrics=True)
        assert row["logits_sha256"] == single["logits_sha256"]
        assert row["logits_sha256"] == single["rows"][0]["logits_sha256"]


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("preserve_mode", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_evaluate_preserves_requested_mode_and_restores_on_error(training, preserve_mode, fail):
    class Observed(Tiny):
        def __init__(self):
            super().__init__()
            self._seen_modes = []

        def project_logits(self, values):
            self._seen_modes.append(self.training)
            return super().project_logits(values)

    mx.random.seed(31)
    model = Observed()
    model.train(training)
    checks = 0

    def check():
        nonlocal checks
        checks += 1
        if fail and checks == 3:
            raise MemoryError("diagnostic budget check")

    kwargs = dict(check=check, include_token_metrics=True, preserve_mode=preserve_mode)
    examples = [{"input_ids": [0, 4, 12, 20], "loss_start": 2}]
    if fail:
        with pytest.raises(MemoryError, match="diagnostic budget"):
            evaluate(model, examples, **kwargs)
    else:
        report = evaluate(model, examples, **kwargs)
        assert report["supervised_tokens"] == 2
    assert model._seen_modes == [training if preserve_mode else False]
    assert model.training == training


def test_evaluate_rejects_nonboolean_preserve_mode():
    with pytest.raises(ValueError, match="preserve_mode"):
        evaluate(Tiny(), [], preserve_mode="yes")
