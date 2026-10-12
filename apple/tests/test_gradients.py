import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from exl3_mlx import EXL3Linear, EXL3QLoRALinear
from exl3_mlx.reference import effective, random_packed
from mlx.utils import tree_flatten


def fixture(k=128, n=384, rate=3, codebook="mul1", chunk_size=128):
    rng = np.random.default_rng(17)
    p = random_packed(k // 16 * (n // 16), rate, rng).reshape(k // 16, n // 16, -1)
    su = rng.uniform(-0.3, 0.3, k).astype(np.float16)
    sv = rng.uniform(-0.8, 0.8, n).astype(np.float16)
    b = rng.normal(0, 0.01, n).astype(np.float32)
    base = EXL3Linear(
        mx.array(p),
        mx.array(su),
        mx.array(sv),
        codebook=codebook,
        bias=mx.array(b),
        chunk_size=chunk_size,
    )
    return base, effective(p, su, sv, codebook), b


@pytest.mark.parametrize("dtype,tol", [(mx.float32, 2e-5), (mx.float16, 2e-3), (mx.bfloat16, 2e-2)])
@pytest.mark.parametrize("rate,cb", [(3, "mul1"), (2.5, "mul1"), (4, "mcg"), (2, "3inst")])
def test_forward_vjp_against_float64(dtype, tol, rate, cb):
    base, w, bias = fixture(rate=rate, codebook=cb)
    rng = np.random.default_rng(6)
    x = mx.array(rng.normal(size=(2, 3, 128)), dtype=dtype)
    dy = mx.array(rng.normal(size=(2, 3, 384)), dtype=dtype)
    xnp, dynp = np.array(x.astype(mx.float32)), np.array(dy.astype(mx.float32))
    y = base(x)
    _, (dx,) = mx.vjp(base, [x], [dy])
    np.testing.assert_allclose(np.array(y.astype(mx.float32)), xnp @ w + bias, atol=tol, rtol=tol)
    np.testing.assert_allclose(np.array(dx.astype(mx.float32)), dynp @ w.T, atol=tol, rtol=tol)


def test_adapter_and_input_gradients_match_dense_and_finite_difference():
    mx.random.seed(123)
    base, w, bias = fixture()
    layer = EXL3QLoRALinear(base, rank=4, alpha=8)
    layer.lora_b = mx.random.normal(layer.lora_b.shape) * 0.1
    x = mx.random.normal((3, 128))
    target = mx.random.normal((3, 384))

    def loss(m, inp):
        return mx.mean((m(inp) - target) ** 2)

    _, grads = nn.value_and_grad(layer, loss)(layer, x)
    dx = mx.grad(lambda z: loss(layer, z))(x)
    wmx, bmx = mx.array(w, dtype=mx.float32), mx.array(bias)

    def dense(inp, a, b):
        y = inp @ wmx + bmx + (inp @ a) @ b * layer.scaling
        return mx.mean((y - target) ** 2)

    expected = mx.grad(dense, argnums=(0, 1, 2))(x, layer.lora_a, layer.lora_b)
    for got, want in zip((dx, grads["lora_a"], grads["lora_b"]), expected):
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=2e-4, atol=2e-6)
    eps = 0.002
    delta = mx.zeros_like(x).at[0, 7].add(eps)
    fd = (loss(layer, x + delta) - loss(layer, x - delta)) / (2 * eps)
    assert float(fd) == pytest.approx(float(dx[0, 7]), rel=0.02, abs=1e-4)
    for key, index in (("lora_a", (3, 1)), ("lora_b", (1, 9))):
        original = getattr(layer, key)
        delta = mx.zeros_like(original).at[index].add(eps)
        setattr(layer, key, original + delta)
        hi = loss(layer, x).item()
        setattr(layer, key, original - delta)
        lo = loss(layer, x).item()
        setattr(layer, key, original)
        assert (hi - lo) / (2 * eps) == pytest.approx(float(grads[key][index]), rel=0.02, abs=1e-4)


def test_training_freezes_base_and_decreases_loss():
    mx.random.seed(8)
    base, _, _ = fixture(n=128)
    layer = EXL3QLoRALinear(base, rank=4, alpha=4)
    frozen = {k: np.array(v) for k, v in tree_flatten(base.parameters())}
    assert set(dict(tree_flatten(layer.trainable_parameters()))) == {"lora_a", "lora_b"}
    x = mx.random.normal((8, 128))
    target = mx.stop_gradient(
        base(x) + (x @ mx.random.normal((128, 4))) @ mx.random.normal((4, 128)) * 0.01
    )

    def loss(m):
        return mx.mean((m(x) - target) ** 2)

    vg = nn.value_and_grad(layer, loss)
    opt = optim.Adam(learning_rate=0.02)
    initial, grad = vg(layer)
    assert set(dict(tree_flatten(grad))) == {"lora_a", "lora_b"}
    assert mx.all(grad["lora_a"] == 0).item()  # expected with zero B initialization
    assert mx.linalg.norm(grad["lora_b"]).item() > 0
    for _ in range(60):
        _, grad = vg(layer)
        opt.update(layer, grad)
        mx.eval(layer.parameters(), opt.state)
    assert loss(layer).item() < initial.item() * 0.1
    for key, value in tree_flatten(base.parameters()):
        np.testing.assert_array_equal(np.array(value), frozen[key])


def test_stacked_layers_and_checkpointing_propagate_dx():
    base, _, _ = fixture(n=128)
    first = EXL3QLoRALinear(base, rank=4)
    second = EXL3QLoRALinear(base, rank=4)
    x = mx.ones((2, 128))

    def loss(m):
        return mx.mean(mx.checkpoint(second)(mx.tanh(m(x))) ** 2)

    _, grads = nn.value_and_grad(first, loss)(first)
    assert mx.linalg.norm(grads["lora_b"]).item() > 0


def test_chunked_and_unchunked_vector_noncontiguous_empty():
    small, _, _ = fixture()
    wide, _, _ = fixture(chunk_size=512)
    x = mx.random.normal((128, 3)).T
    for inp in (x, x[0], mx.zeros((0, 128))):
        np.testing.assert_allclose(np.array(small(inp)), np.array(wide(inp)), atol=1e-5)


def test_rejects_unpadded_dimensions_and_bad_chunk():
    with pytest.raises(ValueError, match="multiples of 128"):
        EXL3Linear(mx.zeros((4, 8, 48), dtype=mx.int16), mx.ones(64), mx.ones(128))
    with pytest.raises(ValueError, match="chunk_size"):
        EXL3Linear(mx.zeros((8, 8, 48), dtype=mx.int16), mx.ones(128), mx.ones(128), chunk_size=16)
