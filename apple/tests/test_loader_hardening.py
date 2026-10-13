"""Real MLX integration gates; run only in a coordinated GPU test window."""

import hashlib
import json

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest
import test_workflow as workflow_tests
from mlx.utils import tree_flatten, tree_unflatten

from exl3_mlx.training import evaluate, load_checkpoint, optimizer_step, save_checkpoint

base = workflow_tests.base
Toy = workflow_tests.Toy
examples = workflow_tests.examples


def snapshot(model, optimizer):
    mx.eval(model.trainable_parameters(), optimizer.state, mx.random.state)
    return (
        id(optimizer.state),
        optimizer._initialized,
        [
            (name, id(value), np.array(value).tobytes())
            for name, value in tree_flatten(model.trainable_parameters())
        ],
        [
            (name, id(value), np.array(value).tobytes())
            for name, value in tree_flatten(optimizer.state)
        ],
        [(id(value), np.array(value).tobytes()) for value in mx.random.state],
    )


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("kind", ["nan_m", "negative_v", "rng_shape", "adapter_nan"])
def test_real_rejected_checkpoint_does_not_mutate_destination(base, tmp_path, initialized, kind):
    source, destination = Toy(base), Toy(base)
    source_opt, destination_opt = optim.Adam(1e-3), optim.Adam(1e-3)
    rows = examples()
    optimizer_step(source, source_opt, rows)
    if initialized:
        optimizer_step(destination, destination_opt, rows)
    contract = {"learning_rate": 1e-3, "grad_accumulation": 2}
    directory = tmp_path / "saved"
    save_checkpoint(source, source_opt, directory, dict(step=1, cursor=2, contract=contract))
    if kind in {"nan_m", "negative_v"}:
        filename = "optimizer.safetensors"
        arrays = mx.load(str(directory / filename))
        key = next(name for name in arrays if name.endswith("." + kind[-1]))
        arrays[key] = mx.full_like(arrays[key], float("nan") if kind == "nan_m" else -1)
    elif kind == "rng_shape":
        filename = "random.safetensors"
        arrays = mx.load(str(directory / filename))
        arrays["0"] = mx.zeros((3,), dtype=arrays["0"].dtype)
    else:
        filename = "adapter.safetensors"
        arrays = mx.load(str(directory / filename))
        key = next(iter(arrays))
        arrays[key] = mx.full_like(arrays[key], float("nan"))
    mx.save_safetensors(str(directory / filename), arrays)
    if kind == "adapter_nan":
        config_path = directory / "adapter_config.json"
        config = json.loads(config_path.read_text())
        config["adapter_sha256"] = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        config_path.write_text(json.dumps(config))
    state_path = directory / "trainer_state.json"
    state = json.loads(state_path.read_text())
    state["files"] = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in state["files"]
    }
    state_path.write_text(json.dumps(state))
    before = snapshot(destination, destination_opt)
    with pytest.raises(ValueError):
        load_checkpoint(destination, destination_opt, directory, contract)
    assert snapshot(destination, destination_opt) == before


@pytest.mark.parametrize("initialized", [False, True, "malformed"])
def test_real_valid_resume_with_strict_contract_is_bitwise_exact(base, tmp_path, initialized):
    source, destination = Toy(base), Toy(base)
    source_opt, destination_opt = optim.Adam(1e-3), optim.Adam(1e-3)
    rows = examples()
    optimizer_step(source, source_opt, rows)
    if initialized:
        optimizer_step(destination, destination_opt, rows)
        optimizer_step(destination, destination_opt, rows)
    if initialized == "malformed":
        destination_opt.state = {
            "step": mx.array(1.5),
            "learning_rate": mx.array([3e-4]),
            "foreign": {"m": mx.zeros((1,))},
        }
    contract = {"learning_rate": 1e-3, "grad_accumulation": 2}
    directory = tmp_path / "saved"
    save_checkpoint(source, source_opt, directory, dict(step=1, cursor=2, contract=contract))
    expected_random = mx.random.uniform(shape=(9,))
    mx.eval(expected_random)
    expected_loss = optimizer_step(source, source_opt, rows)
    load_checkpoint(destination, destination_opt, directory, contract)
    assert mx.array_equal(expected_random, mx.random.uniform(shape=(9,))).item()
    observed_loss = optimizer_step(destination, destination_opt, rows)
    assert expected_loss == observed_loss
    for expected, observed in (
        (source.trainable_parameters(), destination.trainable_parameters()),
        (source_opt.state, destination_opt.state),
    ):
        expected, observed = dict(tree_flatten(expected)), dict(tree_flatten(observed))
        assert expected.keys() == observed.keys()
        for key in expected:
            assert mx.array_equal(expected[key], observed[key]).item(), key
    assert evaluate(source, rows) == evaluate(destination, rows)


@pytest.mark.parametrize(
    "kind", ["missing_m", "moment_shape", "moment_dtype", "fractional_step", "extra_parameter"]
)
def test_real_malformed_destination_does_not_authorize_matching_bad_payload(base, tmp_path, kind):
    source, destination = Toy(base), Toy(base)
    source_opt, destination_opt = optim.Adam(1e-3), optim.Adam(1e-3)
    optimizer_step(source, source_opt, examples())
    destination_opt.init(destination.trainable_parameters())
    directory = tmp_path / "saved"
    contract = {"learning_rate": 1e-3, "grad_accumulation": 2}
    save_checkpoint(source, source_opt, directory, dict(step=1, cursor=2, contract=contract))
    arrays = mx.load(str(directory / "optimizer.safetensors"))
    existing = dict(tree_flatten(destination_opt.state))
    key = next(name for name in arrays if name.endswith(".m"))
    if kind == "missing_m":
        del arrays[key], existing[key]
    else:
        if kind == "moment_shape":
            value = mx.zeros((1, arrays[key].size), dtype=arrays[key].dtype)
        elif kind == "moment_dtype":
            value = arrays[key].astype(mx.float16)
        elif kind == "fractional_step":
            key, value = "step", mx.array(1.5)
        else:
            key, value = "foreign.m", mx.zeros((1,))
        arrays[key] = existing[key] = value
    destination_opt.state = tree_unflatten(list(existing.items()))
    destination_opt._initialized = True
    mx.save_safetensors(str(directory / "optimizer.safetensors"), arrays)
    state_path = directory / "trainer_state.json"
    state = json.loads(state_path.read_text())
    state["files"]["optimizer.safetensors"] = hashlib.sha256(
        (directory / "optimizer.safetensors").read_bytes()
    ).hexdigest()
    state_path.write_text(json.dumps(state))
    before = snapshot(destination, destination_opt)
    with pytest.raises(ValueError, match="Optimizer state structure"):
        load_checkpoint(destination, destination_opt, directory, contract)
    assert snapshot(destination, destination_opt) == before
