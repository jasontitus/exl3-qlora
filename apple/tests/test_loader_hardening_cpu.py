"""CPU tests execute the actual loaders with NumPy safetensors instead of Metal.

No function body is rewritten: only its global MLX I/O/tree dependencies are
replaced. The full Metal resume tests remain the integration gate before merge.
"""

import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_activation_mode_config as activation_tests
from safetensors.numpy import load_file, save_file
from test_activation_mode_config import train_namespace

adapter_io = activation_tests.adapter_io


def write(path, value):
    Path(path).write_text(json.dumps(value))


def read(path):
    return json.loads(Path(path).read_text())


def flatten(tree, prefix=""):
    if isinstance(tree, dict):
        return [
            item
            for name, value in tree.items()
            for item in flatten(value, f"{prefix}.{name}" if prefix else name)
        ]
    return [(prefix, tree)]


def unflatten(entries):
    result = {}
    for key, value in entries:
        node = result
        names = key.split(".")
        for name in names[:-1]:
            node = node.setdefault(name, {})
        node[names[-1]] = value
    return result


class CPUAdam:
    """MLX-style init/state-set behavior, including in-place container mutation."""

    def __init__(self, learning_rate=3e-4):
        self.betas = [0.9, 0.999]
        self.eps = 1e-8
        self.bias_correction = False
        self._schedulers = {}
        self._state = {
            "step": np.array(7, dtype=np.uint64),
            "learning_rate": np.array(learning_rate, dtype=np.float32),
        }
        self._initialized = False
        self.init_calls = 0

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self._initialized = False

    def init(self, parameters):
        self.init_calls += 1
        for name, value in parameters.items():
            if name not in self.state:
                self.state[name] = {"m": np.zeros_like(value), "v": np.zeros_like(value)}
        self._initialized = True


@pytest.fixture
def checkpoint_io(tmp_path, monkeypatch):
    io = train_namespace()
    root = tmp_path / "base"
    for name in (
        "config.json",
        "expert-manifest-v2.json",
        "engram/engram-manifest.json",
        "engram-token-map.u32.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    parameters = {"a": np.ones((2, 2), np.float32)}
    model = SimpleNamespace(
        _root=root,
        adapter_config=dict(rank=4, alpha=8.0, adapter_layers=[39], expert_adapter_layers=[]),
        trainable_parameters=lambda: parameters,
        load_weights=lambda entries, strict: parameters.update(entries),
    )
    directory = tmp_path / "saved"
    io["save_adapter"](model, directory, {})
    saved = {
        "step": np.array(1, np.uint64),
        "learning_rate": np.array(1e-4, np.float32),
        "a.m": np.full((2, 2), 0.125, np.float32),
        "a.v": np.full((2, 2), 0.25, np.float32),
    }
    save_file(saved, directory / "optimizer.safetensors")
    rng = {"0": np.array([42, 24], np.uint32)}
    save_file(rng, directory / "random.safetensors")
    contract = {"learning_rate": 1e-4, "grad_accumulation": 2}
    state = dict(format="exl3-mlx-training-state-v1", contract=contract, step=1, cursor=2)

    def rehash():
        state["files"] = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in (
                "adapter.safetensors",
                "adapter_config.json",
                "optimizer.safetensors",
                "random.safetensors",
            )
        }
        write(directory / "trainer_state.json", state)

    rehash()
    source = Path(__file__).parents[1] / "src/exl3_mlx/training.py"
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "load_checkpoint"
    )
    fake_mx = SimpleNamespace(
        load=load_file,
        eval=lambda *values: None,
        all=np.all,
        isfinite=np.isfinite,
        array=np.array,
        random=SimpleNamespace(state=[np.array([9, 8], np.uint32)]),
    )
    monkeypatch.setitem(
        sys.modules, "exl3_mlx.train", SimpleNamespace(load_adapter=io["load_adapter"])
    )
    namespace = dict(
        __package__="exl3_mlx",
        Path=Path,
        json=json,
        math=math,
        mx=fake_mx,
        optim=SimpleNamespace(Adam=CPUAdam),
        sha256=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        tree_flatten=flatten,
        tree_unflatten=unflatten,
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    parameters["a"] = np.full((2, 2), -5, np.float32)
    optimizer = CPUAdam()
    return SimpleNamespace(
        io=io,
        model=model,
        parameters=parameters,
        directory=directory,
        saved=saved,
        rng=rng,
        state=state,
        contract=contract,
        rehash=rehash,
        mx=fake_mx,
        optimizer=optimizer,
        load=lambda: namespace["load_checkpoint"](model, optimizer, directory, contract),
    )


def snapshot(fixture):
    return (
        id(fixture.optimizer.state),
        fixture.optimizer._initialized,
        fixture.optimizer.init_calls,
        [(name, id(value), value.tobytes()) for name, value in flatten(fixture.parameters)],
        [(name, id(value), value.tobytes()) for name, value in flatten(fixture.optimizer.state)],
        [(id(value), value.tobytes()) for value in fixture.mx.random.state],
    )


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "nan_m",
        "inf_m",
        "nan_v",
        "inf_v",
        "negative_v",
        "nan_lr",
        "inf_lr",
        "zero_lr",
        "negative_lr",
        "different_lr",
        "optimizer_shape",
        "optimizer_dtype",
        "optimizer_keys",
        "rng_shape",
        "rng_dtype",
        "rng_keys",
        "step_mismatch",
        "cursor_mismatch",
        "adapter_nan",
        "adapter_semantics",
    ],
)
def test_rejected_payload_preserves_model_optimizer_rng(checkpoint_io, initialized, kind):
    f = checkpoint_io
    if initialized:
        f.optimizer.init(f.parameters)
        f.optimizer.state["a"]["m"][:] = 7
    if kind in {"nan_m", "inf_m", "nan_v", "inf_v", "negative_v"}:
        key = "a." + kind[-1]
        f.saved[key][0, 0] = {"nan": np.nan, "inf": np.inf, "negative": -1}[kind.split("_")[0]]
    elif kind.endswith("lr"):
        f.saved["learning_rate"] = np.array(
            {
                "nan_lr": np.nan,
                "inf_lr": np.inf,
                "zero_lr": 0,
                "negative_lr": -1,
                "different_lr": 2e-4,
            }[kind],
            np.float32,
        )
    elif kind == "optimizer_shape":
        f.saved["a.m"] = np.ones((1, 4), np.float32)
    elif kind == "optimizer_dtype":
        f.saved["a.m"] = f.saved["a.m"].astype(np.float64)
    elif kind == "optimizer_keys":
        del f.saved["a.m"]
    elif kind.startswith("rng"):
        f.rng = {
            "rng_shape": {"0": np.ones(3, np.uint32)},
            "rng_dtype": {"0": np.ones(2, np.int32)},
            "rng_keys": {"wrong": np.ones(2, np.uint32)},
        }[kind]
    elif kind == "step_mismatch":
        f.saved["step"] = np.array(2, np.uint64)
    elif kind == "cursor_mismatch":
        f.state["cursor"] = 3
    elif kind == "adapter_nan":
        save_file({"a": np.full((2, 2), np.nan, np.float32)}, f.directory / "adapter.safetensors")
        config = read(f.directory / "adapter_config.json")
        config["adapter_sha256"] = hashlib.sha256(
            (f.directory / "adapter.safetensors").read_bytes()
        ).hexdigest()
        write(f.directory / "adapter_config.json", config)
    elif kind == "adapter_semantics":
        config = read(f.directory / "adapter_config.json")
        config["engine_semantics"] = "older-engine"
        write(f.directory / "adapter_config.json", config)
    save_file(f.saved, f.directory / "optimizer.safetensors")
    save_file(f.rng, f.directory / "random.safetensors")
    f.rehash()
    before = snapshot(f)
    with pytest.raises(ValueError):
        f.load()
    assert snapshot(f) == before


@pytest.mark.parametrize("field", ["step", "cursor"])
@pytest.mark.parametrize("bad", [True, -1, 1.5, "1", None])
def test_illegal_counters_reject_before_destination_initialization(checkpoint_io, field, bad):
    f = checkpoint_io
    f.state[field] = bad
    f.rehash()
    before = snapshot(f)
    with pytest.raises(ValueError, match="integers"):
        f.load()
    assert snapshot(f) == before


@pytest.mark.parametrize("legacy_contract", [False, True])
def test_valid_current_and_legacy_checkpoint_loads_without_destination_init(
    checkpoint_io, legacy_contract
):
    f = checkpoint_io
    if legacy_contract:
        f.contract.clear()
        f.state["cursor"] = 7  # Older receipts lack an accumulation relation.
    f.saved["a.m"][:] = 1e10
    f.saved["a.v"][:] = 1e30  # Large finite values are legal; no arbitrary threshold.
    save_file(f.saved, f.directory / "optimizer.safetensors")
    f.rehash()
    loaded = f.load()
    assert loaded == f.state
    assert f.optimizer.init_calls == 0
    np.testing.assert_array_equal(f.parameters["a"], np.ones((2, 2), np.float32))
    for name, value in flatten(f.optimizer.state):
        np.testing.assert_array_equal(value, f.saved[name])
    np.testing.assert_array_equal(f.mx.random.state[0], f.rng["0"])


@pytest.mark.parametrize(
    "kind",
    [
        "missing_m",
        "missing_v",
        "moment_shape",
        "moment_dtype",
        "extra_moment",
        "extra_parameter",
        "fractional_step",
        "step_signed",
        "step_shape",
        "learning_rate_dtype",
        "learning_rate_shape",
    ],
)
def test_destination_state_cannot_define_malformed_incoming_schema(checkpoint_io, kind):
    f = checkpoint_io
    f.optimizer.init(f.parameters)
    if kind in {"missing_m", "missing_v"}:
        moment = kind[-1]
        del f.optimizer.state["a"][moment]
        del f.saved[f"a.{moment}"]
    elif kind == "moment_shape":
        f.optimizer.state["a"]["m"] = np.zeros((1, 4), np.float32)
        f.saved["a.m"] = np.zeros((1, 4), np.float32)
    elif kind == "moment_dtype":
        f.optimizer.state["a"]["m"] = np.zeros((2, 2), np.float64)
        f.saved["a.m"] = np.zeros((2, 2), np.float64)
    elif kind == "extra_moment":
        f.optimizer.state["a"]["foreign"] = np.zeros((2, 2), np.float32)
        f.saved["a.foreign"] = np.zeros((2, 2), np.float32)
    elif kind == "extra_parameter":
        f.optimizer.state["foreign"] = {"m": np.zeros((2, 2), np.float32)}
        f.saved["foreign.m"] = np.zeros((2, 2), np.float32)
    else:
        key = "step" if kind.startswith("step") or kind == "fractional_step" else "learning_rate"
        invalid = {
            "fractional_step": np.array(1.5, np.float32),
            "step_signed": np.array(1, np.int64),
            "step_shape": np.array([1], np.uint64),
            "learning_rate_dtype": np.array(1e-4, np.float64),
            "learning_rate_shape": np.array([1e-4], np.float32),
        }[kind]
        f.optimizer.state[key] = invalid.copy()
        f.saved[key] = invalid.copy()
    save_file(f.saved, f.directory / "optimizer.safetensors")
    f.rehash()
    before = snapshot(f)
    with pytest.raises(ValueError, match="Optimizer state structure"):
        f.load()
    assert snapshot(f) == before


def test_valid_payload_replaces_malformed_destination_state(checkpoint_io):
    f = checkpoint_io
    f.optimizer.init(f.parameters)
    f.optimizer.state["a"] = {"m": np.zeros((4, 1), np.float64)}
    f.optimizer.state["foreign"] = {"v": np.zeros((3,), np.float32)}
    f.optimizer.state["step"] = np.array(1.5, np.float32)
    f.optimizer.state["learning_rate"] = np.array([3e-4], np.float64)
    loaded = f.load()
    assert loaded == f.state
    assert f.optimizer.init_calls == 1  # Only the test's initialization touched it.
    assert dict(flatten(f.optimizer.state)).keys() == f.saved.keys()
    for name, value in flatten(f.optimizer.state):
        np.testing.assert_array_equal(value, f.saved[name])
        assert value.dtype == f.saved[name].dtype


@pytest.mark.parametrize("kind", ["optimizer_type", "betas", "eps", "bias_correction", "scheduler"])
def test_unsupported_optimizer_algorithm_rejects_without_mutation(checkpoint_io, kind):
    f = checkpoint_io
    if kind == "optimizer_type":

        class OtherOptimizer(CPUAdam):
            pass

        f.optimizer.__class__ = OtherOptimizer
    elif kind == "scheduler":
        f.optimizer._schedulers = {"learning_rate": lambda step: 0.5}
    else:
        setattr(
            f.optimizer, kind, {"betas": [0.8, 0.999], "eps": 1e-6, "bias_correction": True}[kind]
        )
    before = snapshot(f)
    with pytest.raises(ValueError, match="fixed-rate Adam"):
        f.load()
    assert snapshot(f) == before


@pytest.mark.parametrize("semantics", ["cpu-fixture", "missing"])
def test_current_and_unversioned_legacy_adapters_remain_loadable(adapter_io, semantics):
    io, model, arrays, directory = adapter_io
    io["save_adapter"](model, directory, {})
    config = read(directory / "adapter_config.json")
    if semantics == "missing":
        config.pop("engine_semantics")
        config.pop("activation_mode")
        write(directory / "adapter_config.json", config)
    io["load_adapter_config"](directory)
    arrays["a"] = np.zeros_like(arrays["a"])
    io["load_adapter"](model, directory)
    np.testing.assert_array_equal(arrays["a"], np.ones((2, 2), np.float32))
    assert ("engine_semantics" in read(directory / "adapter_config.json")) == (
        semantics != "missing"
    )


@pytest.mark.parametrize("semantics", ["older-engine", None, 42, True])
def test_explicit_semantic_mismatch_rejects_before_weights_change(adapter_io, semantics):
    io, model, arrays, directory = adapter_io
    io["save_adapter"](model, directory, {})
    config = read(directory / "adapter_config.json")
    config["engine_semantics"] = semantics
    write(directory / "adapter_config.json", config)
    before = arrays["a"]
    with pytest.raises(ValueError, match="engine_semantics"):
        io["load_adapter_config"](directory)
    with pytest.raises(ValueError, match="engine_semantics"):
        io["load_adapter"](model, directory)
    assert arrays["a"] is before
