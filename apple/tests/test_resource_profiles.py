"""Resource/configuration boundary tests with no model, Metal or GPU imports."""

import argparse
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from exl3_mlx.activation_config import ACTIVATION_MODES, validate_activation_mode
from exl3_mlx.resources import (
    RESOURCE_FIELDS,
    add_resource_arguments,
    resolve_resources,
    resources_from_args,
)


@pytest.mark.parametrize(
    "change",
    [
        {"memory_gib": float("nan")},
        {"memory_gib": float("inf")},
        {"memory_gib": True},
        {"memory_gib": 1025},
        {"memory_gib": 10**1000},
        {"reserve_gib": float("nan")},
        {"reserve_gib": -1},
        {"decoded_cache_mib": float("nan")},
        {"decoded_cache_mib": True},
        {"decoded_cache_mib": 1.5},
        {"decoded_cache_mib": -1},
        {"packed_cache_mib": float("inf")},
        {"allocator_cache_mib": 1025},
        {"allocator_cache_mib": False},
        {"decoded_cache_mib": 16384},
        {"prefix_cache_mib": -1},
        {"prefix_cache_mib": True},
        {"prefix_cache_mib": 2.5},
        {"prefix_cache_mib": "2048"},
        {"prefix_cache_mib": 16384},
    ],
)
def test_invalid_or_overcommitted_resource_configuration_fails(change):
    with pytest.raises(ValueError):
        resolve_resources(**change)


def test_development_cache_bounds_are_inside_process_budget():
    high = resolve_resources("development")
    assert high == dict(
        profile="development",
        memory_gib=80,
        reserve_gib=32,
        packed_cache_mib=8192,
        decoded_cache_mib=0,
        prefix_cache_mib=0,
        prefix_disk_cache=None,
        prefix_disk_cache_gib=0.0,
        allocator_cache_mib=256,
    )
    with pytest.raises(ValueError, match="Combined"):
        resolve_resources("development", memory_gib=32, decoded_cache_mib=32768)
    reduced = resolve_resources("development", memory_gib=32, decoded_cache_mib=8192)
    assert reduced["memory_gib"] == 32
    assert resolve_resources(default_memory_gib=32)["memory_gib"] == 32
    assert resolve_resources()["prefix_cache_mib"] == 0
    assert resolve_resources("development", prefix_cache_mib=2048)["prefix_cache_mib"] == 2048
    with pytest.raises(ValueError, match="Combined"):
        resolve_resources("development", prefix_cache_mib=65536)


def workflow_namespace():
    # Exercise the actual parser/config/model-forwarding functions, excluding
    # imports and computation that initialize the MLX workflow module.
    source = Path(__file__).parents[1] / "src/exl3_mlx/workflow.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {
            "common",
            "build_parser",
            "parse_args",
            "make_model",
            "model_settings",
        }
    ]
    namespace = dict(
        argparse=argparse,
        json=json,
        math=math,
        Path=Path,
        add_resource_arguments=add_resource_arguments,
        resources_from_args=resources_from_args,
        RESOURCE_FIELDS=RESOURCE_FIELDS,
        ACTIVATION_MODES=ACTIVATION_MODES,
        validate_activation_mode=validate_activation_mode,
        __doc__="CPU parser test",
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def test_workflow_profile_cli_config_override_and_legacy_alias(tmp_path):
    namespace = workflow_namespace()
    parse = namespace["parse_args"]
    low = parse(["doctor", "--model", "M"])
    assert (low.memory_gib, low.decoded_cache_mib, low.cache_mib) == (16, 0, 128)
    high = parse(["doctor", "--model", "M", "--profile", "development", "--cache-mib", "4096"])
    assert (high.memory_gib, high.cache_mib, high.decoded_cache_mib) == (80, 4096, 0)
    config = tmp_path / "run.json"
    config.write_text(
        json.dumps(dict(model="M", data="D", output="O", profile="development", cache_mib=2048))
    )
    args = parse(
        [
            "train",
            "--config",
            str(config),
            "--decoded-cache-mib",
            "4096",
            "--prefix-cache-mib",
            "2048",
        ]
    )
    assert (args.memory_gib, args.cache_mib, args.decoded_cache_mib) == (80, 2048, 4096)
    seen = {}

    def model(path, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(report=lambda: {"trainable_parameters": 1})

    namespace.update(DeepSeekTextModel=model, model_settings=lambda _: {})
    namespace["make_model"](args, lambda _: None)
    assert (seen["cache_mib"], seen["decoded_cache_mib"], seen["chunk_size"]) == (2048, 4096, 512)
    assert seen["prefix_cache_mib"] == 2048
    config.write_text(json.dumps(dict(model="M", data="D", output="O", **resolve_resources())))
    args = parse(
        [
            "train",
            "--config",
            str(config),
            "--profile",
            "development",
            "--decoded-cache-mib",
            "4096",
        ]
    )
    assert (args.memory_gib, args.cache_mib, args.decoded_cache_mib) == (80, 8192, 4096)


def test_ktune_profile_keeps_existing_low_memory_default():
    from exl3_mlx.ktune_cli import build_parser

    parser = build_parser()
    base = ["finetune", "--source", "S", "--model", "M", "--output", "O"]
    assert resources_from_args(parser.parse_args(base), default_memory_gib=32)["memory_gib"] == 32
    assert (
        resources_from_args(
            parser.parse_args(base + ["--profile", "development"]), default_memory_gib=32
        )["memory_gib"]
        == 80
    )


def test_actual_memory_guard_uses_112_gib_headroom_and_restores_limits(monkeypatch):
    source = Path(__file__).parents[1] / "src/exl3_mlx/memory.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    tree.body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.Import) and any(alias.name == "mlx.core" for alias in n.names))
    ]
    calls = []
    fake_mx = SimpleNamespace(
        set_memory_limit=lambda value: calls.append(("memory", value)) or 123,
        set_cache_limit=lambda value: calls.append(("cache", value)) or 456,
    )
    namespace = {
        "__name__": "exl3_mlx.memory_cpu_profile_test",
        "__package__": "exl3_mlx",
        "mx": fake_mx,
    }
    exec(compile(tree, str(source), "exec"), namespace)
    budget = namespace["MemoryBudget"](80, cache_mib=256, reserve_gib=32)
    monkeypatch.setattr(namespace["sys"], "platform", "darwin")
    monkeypatch.setattr(budget, "usage", lambda: 0)
    for free, passes in ((111, False), (112, True)):
        text = (
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            f"Pages free: {free * 2**30 // 16384}.\nPages inactive: 0.\nPages speculative: 0.\n"
        )
        monkeypatch.setattr(
            namespace["subprocess"], "run", lambda *a, **kw: SimpleNamespace(stdout=text)
        )
        if passes:
            with budget:
                pass
        else:
            with pytest.raises(MemoryError, match="budget plus reserve"):
                budget.check_startup_headroom()
            assert calls == []
    assert calls == [
        ("memory", 80 * 2**30),
        ("cache", 256 * 2**20),
        ("cache", 456),
        ("memory", 123),
    ]


def test_layer_progress_reports_current_cumulative_cache_phase_evidence():
    source = Path(__file__).parents[1] / "src/exl3_mlx/deepseek.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    model = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepSeekTextModel"
    )
    method = next(
        n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "_cache_progress"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    messages = []
    stats = SimpleNamespace(
        loads=20,
        hits=5,
        decoded_loads=100,
        decoded_hits=0,
        decoded_evictions=3,
        cached_bytes=2**30,
        decoded_cached_bytes=4 * 2**30,
    )
    fake = SimpleNamespace(_progress=messages.append, _store=SimpleNamespace(stats=stats))
    namespace["_cache_progress"](fake, "Forward layer 2/40")
    stats.decoded_hits = 100
    namespace["_cache_progress"](fake, "Layer 2/40: backward complete")
    assert "Forward layer 2/40" in messages[0]
    assert "decoded loads/hits/evictions 100/0/3" in messages[0]
    assert "backward complete" in messages[1]
    assert "decoded loads/hits/evictions 100/100/3" in messages[1]
    assert "retained packed/decoded 1.00/4.00 GiB" in messages[1]


@pytest.mark.parametrize(
    "settings",
    [
        {"prefix_disk_cache": "cache"},
        {"prefix_disk_cache_gib": 8},
        {"prefix_disk_cache": "", "prefix_disk_cache_gib": 8},
        {"prefix_disk_cache": 3, "prefix_disk_cache_gib": 8},
        {"prefix_disk_cache": "cache", "prefix_disk_cache_gib": float("nan")},
        {"prefix_disk_cache": "cache", "prefix_disk_cache_gib": True},
        {"prefix_disk_cache": "cache", "prefix_disk_cache_gib": -1},
        {"prefix_disk_cache": "cache", "prefix_disk_cache_gib": 1025},
    ],
)
def test_disk_prefix_cache_resource_pair_validation(settings):
    with pytest.raises(ValueError, match="required together"):
        resolve_resources(**settings)


def test_disk_prefix_cache_is_not_counted_as_resident_memory():
    values = resolve_resources(prefix_disk_cache="cache", prefix_disk_cache_gib=64)
    assert values["prefix_disk_cache"] == "cache"
    assert values["prefix_disk_cache_gib"] == 64
    assert values["memory_gib"] == 16


def test_explicit_budget_above_eighty_is_supported_with_runtime_headroom_guard():
    result = resolve_resources("development", memory_gib=90, reserve_gib=8)
    assert result["memory_gib"] == 90 and result["reserve_gib"] == 8
