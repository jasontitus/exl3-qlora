"""CPU-only observer behavior; no MLX import or probe execution."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/diagnose_final_block_memory.py"
spec = importlib.util.spec_from_file_location("memory_phase_diagnostic", SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_sampling_does_not_copy_change_or_retain_inputs_and_restores_original():
    seen, calls = [], []
    value = object()

    def original(argument, *, tag):
        calls.append(tag)
        assert argument is value
        return argument

    owner = SimpleNamespace(call=original)

    def classify(args, kwargs):
        return ("forward" if kwargs["tag"] < 3 else "backward", {})

    with diagnostic.instrument(
        [(owner, "call", classify, 2)], lambda phase, **kw: seen.append((phase, kw))
    ) as counts:
        for tag in range(6):
            assert owner.call(value, tag=tag) is value
        assert counts == {"forward": 3, "backward": 3}
    assert owner.call is original
    assert calls == list(range(6))
    assert [phase for phase, _ in seen] == ["forward.enter", "forward.exit"] * 2 + [
        "backward.enter",
        "backward.exit",
    ] * 2
    assert all(set(metadata) == {"call"} for _, metadata in seen)


def test_failure_is_reported_and_wrappers_restore():
    events = []

    def fail():
        raise RuntimeError("intentional")

    owner = SimpleNamespace(call=fail)
    with pytest.raises(RuntimeError, match="intentional"):
        with diagnostic.instrument(
            [(owner, "call", lambda a, k: ("test", {}), 1)],
            lambda phase, **kw: events.append((phase, kw)),
        ):
            owner.call()
    assert owner.call is fail
    assert [p for p, _ in events] == ["test.enter", "test.error"]
    assert events[-1][1]["error"] == "RuntimeError"
