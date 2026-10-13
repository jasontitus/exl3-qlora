"""CPU-only fail-closed checks for fresh paired evaluation receipts."""

from pathlib import Path

import pytest
import test_sft_project as project_tests
from test_sft_project import _arg, _checkpoint, _evaluation_result, _write

project = project_tests.project


def operation(project, kind):
    return project.verify_training_fit if kind == "fit" else project.evaluate


@pytest.mark.parametrize("kind", ["fit", "heldout"])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_context",
        "input_hash",
        "selected_hash",
        "implementation",
        "argv",
        "mode",
        "adapter",
        "model",
        "examples_bool",
        "token_count",
        "nonfinite_loss",
    ],
)
def test_wrong_new_child_receipt_is_not_published(project, monkeypatch, kind, mutation):
    _checkpoint(project, 1)
    monkeypatch.setattr(project, "_paired_preview", lambda *args: {"status": "fixture"})

    def child(command, *, label, directory):
        value = _evaluation_result(project, command)
        context = value["evaluation_context"]
        if mutation == "missing_context":
            value.pop("evaluation_context")
        elif mutation == "input_hash":
            context["input_identity"]["data_sha256"] = "different"
        elif mutation == "selected_hash":
            context["selected_examples_sha256"] = "different"
        elif mutation == "implementation":
            context["implementation"] = {"source": "other code"}
        elif mutation == "argv":
            context["argv"].extend(["--chunk-size", "8192"])
        elif mutation == "mode":
            value["activation_mode"] = "bf16"
        elif mutation == "adapter":
            value["adapter"] = str(project.root / "wrong-adapter")
        elif mutation == "model":
            value["checkpoint"] = str(project.root / "wrong-model")
        elif mutation == "examples_bool":
            value["examples"] = True
        elif mutation == "token_count":
            value["supervised_tokens"] += 1
        elif mutation == "nonfinite_loss":
            value["loss"] = float("nan")
        _write(Path(_arg(command, "--report")), value)

    monkeypatch.setattr(project, "_child", child)
    with pytest.raises(ValueError, match="Child"):
        operation(project, kind)()
    assert not (project.root / ("fit.json" if kind == "fit" else "comparison.json")).exists()


@pytest.mark.parametrize("kind", ["fit", "heldout"])
@pytest.mark.parametrize("after_child", [1, 2])
def test_same_size_external_data_edit_during_pair_fails_closed(
    project, monkeypatch, kind, after_child
):
    _checkpoint(project, 1)
    monkeypatch.setattr(project, "_paired_preview", lambda *args: {"status": "fixture"})
    calls = []

    def child(command, *, label, directory):
        calls.append(command)
        _write(Path(_arg(command, "--report")), _evaluation_result(project, command))
        if len(calls) == after_child:
            path = Path(_arg(command, "--data"))
            path.write_text(path.read_text().replace("[0, 2,", "[0, 9,"))

    monkeypatch.setattr(project, "_child", child)
    with pytest.raises(ValueError, match="dataset changed"):
        operation(project, kind)()
    assert len(calls) == after_child
    assert not (project.root / ("fit.json" if kind == "fit" else "comparison.json")).exists()


@pytest.mark.parametrize("kind", ["fit", "heldout"])
def test_different_loaded_context_rejected_even_if_data_restored(project, monkeypatch, kind):
    _checkpoint(project, 1)
    monkeypatch.setattr(project, "_paired_preview", lambda *args: {"status": "fixture"})

    def child(command, *, label, directory):
        path = Path(_arg(command, "--data"))
        original = path.read_bytes()
        try:
            path.write_bytes(original.replace(b"[0, 2,", b"[0, 9,"))
            value = _evaluation_result(project, command)
        finally:
            path.write_bytes(original)
        _write(Path(_arg(command, "--report")), value)

    monkeypatch.setattr(project, "_child", child)
    with pytest.raises(ValueError, match="provenance differs"):
        operation(project, kind)()
    assert not (project.root / ("fit.json" if kind == "fit" else "comparison.json")).exists()


def test_project_is_rechecked_after_preview_before_publication(project, monkeypatch):
    _checkpoint(project, 1)

    def child(command, *, label, directory):
        _write(Path(_arg(command, "--report")), _evaluation_result(project, command))

    def preview(*args):
        path = project.root / "data/raw-test.jsonl"
        path.write_text(path.read_text() + "\n")
        return {"status": "fixture"}

    monkeypatch.setattr(project, "_child", child)
    monkeypatch.setattr(project, "_paired_preview", preview)
    with pytest.raises(ValueError, match="dataset changed"):
        project.evaluate()
    assert not (project.root / "comparison.json").exists()
