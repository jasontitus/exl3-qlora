"""Real SFT provenance validation and token audit, synthetic terminal children."""

import json
from pathlib import Path

import pytest
import test_sft_project as fixtures
from test_sft_project import _arg, _checkpoint, _evaluation_result, _write

project = fixtures.project
sft = fixtures.sft


def prepare(project, monkeypatch):
    selected = _checkpoint(project, 1)
    calls = []
    monkeypatch.setattr(project, "_paired_preview", lambda *a: {"status": "fixture"})

    def child(command, *, label, directory):
        calls.append(command)
        _write(Path(_arg(command, "--report")), _evaluation_result(project, command))

    monkeypatch.setattr(project, "_child", child)
    return selected, calls


def run(project, kind):
    return project.verify_training_fit() if kind == "fit" else project.evaluate()


@pytest.mark.parametrize("kind", ["fit", "compare"])
@pytest.mark.parametrize("field", ["adapter", "runtime", "resource", "targets"])
def test_rewrite_after_real_validation_stops_before_second_child(project, monkeypatch, kind, field):
    _, calls = prepare(project, monkeypatch)
    original = project._verify_evaluation_receipt

    def validate(value, *args):
        original(value, *args)
        path = Path(_arg(calls[-1], "--report"))
        changed = json.loads(path.read_bytes())
        if field == "adapter":
            changed["adapter"] = "/unvalidated/adapter"
        elif field == "runtime":
            changed["evaluation_context"]["implementation"] = {"source": "changed"}
        elif field == "resource":
            changed["evaluation_context"]["argv"].extend(["--process-memory-gib", "99"])
        else:
            changed["rows"][0]["target_ids"][0] = 9
        _write(path, changed)

    monkeypatch.setattr(project, "_verify_evaluation_receipt", validate)
    with pytest.raises(ValueError, match="Receipt changed"):
        run(project, kind)
    assert len(calls) == 1
    assert not (project.root / ("fit.json" if kind == "fit" else "comparison.json")).exists()


@pytest.mark.parametrize("kind", ["fit", "compare"])
def test_prior_report_drift_during_second_child_refuses_publication(project, monkeypatch, kind):
    _, calls = prepare(project, monkeypatch)
    original = project._child

    def child(*args, **kwargs):
        original(*args, **kwargs)
        if len(calls) == 2:
            path = Path(_arg(calls[0], "--report"))
            path.write_bytes(path.read_bytes() + b" ")

    monkeypatch.setattr(project, "_child", child)
    with pytest.raises(ValueError, match="Receipt changed"):
        run(project, kind)
    assert len(calls) == 2
    assert not (project.root / ("fit.json" if kind == "fit" else "comparison.json")).exists()


@pytest.mark.parametrize("kind", ["fit", "compare"])
def test_in_memory_validator_mutation_cannot_change_published_result(project, monkeypatch, kind):
    _, calls = prepare(project, monkeypatch)
    original = project._verify_evaluation_receipt

    def validate(value, *args):
        original(value, *args)
        value["adapter"] = "mutated callback result"

    monkeypatch.setattr(project, "_verify_evaluation_receipt", validate)
    with pytest.raises(ValueError, match="validator mutated"):
        run(project, kind)
    assert len(calls) == 1


def test_fit_final_parity_window_rechecks_both_reports(project, monkeypatch):
    _, calls = prepare(project, monkeypatch)
    original = project._final_fit_parity

    def parity(*args, **kwargs):
        value = original(*args, **kwargs)
        path = Path(_arg(calls[0], "--report"))
        path.write_bytes(path.read_bytes() + b" ")
        return value

    monkeypatch.setattr(project, "_final_fit_parity", parity)
    with pytest.raises(ValueError, match="Receipt changed"):
        project.verify_training_fit()
    assert not (project.root / "fit.json").exists()


def test_fit_historical_report_final_guard_and_original_hash(project, monkeypatch):
    selected, _ = prepare(project, monkeypatch)
    historical = selected.parent / "training.json"
    _write(historical, {"status": "completed"})
    initial = sft.digest(historical)
    good = project.verify_training_fit()
    assert good["receipt_inputs_sha256"]["training_report"] == initial
    previous = (project.root / "fit.json").read_bytes()
    original = project._final_fit_parity

    def parity(*args, **kwargs):
        value = original(*args, **kwargs)
        historical.write_bytes(historical.read_bytes() + b" ")
        return value

    monkeypatch.setattr(project, "_final_fit_parity", parity)
    with pytest.raises(ValueError, match="Receipt changed"):
        project.verify_training_fit()
    assert (project.root / "fit.json").read_bytes() == previous


def test_compare_final_preview_retains_existing_token_audit_guard(project, monkeypatch):
    _, calls = prepare(project, monkeypatch)
    good = project.evaluate()
    assert good["loss_delta"] == 0.0
    previous = (project.root / "comparison.json").read_bytes()

    def preview(*args):
        path = Path(_arg(calls[-2], "--report"))
        path.write_bytes(path.read_bytes() + b" ")
        return {"status": "fixture"}

    monkeypatch.setattr(project, "_paired_preview", preview)
    with pytest.raises(ValueError, match="Audit input changed"):
        project.evaluate()
    assert (project.root / "comparison.json").read_bytes() == previous


def test_historical_fit_report_replacement_during_real_probe_check(project, monkeypatch):
    from test_final_fit_probe import pair as pair_fixture

    # Use the real final-probe comparator and its synthetic CPU fixture.
    pair = pair_fixture.__wrapped__(project.root)
    checkpoint = pair["checkpoint"]
    historical = checkpoint.parent / "training.json"
    _write(historical, {"status": "completed", "training_fit_probe": pair["probe"]})
    project.args.final_fit_examples = pair["expected"]["requested_examples"]
    project.args.chunk_size = pair["expected"]["chunk_size"]
    original = sft.compare_final_fit_probe

    def validate(*args, **kwargs):
        result = original(*args, **kwargs)
        historical.write_bytes(historical.read_bytes() + b" ")
        return result

    monkeypatch.setattr(sft, "compare_final_fit_probe", validate)
    with pytest.raises(ValueError, match="Receipt changed"):
        project._final_fit_parity(
            checkpoint,
            pair["fresh"],
            pair["selected_rows"],
            pair["selected_rows"],
            pair["expected"],
        )
