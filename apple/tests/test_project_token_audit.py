"""Normal compare exercises the CPU auditor, with no model/network/GPU."""

import json
from pathlib import Path

import pytest
import test_sft_project as fixtures
from test_sft_project import _arg, _checkpoint, _evaluation_result, _write

sft = fixtures.sft

project = fixtures.project


def setup_pair(project, monkeypatch, mutation=None):
    _checkpoint(project, 1)
    monkeypatch.setattr(project, "_paired_preview", lambda *args: {"status": "fixture"})

    def child(command, *, label, directory):
        value = _evaluation_result(project, command, loss=1.0 if label == "base" else 0.75)
        if mutation and label == "adapter":
            mutation(value)
        _write(Path(_arg(command, "--report")), value)

    monkeypatch.setattr(project, "_child", child)


@pytest.mark.parametrize("count,selected", [(1, 1), (37, 19), (129, 129)])
def test_normal_compare_arbitrary_sizes_and_exact_subset(project, monkeypatch, count, selected):
    rows = [
        {"input_ids": [0, 2, 3] + [6] * (1 + i % 3) + [1], "loss_start": 3} for i in range(count)
    ]
    path = project.root / "data/test.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads((project.root / "project.json").read_text())
    manifest["data_files"][path.name] = sft.digest(path)
    manifest["data"]["counts"]["test"] = count
    manifest["data"]["seq_length"] = 6
    manifest["data"]["max_length"] = 7
    manifest["config"]["comparison_examples"] = selected
    project.args.comparison_examples = selected
    _write(project.root / "project.json", manifest)
    setup_pair(project, monkeypatch)
    result = project.evaluate()
    assert result["status"] == "likelihood_improved"
    assert result["quality_certified"] is False
    audit = result["token_audit"]
    assert audit["examples"] == selected
    assert audit["aggregate"]["all"]["tokens"] == sum(
        len(r["input_ids"]) - 3 for r in rows[:selected]
    )
    assert audit["aggregate"]["generic/observed_id/eos"]["tokens"] == selected
    assert result["loss_delta"] == -0.25
    assert result["category_origin"]["kind"] == "observed_token_id_categories"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["rows"][0]["target_ids"].__setitem__(0, 9),
        lambda r: r["rows"][0]["predicted_ids"].__setitem__(0, 32),
        lambda r: r["rows"][0]["token_nll"].__setitem__(0, 0.1),
        lambda r: r["rows"][0].update(correct_tokens=0),
    ],
)
def test_bad_token_receipt_preserves_previous_result(project, monkeypatch, mutation):
    previous = b'{"status":"previous"}\n'
    (project.root / "comparison.json").write_bytes(previous)
    setup_pair(project, monkeypatch, mutation)
    with pytest.raises(ValueError):
        project.evaluate()
    assert (project.root / "comparison.json").read_bytes() == previous
    assert not (project.root / "report.html").exists()


def test_optional_categories_are_exact_selection_and_not_inferred(project, monkeypatch, tmp_path):
    setup_pair(project, monkeypatch)
    first = project.evaluate()
    origin = Path(first["token_audit"]["input_roles"]["categories"])
    category = tmp_path / "custom.json"
    value = json.loads(origin.read_text())
    value["rows"][0]["categories"] = ["user/code/target", "user/code/target"]
    value["counts"] = {"user/code/target": 2}
    _write(category, value)
    result = project.evaluate(categories=category)
    assert result["category_origin"]["kind"] == "supplied_categories_not_semantically_certified"
    before = (project.root / "comparison.json").read_bytes()
    value["rows"][0]["input_ids_sha256"] = "0" * 64
    _write(category, value)
    with pytest.raises(ValueError, match="sequence mismatch"):
        project.evaluate(categories=category)
    assert (project.root / "comparison.json").read_bytes() == before


def test_multiple_tokenizer_eos_ids_only(project, monkeypatch):
    import importlib

    data = importlib.import_module(fixtures.sft.__package__ + ".data")

    class Tokenizer:
        eos_token_id = [1, 6]

        def __len__(self):
            return 32

    monkeypatch.setattr(data, "load_tokenizer", lambda _: Tokenizer())
    setup_pair(project, monkeypatch)
    result = project.evaluate()
    assert result["category_origin"]["tokenizer_eos_ids"] == [1, 6]
    assert result["token_audit"]["aggregate"]["generic/observed_id/eos"]["tokens"] == 2


def test_category_source_drift_during_pair_keeps_old_comparison(project, monkeypatch, tmp_path):
    setup_pair(project, monkeypatch)
    first = project.evaluate()
    categories = tmp_path / "categories.json"
    categories.write_bytes(Path(first["token_audit"]["input_roles"]["categories"]).read_bytes())
    previous = (project.root / "comparison.json").read_bytes()
    original = project._child

    def child(*args, **kwargs):
        original(*args, **kwargs)
        if kwargs["label"] == "adapter":
            categories.write_bytes(categories.read_bytes() + b" ")

    monkeypatch.setattr(project, "_child", child)
    with pytest.raises(ValueError, match="Category source changed"):
        project.evaluate(categories=categories)
    assert (project.root / "comparison.json").read_bytes() == previous


def test_duplicate_child_json_rejected(project, monkeypatch):
    setup_pair(project, monkeypatch)
    original = project._child

    def child(command, **kwargs):
        original(command, **kwargs)
        path = Path(_arg(command, "--report"))
        path.write_text(path.read_text().replace("{", '{"loss": 99,', 1))

    monkeypatch.setattr(project, "_child", child)
    with pytest.raises(ValueError, match="Duplicate JSON"):
        project.evaluate()
    assert not (project.root / "comparison.json").exists()


def test_cli_exposes_optional_category_path():
    args = fixtures.cli.build_parser().parse_args(
        ["compare", "project", "--categories", "categories.json"]
    )
    assert args.categories == "categories.json"


@pytest.mark.parametrize(
    "field",
    [
        "index",
        "loss_start",
        "supervised_tokens",
        "input_ids_sha256",
        "categories",
        "counts",
        "nonmapping",
    ],
)
def test_invalid_categories_rejected_before_any_child(project, monkeypatch, tmp_path, field):
    setup_pair(project, monkeypatch)
    first = project.evaluate()
    category = json.loads(Path(first["token_audit"]["input_roles"]["categories"]).read_text())
    if field == "nonmapping":
        category = []
    elif field == "counts":
        category["counts"] = {}
    elif field == "categories":
        category["rows"][0][field] = ["invalid", "invalid"]
    elif field == "input_ids_sha256":
        category["rows"][0][field] = "0" * 64
    else:
        category["rows"][0][field] = 99
    path = tmp_path / "bad.json"
    _write(path, category)
    calls = []
    monkeypatch.setattr(project, "_child", lambda *a, **kw: calls.append(kw))
    previous = (project.root / "comparison.json").read_bytes()
    with pytest.raises(ValueError):
        project.evaluate(categories=path)
    assert calls == []
    assert (project.root / "comparison.json").read_bytes() == previous


def test_status_uses_token_weighted_not_macro_example_loss(project, monkeypatch):
    rows = [
        {"input_ids": [0, 2, 6, 1], "loss_start": 2},
        {"input_ids": [0, 2, 6, 6, 6, 6, 1], "loss_start": 2},
    ]
    path = project.root / "data/test.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads((project.root / "project.json").read_text())
    manifest["data_files"][path.name] = sft.digest(path)
    manifest["data"]["counts"]["test"] = 2
    manifest["data"]["seq_length"] = 6
    manifest["data"]["max_length"] = 7
    _write(project.root / "project.json", manifest)
    setup_pair(project, monkeypatch)

    def child(command, *, label, directory):
        result = _evaluation_result(project, command, loss=2.0)
        if label == "adapter":
            for row, loss in zip(result["rows"], [4.0, 1.0], strict=True):
                row["loss"] = loss
                row["token_nll"] = [loss] * len(row["target_ids"])
            result["loss"] = 13 / 7
        _write(Path(_arg(command, "--report")), result)

    monkeypatch.setattr(project, "_child", child)
    result = project.evaluate()
    assert result["status"] == "likelihood_improved"
    assert result["loss_delta"] == pytest.approx(-1 / 7)
    assert result["token_audit"]["macro_example_nll"]["delta"] == 0.5


def test_audit_artifact_mutated_during_preview_keeps_previous_comparison(project, monkeypatch):
    setup_pair(project, monkeypatch)
    project.evaluate()
    previous = (project.root / "comparison.json").read_bytes()
    previous_html = (project.root / "report.html").read_bytes()

    def preview(row, adapter, directory):
        artifact = directory / "token-audit.json"
        artifact.write_bytes(artifact.read_bytes() + b" ")
        return {"status": "fixture"}

    monkeypatch.setattr(project, "_paired_preview", preview)
    with pytest.raises(ValueError, match="Token audit artifact changed"):
        project.evaluate()
    assert (project.root / "comparison.json").read_bytes() == previous
    assert (project.root / "report.html").read_bytes() == previous_html
