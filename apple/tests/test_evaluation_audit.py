"""CPU-only receipt auditing: no tokenizer, MLX or model execution required."""

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "src/exl3_mlx/evaluation_audit.py"
spec = importlib.util.spec_from_file_location("standalone_evaluation_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def checksum(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fixture(count=3):
    prepared = []
    categories = {"prediction_vocab_size": 32, "tokenizer_size": 30, "rows": [], "counts": {}}
    reports = [{"rows": []}, {"rows": []}]
    for i in range(count):
        targets = [3, 4, 2][: i % 3 + 1]
        ids = [1, 5, *targets]
        prepared.append({"input_ids": ids, "loss_start": 2})
        label = "metal/added/code_or_other" if i % 2 else "host/removed/comment"
        categories["rows"].append(
            {
                "index": i,
                "loss_start": 2,
                "input_ids_sha256": checksum(ids),
                "supervised_tokens": len(targets),
                "categories": [label] * len(targets),
            }
        )
        categories["counts"][label] = categories["counts"].get(label, 0) + len(targets)
        for condition, report in enumerate(reports):
            nll = [0.25 * (i + 1) + 0.05 * j + 0.1 * condition for j in range(len(targets))]
            predicted = list(targets)
            if (i + condition) % 2:
                predicted[0] = 31  # A valid padded-model-vocabulary prediction.
            report["rows"].append(
                {
                    "target_ids": list(targets),
                    "token_nll": nll,
                    "predicted_ids": predicted,
                    "supervised_tokens": len(targets),
                    "correct_tokens": sum(a == b for a, b in zip(predicted, targets)),
                    "loss": math.fsum(nll) / len(nll),
                    "logits_sha256": "a" * 64,
                }
            )
    for report in reports:
        values = [v for row in report["rows"] for v in row["token_nll"]]
        report.update(
            examples=count,
            supervised_tokens=len(values),
            loss=math.fsum(values) / len(values),
            evaluation_context={
                "selected_examples_sha256": checksum(prepared),
                "seq_length": max(len(p["input_ids"]) - 1 for p in prepared),
            },
        )
    return prepared, categories, *reports


@pytest.mark.parametrize("count", [1, 3, 17])
def test_arbitrary_nonempty_counts_weighted_and_macro_separate(count):
    inputs = fixture(count)
    result = audit.compare_token_evaluations(*inputs)
    assert result["examples"] == count
    assert result["quality_certified"] is False
    assert result["status"] == "metrics_consistent_pending_provenance_audit"
    assert result["aggregate"]["all"]["base_nll"] == inputs[2]["loss"]
    assert result["macro_example_nll"]["base"] == pytest.approx(
        sum(row["loss"] for row in inputs[2]["rows"]) / count
    )
    assert result["aggregate"]["all"]["nll_delta"] == pytest.approx(0.1)
    if count > 1:
        assert result["aggregate"]["all"]["base_nll"] != result["macro_example_nll"]["base"]
    assert result["aggregate"]["all"]["corrected_tokens"] == count // 2
    assert result["aggregate"]["all"]["regressed_tokens"] == (count + 1) // 2


def test_empty_dataset_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        audit.compare_token_evaluations([], {}, {}, {})


@pytest.mark.parametrize("bad", [0, -1, True, 3, 4, 2.0, None])
def test_loss_boundary_must_leave_nonempty_target(bad):
    data = fixture(1)
    data[0][0]["loss_start"] = bad
    with pytest.raises(ValueError):
        audit.compare_token_evaluations(*data)


@pytest.mark.parametrize("bad", [True, -1, 30, 32, 3.0, None])
def test_prompt_and_target_ids_must_be_tokenizer_and_model_valid(bad):
    data = fixture()
    data[0][0]["input_ids"][0] = bad
    with pytest.raises(ValueError, match="vocabulary"):
        audit.compare_token_evaluations(*data)


def test_full_prompt_and_mask_are_bound_not_only_targets():
    for change in ("prompt", "mask"):
        data = fixture()
        if change == "prompt":
            data[0][0]["input_ids"][0] = 7
        else:
            data[0][0]["loss_start"] = 1
        with pytest.raises(ValueError, match="full sequences|Category sequence mismatch|Category mask mismatch"):
            audit.compare_token_evaluations(*data)


@pytest.mark.parametrize(
    "where,key,value",
    [
        ("category", "index", True),
        ("category", "index", 1),
        ("category", "loss_start", 1),
        ("category", "input_ids_sha256", "b" * 64),
        ("category", "supervised_tokens", True),
        ("category", "categories", []),
        ("category", "categories", ["metal//code"]),
        ("category", "categories", [" metal/added/code"]),
        ("category", "categories", ["family:metal/added/code"]),
        ("row", "target_ids", [True]),
        ("row", "target_ids", [4]),
        ("row", "predicted_ids", []),
        ("row", "predicted_ids", [True]),
        ("row", "predicted_ids", [-1]),
        ("row", "predicted_ids", [32]),
        ("row", "token_nll", []),
        ("row", "token_nll", [True]),
        ("row", "token_nll", [float("nan")]),
        ("row", "token_nll", [float("inf")]),
        ("row", "token_nll", [-0.01]),
        ("row", "token_nll", [10**1000]),
        ("row", "supervised_tokens", True),
        ("row", "correct_tokens", 0),
        ("row", "correct_tokens", True),
        ("row", "loss", 0.3),
        ("row", "loss", float("nan")),
        ("row", "logits_sha256", "bad"),
        ("report", "examples", True),
        ("report", "examples", 2),
        ("report", "supervised_tokens", True),
        ("report", "supervised_tokens", 2),
        ("report", "loss", 0.7),
        ("report", "loss", True),
    ],
)
def test_inconsistent_receipts_fail_closed(where, key, value):
    data = fixture(1)
    target = {"category": data[1]["rows"][0], "row": data[2]["rows"][0], "report": data[2]}[where]
    target[key] = value
    with pytest.raises(ValueError):
        audit.compare_token_evaluations(*data)


@pytest.mark.parametrize("side", [2, 3])
def test_both_conditions_require_full_context_and_matching_row_count(side):
    data = fixture()
    data[side]["evaluation_context"]["seq_length"] = 2
    with pytest.raises(ValueError, match="truncation"):
        audit.compare_token_evaluations(*data)
    data = fixture()
    data[side]["rows"].pop()
    with pytest.raises(ValueError, match="count"):
        audit.compare_token_evaluations(*data)


@pytest.mark.parametrize("bad", [True, 0, -1, 31.0, None])
def test_vocabulary_dimensions_are_positive_integers(bad):
    data = fixture()
    data[1]["prediction_vocab_size"] = bad
    with pytest.raises(ValueError):
        audit.compare_token_evaluations(*data)


def test_category_count_mapping_and_weighted_global_loss_checked():
    data = fixture()
    data[1]["counts"]["host/removed/comment"] = 99
    with pytest.raises(ValueError, match="Category counts"):
        audit.compare_token_evaluations(*data)
    data = fixture()
    data[2]["loss"] = sum(row["loss"] for row in data[2]["rows"]) / 3
    with pytest.raises(ValueError, match="token-weighted"):
        audit.compare_token_evaluations(*data)


def file_fixture(tmp_path):
    data = fixture()
    paths = {}
    for role, value in zip(("prepared", "categories", "base", "adapter"), data, strict=True):
        path = tmp_path / (role + ".json")
        text = (
            "".join(json.dumps(row) + "\n" for row in value)
            if role == "prepared"
            else json.dumps(value)
        )
        path.write_text(text)
        paths[role] = path
    paths["output"] = tmp_path / "audit.json"
    return paths


def test_file_audit_binds_inputs_and_exclusive_publication(tmp_path):
    paths = file_fixture(tmp_path)
    result = audit.audit_files(**paths)
    assert json.loads(paths["output"].read_text()) == result
    assert result["inputs_sha256"] == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for role, path in paths.items()
        if role != "output"
    }
    with pytest.raises(ValueError, match="already exists"):
        audit.audit_files(**paths)
    assert not list(tmp_path.glob(".token-audit-*"))


def test_input_drift_leaves_no_output(tmp_path, monkeypatch):
    paths = file_fixture(tmp_path)
    original = audit.compare_token_evaluations

    def change_after_comparison(*args):
        result = original(*args)
        paths["base"].write_text("{}")
        return result

    monkeypatch.setattr(audit, "compare_token_evaluations", change_after_comparison)
    with pytest.raises(ValueError, match="changed during"):
        audit.audit_files(**paths)
    assert not paths["output"].exists()


def test_invalid_input_leaves_no_output(tmp_path):
    paths = file_fixture(tmp_path)
    paths["base"].write_text("{}")
    with pytest.raises(ValueError):
        audit.audit_files(**paths)
    assert not paths["output"].exists()


@pytest.mark.parametrize(
    "role,old,new",
    [
        (
            "categories",
            '"prediction_vocab_size": 32',
            '"prediction_vocab_size": 9,"prediction_vocab_size": 32',
        ),
        ("base", '"loss": 0.25', '"loss": 999,"loss": 0.25'),
        ("adapter", '"target_ids": [3]', '"target_ids": [999],"target_ids": [3]'),
        ("prepared", '"loss_start": 2', '"loss_start": 1,"loss_start": 2'),
        ("base", '"loss": 0.25', '"loss": NaN'),
    ],
)
def test_ambiguous_or_nonfinite_json_rejected_before_publication(tmp_path, role, old, new):
    paths = file_fixture(tmp_path)
    content = paths[role].read_text()
    assert old in content
    paths[role].write_text(content.replace(old, new, 1))
    with pytest.raises(ValueError, match="Duplicate JSON key|Nonfinite JSON"):
        audit.audit_files(**paths)
    assert not paths["output"].exists()


def test_cli_success_failure_and_registered_entry_point(tmp_path):
    paths = file_fixture(tmp_path)
    command = [sys.executable, str(SCRIPT)]
    for role, path in paths.items():
        command.extend(["--" + role, str(path)])
    run = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(run.stdout)["examples"] == 3
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode == 2 and "already exists" in again.stderr
    config = tomllib.loads((SCRIPT.parents[2] / "pyproject.toml").read_text())
    assert config["project"]["scripts"]["exl3-mlx-audit"] == "exl3_mlx.evaluation_audit:main"


def test_no_mlx_or_tokenizer_import_in_fresh_process():
    code = (
        "import runpy,sys; runpy.run_path(sys.argv[1],run_name='cpu_only'); "
        "assert not any(k.startswith(('mlx','tokenizers')) for k in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code, str(SCRIPT)], check=True)
