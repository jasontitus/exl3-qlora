"""CPU-only fail-closed validation of fit-summary evidence."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "summarize_fit_tokens", Path(__file__).parents[1] / "scripts" / "summarize_fit_tokens.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.fixture
def evidence():
    base = {
        "loss": 0.5,
        "supervised_tokens": 2,
        "target_ids": [8, 1],
        "predicted_ids": [9, 1],
        "token_nll": [0.7, 0.3],
        "logits_sha256": "base",
    }
    after = {
        "loss": 0.25,
        "supervised_tokens": 2,
        "target_ids": [8, 1],
        "predicted_ids": [8, 1],
        "token_nll": [0.4, 0.1],
        "logits_sha256": "trained",
    }
    report = {
        "status": "completed",
        "settings": {"rows": [1]},
        "binding": {"data_sha256": "prepared"},
        "base_scores": {"1": base},
        "history": [{"step": 1, "post_update_scores": {"1": after}}],
    }
    categories = {
        "1": {
            "supervised_token_categories": ["added_code", "eos"],
            "supervised_token_ids": [8, 1],
            "source_sha256": {"prepared": "prepared"},
        }
    }
    return report, categories


def test_valid_fit_summary(evidence):
    result = module.summarize(*evidence)
    assert result["every_row_improved"]
    assert result["weighted"] == {"base_loss": 0.5, "trained_loss": 0.25}
    assert result["rows"]["1"]["categories"]["added_code"]["correct_token_delta"] == 1
    assert not result["quality_certified"]
    assert result["selected_step"] == 1


@pytest.mark.parametrize("status", ["training", "failed", "captured"])
def test_partial_or_failed_status_rejected(evidence, status):
    report, labels = evidence
    report["status"] = status
    with pytest.raises(ValueError, match="completed"):
        module.summarize(report, labels)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_losses_rejected(evidence, value):
    report, labels = evidence
    report["base_scores"]["1"]["token_nll"][0] = value
    with pytest.raises(ValueError, match="finite"):
        module.summarize(report, labels)


@pytest.mark.parametrize("kind", ["count", "scalar", "source", "targets", "missing_uncached"])
def test_inconsistent_evidence_rejected(evidence, kind):
    report, labels = evidence
    if kind == "count":
        report["base_scores"]["1"]["supervised_tokens"] = 3
    elif kind == "scalar":
        report["base_scores"]["1"]["loss"] = 0.1
    elif kind == "source":
        labels["1"]["source_sha256"]["prepared"] = "different"
    elif kind == "targets":
        labels["1"]["supervised_token_ids"] = [7, 1]
    else:
        report["settings"]["verify_uncached_final"] = True
    with pytest.raises(ValueError):
        module.summarize(report, labels)


def test_fp32_chunk_rounding_tolerance_is_bounded(evidence):
    report, labels = evidence
    report["base_scores"]["1"]["loss"] += 0.5e-6
    module.summarize(report, labels)
    report["base_scores"]["1"]["loss"] += 2e-6
    with pytest.raises(ValueError, match="disagrees"):
        module.summarize(report, labels)


def test_uncached_parity_requires_matching_raw_scores(evidence):
    report, labels = evidence
    report["settings"]["verify_uncached_final"] = True
    report["uncached_final_parity_exact"] = True
    with pytest.raises(ValueError, match="Uncached final scores"):
        module.summarize(report, labels)
    report["uncached_final_scores"] = deepcopy(report["history"][-1]["post_update_scores"])
    module.summarize(report, labels)
    report["uncached_final_scores"]["1"]["logits_sha256"] = "different"
    with pytest.raises(ValueError, match="Uncached final scores"):
        module.summarize(report, labels)
