"""CPU-only adversarial tests using the inspected Qwen official/local schemas."""

import copy
import gzip
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "src/exl3_mlx/ktune_eval.py"
_SPEC = importlib.util.spec_from_file_location("ktune_eval_cpu_tests", _PATH)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def official():
    # Names, nesting and two-pair contract taken from the actual tracked dump.
    # Values are controlled fixtures, not claims about a measured optimization.
    metrics = {
        "passed_correctness": True,
        "partial_result": False,
        "baseline_band_passed": True,
        "passed_decode_speedup_floor": True,
        "passed_prefill_speedup_floor": True,
        "decode_speedup_floor": 0.95,
        "prefill_speedup_floor": 0.95,
        "error": "",
        "max_abs_diff": 0,
        "case_count": 1,
        "checked_steps": 64,
        "harness_hash": "a" * 64,
        "golden_hash": "b" * 64,
        "baseline_golden_sha256": "b" * 64,
        "weights_hash": "c" * 64,
        "commit": "d" * 40,
        "baseline_reference_commit": "e" * 40,
        "engine_device": "m5",
        "baseline_box": "fixture-m5",
        "paired_legs": [
            {
                "pair": n,
                "control_decode_seconds_per_token": 0.032,
                "candidate_decode_seconds_per_token": 0.016,
                "control_prefill_seconds_per_token": 0.0008,
                "candidate_prefill_seconds_per_token": 0.0004,
            }
            for n in (1, 2)
        ],
        "baseline_decode_seconds_per_token": 0.032,
        "decode_seconds_per_token": 0.016,
        "decode_speedup": 2.0,
        "baseline_prefill_seconds_per_token": 0.0008,
        "prefill_seconds_per_token": 0.0004,
        "prefill_speedup": 2.0,
    }
    return {
        "id": "test-submission",
        "benchmarkId": "test-benchmark",
        "createdAt": "2026-09-24T16:23:59.160Z",
        "status": "accepted",
        "promotionStatus": "promoted",
        "improved": True,
        "officialScore": 2.0,
        "submissionCommitSha": "d" * 40,
        "promotedSourceRef": "d" * 40,
        "officialMetrics": metrics,
    }


def local(ratios=(1.02, 1.021, 1.019, 1.02)):
    pairs = [
        {
            "control": {"decode": r, "prefill": 1.0, "passed_correctness": True},
            "candidate": {"decode": 1.0, "prefill": 1.0, "passed_correctness": True},
        }
        for r in ratios
    ]
    logs = [math.log(r) for r in ratios]
    gain = math.exp(statistics.fmean(logs))
    aggregate = {
        "decode_gain": gain,
        "prefill_gain": 1.0,
        "composite_gain": gain**0.75,
        "decode_latency_reduction": 1 - 1 / gain,
        "pair_wins": sum(r > 1 for r in ratios),
        "pair_count": 4,
        "one_sided_95_log_gain_lower_bound": (
            statistics.fmean(logs) - 2.353 * statistics.stdev(logs) / 2
        ),
        "unrelated_prefill_movement": 0.0,
    }
    passed = (
        aggregate["decode_latency_reduction"] >= 0.002
        and aggregate["pair_wins"] >= 3
        and aggregate["one_sided_95_log_gain_lower_bound"] > 0
    )
    return {
        "submission_authority": False,
        "source_parent_id": "f" * 64,
        "task_id": "fixture-task",
        "candidate_id": "0001",
        "correctness": {"status": "passed_exact", "all_legs": True},
        "status": "promotion_screen_passed" if passed else "retired_local_result",
        "attachable": passed,
        "performance": {
            "confirmed": passed,
            "local_only": True,
            "comparable": True,
            "score": aggregate["composite_gain"],
            "aggregate": aggregate,
            "schedule": copy.deepcopy(audit._SCHEDULE),
            "pairs": pairs,
        },
    }


def test_official_receipt_scope_and_integrity_are_explicit():
    result = audit.validate_official_submission(official())
    assert result["valid"] and result["evidence_level"] == "official_promoted"
    assert result["metrics"]["case_count"] == 1
    assert result["metrics"]["checked_steps"] == 64
    assert result["quality_certified"] is False
    assert result["source_payload_verified"] is False
    assert result["independently_reproduced"] is False
    changed = official()
    changed["createdAt"] = "2026-09-25T16:23:59.160Z"
    assert audit.validate_official_submission(changed)["receipt_sha256"] != result["receipt_sha256"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "rejected"),
        ("promotionStatus", "superseded"),
        ("improved", 1),
        ("promotedSourceRef", "1" * 40),
        ("submissionCommitSha", "1" * 40),
        ("officialScore", 3),
        ("officialScore", float("nan")),
        ("officialMetrics", None),
    ],
)
def test_official_rejects_nonpromotions_and_forged_identity_or_score(key, value):
    row = official()
    row[key] = value
    assert not audit.validate_official_submission(row)["valid"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("passed_correctness", False),
        ("passed_correctness", 1),
        ("partial_result", True),
        ("baseline_band_passed", False),
        ("error", "wrong output"),
        ("max_abs_diff", 1e-10),
        ("passed_prefill_speedup_floor", False),
        ("decode_speedup_floor", 3.0),
        ("first_failing_step", 0),
        ("case_count", 0),
        ("checked_steps", True),
        ("golden_hash", "f" * 64),
        ("harness_hash", ""),
        ("weights_hash", None),
        ("baseline_reference_commit", ""),
        ("decode_speedup", 9),
        ("baseline_decode_seconds_per_token", 0.02),
        ("decode_seconds_per_token", 0.03),
    ],
)
def test_official_rejects_contradictory_or_missing_evidence(key, value):
    row = official()
    row["officialMetrics"][key] = value
    assert not audit.validate_official_submission(row)["valid"]


@pytest.mark.parametrize("timing", [0, -1, True, "0.01", float("nan"), float("inf")])
def test_official_invalid_pair_cannot_pass_with_unchanged_summary(timing):
    row = official()
    row["officialMetrics"]["paired_legs"][0]["candidate_decode_seconds_per_token"] = timing
    assert not audit.validate_official_submission(row)["valid"]


def test_official_dropped_or_duplicated_pair_is_rejected():
    row = official()
    row["officialMetrics"]["paired_legs"].pop()
    assert not audit.validate_official_submission(row)["valid"]
    row = official()
    row["officialMetrics"]["paired_legs"][1]["pair"] = 1
    assert not audit.validate_official_submission(row)["valid"]


def test_local_screen_pass_does_not_claim_promotion():
    result = audit.validate_local_qwen_result(local())
    assert result["valid"] and result["screen_passed"]
    assert result["official_promotion"] is False
    assert result["evidence_level"] == "local_screening_only"
    assert result["source_payload_verified"] is False


@pytest.mark.parametrize(
    "ratios",
    [
        (1.0, 1.0, 1.0, 1.0),
        (0.99, 0.99, 0.99, 0.99),
        (1.02, 1.03, 0.99, 1.0),
        (1.001, 1.001, 1.001, 1.001),
    ],
)
def test_neutral_regressing_noisy_and_impractical_gains_remain_negative(ratios):
    row = local(ratios)
    result = audit.validate_local_qwen_result(row)
    assert result["valid"] and not result["screen_passed"]
    row["status"] = "promotion_screen_passed"
    row["attachable"] = row["performance"]["confirmed"] = True
    assert not audit.validate_local_qwen_result(row)["valid"]


def test_local_timing_aggregate_is_recomputed():
    row = local()
    row["performance"]["pairs"][2]["candidate"]["decode"] = 2.0
    result = audit.validate_local_qwen_result(row)
    assert not result["valid"]
    assert any("aggregate.decode_gain" in error for error in result["errors"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(submission_authority=True),
        lambda r: r.update(source_parent_id=""),
        lambda r: r["performance"].update(local_only=False),
        lambda r: r["performance"]["pairs"][1]["candidate"].update(passed_correctness=False),
        lambda r: r["performance"]["pairs"][1]["control"].update(decode=float("inf")),
        lambda r: r["performance"]["schedule"][0].reverse(),
        lambda r: r["performance"]["pairs"].pop(),
        lambda r: r["performance"]["aggregate"].update(pair_wins=2),
    ],
)
def test_local_contradictions_are_rejected(mutation):
    row = local()
    mutation(row)
    assert not audit.validate_local_qwen_result(row)["valid"]


def test_malformed_objects_fail_without_importing_gpu():
    for value in (None, [], "accepted", {}):
        assert not audit.validate_official_submission(value)["valid"]
        assert not audit.validate_local_qwen_result(value)["valid"]


def test_invalid_numbers_still_produce_serializable_audit():
    for value in (float("nan"), float("inf"), 10**400):
        row = official()
        row["officialMetrics"]["case_count"] = value
        row["officialScore"] = value
        result = audit.validate_official_submission(row)
        assert not result["valid"]
        json.dumps(result, allow_nan=False)


def test_cli_audits_compressed_dump_and_retains_negative_receipts(tmp_path, capsys):
    positive, negative = official(), official()
    negative["status"] = "rejected"
    source, output = tmp_path / "submissions.json.gz", tmp_path / "audit.json"
    source.write_bytes(gzip.compress(json.dumps([positive, negative]).encode()))
    audit.main(["--kind", "official", "--input", str(source), "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["receipts"] == 2 and report["valid"] == 1
    assert len(report["audits"]) == 2 and report["quality_certified"] is False
    assert report["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert json.loads(capsys.readouterr().out)["valid"] == 1
