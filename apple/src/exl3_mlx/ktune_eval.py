"""CPU-only audits of the actual ktune Qwen receipt formats.

These checks establish internal consistency, not authenticity, full source-payload
equivalence, independent reruns, or model quality. No benchmark code is executed.
The official schema comes from the tracked Yukon submission dump; the local
four-pair formula comes from ktune/qwen38.py. Local screening is never promotion.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

_SCHEDULE = [
    ["control", "candidate"],
    ["candidate", "control"],
    ["candidate", "control"],
    ["control", "candidate"],
]


def _number(value, *, positive=False):
    try:
        return type(value) in (int, float) and math.isfinite(value) and (not positive or value > 0)
    except OverflowError:
        return False


def _hash(value, length=64):
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _close(actual, expected):
    return _number(actual) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-12)


def _receipt(value, errors, kind, metrics=None):
    try:
        digest = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    except (ValueError, TypeError):
        digest = None
    return {
        "format": "exl3-ktune-receipt-audit-v1",
        "kind": kind,
        "valid": not errors,
        "errors": errors,
        "receipt_sha256": digest,
        "metrics": _json_safe(metrics or {}),
        "quality_certified": False,
        "source_payload_verified": False,
        "independently_reproduced": False,
        "limitations": [
            "Internal receipt consistency does not authenticate the reporting authority.",
            "No source payload, compiler execution, numerical test or timing rerun was performed.",
            "Historical benchmark correctness does not establish unseen-workload generalization.",
        ],
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if type(value) in (int, float) and not _number(value):
        return None
    return value


def validate_official_submission(row):
    """Check strict eligibility as a historically promoted Qwen receipt.

    ``valid`` is false for non-promotions and for missing/contradictory evidence.
    A measured/promoted commit mismatch needs external code-equivalence proof;
    it is not a verdict that either implementation was incorrect.
    """
    errors = []
    if not isinstance(row, dict):
        return _receipt(row, ["submission must be an object"], "official_qwen")
    for key, expected in (
        ("status", "accepted"),
        ("promotionStatus", "promoted"),
        ("improved", True),
    ):
        if type(row.get(key)) is not type(expected) or row[key] != expected:
            errors.append(f"{key} must be {expected!r}")
    for key in ("id", "benchmarkId", "createdAt"):
        if not isinstance(row.get(key), str) or not row[key]:
            errors.append(f"missing {key}")
    if not _number(row.get("officialScore"), positive=True):
        errors.append("officialScore must be finite and positive")
    m = row.get("officialMetrics")
    if not isinstance(m, dict):
        return _receipt(row, errors + ["missing officialMetrics"], "official_qwen")
    for key, expected in (
        ("passed_correctness", True),
        ("partial_result", False),
        ("baseline_band_passed", True),
        ("passed_decode_speedup_floor", True),
        ("passed_prefill_speedup_floor", True),
    ):
        if m.get(key) is not expected:
            errors.append(f"officialMetrics.{key} must be {expected!r}")
    if m.get("error") != "":
        errors.append("officialMetrics.error must be empty")
    if not _number(m.get("max_abs_diff")) or m["max_abs_diff"] != 0:
        errors.append("official exact-correctness receipt requires max_abs_diff=0")
    for key in (
        "first_failing_case",
        "first_failing_layer",
        "first_failing_step",
        "actual_token",
        "expected_token",
    ):
        if m.get(key) is not None:
            errors.append(f"contradictory correctness failure marker: {key}")
    for key in ("case_count", "checked_steps"):
        if type(m.get(key)) is not int or m[key] <= 0:
            errors.append(f"{key} must be a positive integer")
    for key in ("harness_hash", "golden_hash", "baseline_golden_sha256", "weights_hash"):
        if not _hash(m.get(key)):
            errors.append(f"missing/invalid identity: {key}")
    if m.get("golden_hash") != m.get("baseline_golden_sha256"):
        errors.append("candidate and baseline golden identities differ")
    for key in ("engine_device", "baseline_box"):
        if not isinstance(m.get(key), str) or not m[key]:
            errors.append(f"missing hardware identity: {key}")
    commits = (row.get("submissionCommitSha"), row.get("promotedSourceRef"), m.get("commit"))
    if not all(_hash(c, 40) for c in commits):
        errors.append("missing/invalid submission, measured or promoted commit identity")
    elif len(set(commits)) != 1:
        errors.append("measured/promoted commit mismatch: source equivalence must be proved")
    if not _hash(m.get("baseline_reference_commit"), 40):
        errors.append("missing/invalid baseline_reference_commit")
    pairs = m.get("paired_legs")
    if not isinstance(pairs, list) or len(pairs) != 2:
        errors.append("tracked Qwen official contract requires two paired_legs")
        pairs = []
    ratios = {}
    if pairs:
        if any(not isinstance(p, dict) or type(p.get("pair")) is not int for p in pairs) or [
            p.get("pair") if isinstance(p, dict) else None for p in pairs
        ] != [1, 2]:
            errors.append("paired_legs must have unique ordered pair IDs 1,2")
        for axis in ("decode", "prefill"):
            control, candidate = [], []
            for pair in pairs:
                for role, values in (("control", control), ("candidate", candidate)):
                    value = (
                        pair.get(f"{role}_{axis}_seconds_per_token")
                        if isinstance(pair, dict)
                        else None
                    )
                    if not _number(value, positive=True):
                        errors.append(f"invalid {role} {axis} pair timing")
                    else:
                        values.append(value)
            if len(control) == len(candidate) == 2:
                c, t = (sum(v * 0.5 for v in values) for values in (control, candidate))
                if not _number(t, positive=True) or not _number(c / t, positive=True):
                    errors.append(f"{axis} timing ratio cannot be represented finitely")
                    continue
                ratios[axis] = c / t
                floor = m.get(f"{axis}_speedup_floor")
                if not _number(floor, positive=True) or c / t < floor:
                    errors.append(f"missing/failed {axis}_speedup_floor")
                for key, expected in (
                    (f"baseline_{axis}_seconds_per_token", c),
                    (f"{axis}_seconds_per_token", t),
                    (f"{axis}_speedup", c / t),
                ):
                    if not _close(m.get(key), expected):
                        errors.append(f"{key} contradicts paired timings")
    if len(ratios) == 2:
        score = ratios["decode"] ** 0.75 * ratios["prefill"] ** 0.25
        if not _number(score, positive=True) or not _close(row.get("officialScore"), score):
            errors.append("officialScore contradicts paired composite score")
    result = _receipt(
        row,
        errors,
        "official_qwen",
        {
            "reported_status": row.get("status"),
            "submission_id": row.get("id"),
            "case_count": m.get("case_count"),
            "checked_steps": m.get("checked_steps"),
            "pair_count": len(pairs),
            "speedup_vs_fixed_baseline": ratios,
            "measured_commit": m.get("commit"),
            "promoted_commit": row.get("promotedSourceRef"),
        },
    )
    result["evidence_level"] = "official_promoted" if not errors else "not_verified_promotion"
    result["limitations"].append(
        "Paired speedups use the fixed baseline, not the immediately preceding champion."
    )
    return result


def validate_local_qwen_result(row):
    """Recompute the recorded four-pair screening gate; never infer promotion."""
    errors = []
    if not isinstance(row, dict):
        return _receipt(row, ["result must be an object"], "local_qwen")
    if row.get("submission_authority") is not False:
        errors.append("local result must declare submission_authority=false")
    if not _hash(row.get("source_parent_id")):
        errors.append("missing/invalid source_parent_id")
    for key in ("task_id", "candidate_id"):
        if not isinstance(row.get(key), str) or not row[key]:
            errors.append(f"missing {key}")
    if row.get("correctness") != {"status": "passed_exact", "all_legs": True}:
        errors.append("missing/contradictory all-legs exact correctness")
    p = row.get("performance")
    if not isinstance(p, dict):
        return _receipt(row, errors + ["missing performance"], "local_qwen")
    for key in ("local_only", "comparable"):
        if p.get(key) is not True:
            errors.append(f"performance.{key} must be true")
    if p.get("schedule") != _SCHEDULE:
        errors.append("four-pair schedule must be AB,BA,BA,AB")
    pairs = p.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 4:
        errors.append("local screening requires four predeclared pairs")
        pairs = []
    logs = {"decode": [], "prefill": []}
    for pair in pairs:
        for axis in logs:
            values = []
            for role in ("control", "candidate"):
                leg = pair.get(role) if isinstance(pair, dict) else None
                if not isinstance(leg, dict) or leg.get("passed_correctness") is not True:
                    errors.append(f"{role} leg lacks exact correctness")
                    continue
                value = leg.get(axis)
                if not _number(value, positive=True):
                    errors.append(f"invalid {role} {axis} timing")
                else:
                    values.append(value)
            if len(values) == 2:
                logs[axis].append(math.log(values[0]) - math.log(values[1]))
    computed = {}
    passed = False
    if all(len(v) == 4 for v in logs.values()):
        dmean, pmean = (statistics.fmean(logs[a]) for a in ("decode", "prefill"))
        try:
            d, pre = math.exp(dmean), math.exp(pmean)
            computed = {
                "decode_gain": d,
                "prefill_gain": pre,
                "composite_gain": math.exp(dmean * 0.75 + pmean * 0.25),
                "decode_latency_reduction": 1 - math.exp(-dmean),
                "pair_wins": sum(v > 0 for v in logs["decode"]),
                "pair_count": 4,
                "one_sided_95_log_gain_lower_bound": (
                    dmean - 2.353 * statistics.stdev(logs["decode"]) / 2
                ),
                "unrelated_prefill_movement": abs(pre - 1),
            }
        except OverflowError:
            errors.append("timing ratio overflows aggregation")
    if computed:
        aggregate = p.get("aggregate")
        for key, expected in computed.items():
            if not isinstance(aggregate, dict) or not _close(aggregate.get(key), expected):
                errors.append(f"aggregate.{key} contradicts paired timings")
        if not _close(p.get("score"), computed["composite_gain"]):
            errors.append("performance.score contradicts paired timings")
        passed = (
            computed["decode_latency_reduction"] >= 0.002
            and computed["pair_wins"] >= 3
            and computed["one_sided_95_log_gain_lower_bound"] > 0
            and computed["unrelated_prefill_movement"] <= 0.01
            and computed["composite_gain"] > 1
        )
        expected_status = "promotion_screen_passed" if passed else "retired_local_result"
        if row.get("status") != expected_status:
            errors.append("status contradicts recomputed local screening gate")
        if row.get("attachable") is not passed or p.get("confirmed") is not passed:
            errors.append("attachable/confirmed contradict recomputed local screening gate")
    result = _receipt(row, list(dict.fromkeys(errors)), "local_qwen", computed)
    result.update(
        evidence_level="local_screening_only",
        screen_passed=passed and not errors,
        official_promotion=False,
    )
    result["limitations"].append(
        "The local result has a parent ID but no measured candidate content hash; "
        "pairing it with a candidate file by directory alone does not bind the measured payload."
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("official", "local"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.input.read_bytes()
    payload = gzip.decompress(raw) if args.input.suffix == ".gz" else raw
    value = json.loads(payload)
    rows = value if isinstance(value, list) else [value]
    validator = (
        validate_official_submission if args.kind == "official" else validate_local_qwen_result
    )
    audits = [validator(row) for row in rows]
    report = {
        "format": "exl3-ktune-receipt-audits-v1",
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "kind": args.kind,
        "receipts": len(rows),
        "valid": sum(a["valid"] for a in audits),
        "quality_certified": False,
        "error_counts": dict(Counter(error for a in audits for error in a["errors"])),
        "audits": audits,
    }
    if args.kind == "official":
        accepted = [
            a
            for row, a in zip(rows, audits, strict=True)
            if isinstance(row, dict) and row.get("status") == "accepted"
        ]
        report.update(
            validity_definition="strict historical-promotion receipt eligibility; nonpromotions fail by design",
            reported_accepted=len(accepted),
            accepted_eligible=sum(a["valid"] for a in accepted),
            accepted_error_counts=dict(Counter(error for a in accepted for error in a["errors"])),
        )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "audits"}))


if __name__ == "__main__":
    main()
