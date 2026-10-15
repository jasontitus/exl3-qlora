"""Summarize explicit workload mixes using measured per-invocation durations.

CPU only. This arithmetic neither qualifies kernels nor infers production traffic.
Run with ``python -m exl3_mlx.workload_cost input.json``.
"""

import argparse
import json
import math
from pathlib import Path


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _mean(values, name):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must contain measured per-invocation samples")
    for value in values:
        _number(value, name, positive=True)
    return _number(math.fsum(v / len(values) for v in values), name, positive=True)


def summarize(document):
    """Return mean-duration cost shares and, if supplied, ratio of total costs.

    ``invocations`` is an explicit workload weight, not the number of timing
    samples. Candidate samples must cover every case or none. Samples are in
    one declared unit and must already be normalized to one invocation.
    """
    if not isinstance(document, dict) or set(document) != {"unit", "mix_basis", "cases"}:
        raise ValueError("Expected unit, mix_basis and cases only")
    if document["unit"] not in ("s", "ms", "us", "ns"):
        raise ValueError("Declare one supported duration unit")
    if document["mix_basis"] not in ("designed", "observed"):
        raise ValueError("mix_basis must distinguish designed from observed traffic")
    cases = document["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("At least one case is required")
    rows, ids, candidate_modes = [], set(), set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        required = {"id", "invocations", "baseline"}
        if not required <= set(case) or set(case) - required - {"candidate"}:
            raise ValueError("Case requires id, invocations, baseline and optional candidate")
        identity = case["id"]
        if not isinstance(identity, str) or not identity.strip() or identity in ids:
            raise ValueError("Case IDs must be nonempty and unique")
        ids.add(identity)
        weight = _number(case["invocations"], "invocations")
        baseline = _mean(case["baseline"], "baseline")
        row = {"id": identity, "invocations": weight, "baseline_mean": baseline,
               "baseline_cost": _number(weight * baseline, "baseline cost")}
        has_candidate = "candidate" in case
        candidate_modes.add(has_candidate)
        if has_candidate:
            candidate = _mean(case["candidate"], "candidate")
            row.update(candidate_mean=candidate,
                       candidate_cost=_number(weight * candidate, "candidate cost"))
        rows.append(row)
    if len(candidate_modes) != 1:
        raise ValueError("Candidate coverage must be complete or absent")
    total = _number(math.fsum(r["baseline_cost"] for r in rows), "total baseline cost", positive=True)
    for row in rows:
        row["baseline_time_share"] = row["baseline_cost"] / total
    result = {"unit": document["unit"], "mix_basis": document["mix_basis"],
              "estimator": "invocation weights times arithmetic mean per-invocation duration",
              "baseline_total_cost": total, "cases": rows,
              "qualification": "arithmetic only; no correctness, speed significance or traffic verification"}
    if True in candidate_modes:
        candidate_total = _number(math.fsum(r["candidate_cost"] for r in rows),
                                  "total candidate cost", positive=True)
        result.update(candidate_total_cost=candidate_total,
                      baseline_over_candidate=_number(total / candidate_total, "cost ratio", positive=True),
                      saved_cost=total - candidate_total)
    return result


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    document = json.loads(args.input.read_text(), object_pairs_hook=_object)
    print(json.dumps(summarize(document), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
