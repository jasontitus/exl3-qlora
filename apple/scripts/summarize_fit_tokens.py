"""Summarize all predeclared rows and diff-token categories of a fitting run.

This describes teacher-forced predictions: preceding answer tokens are supplied.
It cannot establish independent patch generation, correctness or speedup.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

LOSS_MEAN_ATOL = 1e-6  # FP32 chunk sums vs Python sum of per-token FP32 values.


def validate_score(score):
    targets = score["target_ids"]
    count = score["supervised_tokens"]
    if (
        type(count) is not int
        or count < 1
        or not isinstance(targets, list)
        or count != len(targets)
    ):
        raise ValueError("Supervised token count must equal the nonempty target array")
    for name in ("target_ids", "predicted_ids"):
        if (
            not isinstance(score[name], list)
            or len(score[name]) != count
            or any(type(value) is not int or value < 0 for value in score[name])
        ):
            raise ValueError("Target/prediction IDs must be complete nonnegative integer arrays")
    if not isinstance(score["token_nll"], list) or len(score["token_nll"]) != count:
        raise ValueError("Incomplete per-token scores")
    values = [score["loss"], *score["token_nll"]]
    if any(type(value) not in (float, int) or not math.isfinite(value) for value in values):
        raise ValueError("All scalar and per-token losses must be finite numbers")
    if abs(score["loss"] - math.fsum(score["token_nll"]) / count) > LOSS_MEAN_ATOL:
        raise ValueError("Scalar loss disagrees with per-token mean")


def summarize(report, categories):
    if report.get("status") != "completed":
        raise ValueError(
            "Only completed runs may produce an accepted fit summary; preserve partial reports for status"
        )
    if not report.get("history"):
        raise ValueError("A completed fit summary needs at least one optimizer update")
    selected_step = report["history"][-1].get("step")
    if type(selected_step) is not int or selected_step < 1:
        raise ValueError("Selected optimizer step is invalid")
    if (
        report["settings"].get("verify_uncached_final")
        and report.get("uncached_final_parity_exact") is not True
    ):
        raise ValueError("Requested uncached final parity is not verified")
    before = report["base_scores"]
    after = report["history"][-1]["post_update_scores"]
    if report.get("uncached_final_parity_exact") is True:
        uncached = report.get("uncached_final_scores")
        required = ("loss", "supervised_tokens", "logits_sha256")
        if (
            not isinstance(uncached, dict)
            or set(uncached) != set(after)
            or any(
                not isinstance(score, dict)
                or any(name not in score for name in required)
                or any(after[key].get(name) != value for name, value in score.items())
                for key, score in uncached.items()
            )
        ):
            raise ValueError("Uncached final scores do not match the selected cached scores")
    if not before or set(before) != set(after) or set(before) != set(categories):
        raise ValueError("Provide categories for every evaluated row")
    rows = {}
    for key, base in before.items():
        trained, labels = after[key], categories[key]
        validate_score(base)
        validate_score(trained)
        if labels.get("source_sha256", {}).get("prepared") != report["binding"]["data_sha256"]:
            raise ValueError("Category prepared-data SHA differs from the run")
        targets = base["target_ids"]
        if trained["target_ids"] != targets or labels["supervised_token_ids"] != targets:
            raise ValueError("Target IDs differ between scores and categories")
        classes = labels["supervised_token_categories"]
        if (
            not isinstance(classes, list)
            or len(classes) != len(targets)
            or any(not isinstance(value, str) or not value or value == "all" for value in classes)
        ):
            raise ValueError("Category count differs from targets")
        if any(
            len(score[name]) != len(targets)
            for score in (base, trained)
            for name in ("token_nll", "predicted_ids")
        ):
            raise ValueError("Incomplete per-token scores")
        groups = {}
        for category in ["all", *sorted(set(classes))]:
            selected = [
                i for i, name in enumerate(classes) if category == "all" or name == category
            ]
            metrics = {}
            for name, score in (("base", base), ("trained", trained)):
                metrics[name] = {
                    "loss": sum(score["token_nll"][i] for i in selected) / len(selected),
                    "correct_tokens": sum(
                        score["predicted_ids"][i] == targets[i] for i in selected
                    ),
                }
            groups[category] = {
                "count": len(selected),
                **metrics,
                "loss_delta": metrics["trained"]["loss"] - metrics["base"]["loss"],
                "correct_token_delta": metrics["trained"]["correct_tokens"]
                - metrics["base"]["correct_tokens"],
            }
        rows[key] = {
            "categories": groups,
            "logits_changed": base["logits_sha256"] != trained["logits_sha256"],
            "added_code_predictions_corrected": [
                {
                    "answer_token_index": i,
                    "target_id": gold,
                    "base_prediction_id": base["predicted_ids"][i],
                }
                for i, gold in enumerate(targets)
                if classes[i] == "added_code"
                and base["predicted_ids"][i] != gold
                and trained["predicted_ids"][i] == gold
            ],
            "added_code_predictions_regressed": [
                {
                    "answer_token_index": i,
                    "target_id": gold,
                    "trained_prediction_id": trained["predicted_ids"][i],
                }
                for i, gold in enumerate(targets)
                if classes[i] == "added_code"
                and base["predicted_ids"][i] == gold
                and trained["predicted_ids"][i] != gold
            ],
        }
    total = sum(v["supervised_tokens"] for v in before.values())
    return {
        "scope": "training-fit teacher-forced evaluation; preceding answer tokens supplied",
        "quality_certified": False,
        "run_status": report["status"],
        "selected_step": selected_step,
        "uncached_final_parity_exact": report.get("uncached_final_parity_exact") is True,
        "scalar_token_mean_atol": LOSS_MEAN_ATOL,
        "training_row_indices": report["settings"]["rows"],
        "optimizer_updates_this_run": len(report["history"]),
        "evaluated_rows": len(rows),
        "supervised_tokens": total,
        "weighted": {
            name: sum(v["loss"] * v["supervised_tokens"] for v in values.values()) / total
            for name, values in (("base_loss", before), ("trained_loss", after))
        },
        "macro": {
            name: sum(v["loss"] for v in values.values()) / len(values)
            for name, values in (("base_loss", before), ("trained_loss", after))
        },
        "every_row_improved": all(after[k]["loss"] < before[k]["loss"] for k in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--categories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    categories = [json.loads(p.read_text()) for p in args.categories]
    mapped = {str(c["row_index"]): c for c in categories}
    if len(mapped) != len(categories):
        parser.error("Duplicate category row indices")
    result = summarize(json.loads(args.run.read_text()), mapped)
    result["source_hashes"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (args.run, *args.categories, Path(__file__))
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("rows", "source_hashes")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
