"""Audit paired teacher-forced token metrics using only the Python standard library.

This checks receipt consistency, not process authenticity or model quality. Pass
prepared token JSONL, token-category JSON, and two evaluation JSON receipts to
``exl3-mlx-audit``. Model, runtime and checkpoint provenance need a separate audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

_ROW_ATOL = 1e-6
_GLOBAL_ATOL = 1e-5
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(value):
    def pairs(items):
        result = {}
        for key, entry in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = entry
        return result

    def invalid(constant):
        raise ValueError(f"Nonfinite JSON value: {constant}")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)


def _mapping(value, name):
    _require(isinstance(value, dict), f"{name}: expected an object")
    return value


def _list(value, name):
    _require(isinstance(value, list), f"{name}: expected an array")
    return value


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name}: invalid integer")
    return value


def _number(value, name):
    _require(_finite(value), f"{name}: nonfinite number")
    return value


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _summary(values):
    count = len(values)
    base = math.fsum(v[0] for v in values)
    adapter = math.fsum(v[1] for v in values)
    return {
        "tokens": count,
        "base_nll_sum": base,
        "adapter_nll_sum": adapter,
        "base_nll": base / count,
        "adapter_nll": adapter / count,
        "nll_delta": (adapter - base) / count,
        "base_correct": sum(v[2] for v in values),
        "adapter_correct": sum(v[3] for v in values),
        "corrected_tokens": sum(not v[2] and v[3] for v in values),
        "regressed_tokens": sum(v[2] and not v[3] for v in values),
    }


def validate_category_selection(prepared, categories):
    """Validate the complete category contract before expensive evaluations."""
    prepared = _list(prepared, "prepared")
    _require(bool(prepared), "Require a nonempty prepared dataset")
    categories = _mapping(categories, "categories")
    rows = _list(categories.get("rows"), "category rows")
    vocab = _integer(categories.get("prediction_vocab_size"), "prediction vocabulary", 1)
    size = _integer(categories.get("tokenizer_size"), "tokenizer size", 1)
    _require(len(rows) == len(prepared), "Category example count mismatch")
    observed = defaultdict(int)
    for i, (p, c) in enumerate(zip(prepared, rows, strict=True)):
        _mapping(p, "prepared row")
        _mapping(c, "category row")
        ids = _list(p.get("input_ids"), "input IDs")
        start = _integer(p.get("loss_start"), "loss_start", 1)
        _require(start < len(ids), "Empty supervised target")
        _require(
            all(type(v) is int and 0 <= v < min(vocab, size) for v in ids),
            "Input token outside vocabulary",
        )
        _require(_integer(c.get("index"), "category index") == i, "Category row mismatch")
        _require(
            _integer(c.get("loss_start"), "category loss_start", 1) == start,
            "Category mask mismatch",
        )
        digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        _require(c.get("input_ids_sha256") == digest, "Category sequence mismatch")
        labels = _list(c.get("categories"), "target categories")
        _require(
            _integer(c.get("supervised_tokens"), "category supervised tokens", 1)
            == len(labels)
            == len(ids) - start,
            "Incomplete category labels",
        )
        for label in labels:
            _require(
                isinstance(label, str)
                and len(label.split("/")) == 3
                and all(part and part.strip() == part for part in label.split("/"))
                and not label.startswith("family:"),
                "Malformed category: expected family/role/type",
            )
            observed[label] += 1
    counts = _mapping(categories.get("counts"), "category counts")
    _require(
        all(type(v) is int and v > 0 for v in counts.values()) and counts == dict(observed),
        "Category counts mismatch",
    )


def compare_token_evaluations(prepared, categories, base, adapter):
    """Validate and compare an arbitrary nonempty set of complete evaluation rows.

    ``prepared`` is a list of ``input_ids``/``loss_start`` rows. Category rows
    bind their index, full-ID SHA256, loss boundary, supervised count and one
    ``family/role/type`` label per target. Evaluation receipts must contain the
    full selected-example hash and per-target NLL, predicted IDs and target IDs.
    NLL results are weighted by target tokens; macro means are separately named.
    """
    prepared = _list(prepared, "prepared")
    count = len(prepared)
    _require(count > 0, "Require a nonempty prepared dataset")
    categories = _mapping(categories, "categories")
    category_rows = _list(categories.get("rows"), "category rows")
    vocab = _integer(categories.get("prediction_vocab_size"), "prediction vocabulary", 1)
    _require(len(category_rows) == count, "Category example count mismatch")
    validate_category_selection(prepared, categories)
    selection = hashlib.sha256(_canonical(prepared)).hexdigest()
    reports = []
    for name, value in (("base", base), ("adapter", adapter)):
        report = _mapping(value, name)
        context = _mapping(report.get("evaluation_context"), f"{name} evaluation context")
        _require(
            context.get("selected_examples_sha256") == selection,
            f"{name}: selected full sequences differ",
        )
        examples = _integer(report.get("examples"), f"{name} examples", 1)
        report_rows = _list(report.get("rows"), f"{name} rows")
        _require(examples == len(report_rows) == count, f"{name}: example count mismatch")
        seq_length = _integer(context.get("seq_length"), f"{name} seq_length", 1)
        _require(
            seq_length >= max(len(p["input_ids"]) - 1 for p in prepared),
            f"{name}: context truncation",
        )
        reports.append(report)
    base, adapter = reports
    groups = defaultdict(list)
    per_row = []
    for i, (p, value, b, a) in enumerate(
        zip(prepared, category_rows, base["rows"], adapter["rows"], strict=True)
    ):
        targets = p["input_ids"][p["loss_start"] :]
        labels = value["categories"]
        for name, r in (("base", b), ("adapter", a)):
            _mapping(r, f"{name} row {i}")
            target_ids = _list(r.get("target_ids"), "target IDs")
            _require(
                all(type(v) is int for v in target_ids) and target_ids == targets,
                "Target tokens differ",
            )
            nll = _list(r.get("token_nll"), "token NLL")
            predictions = _list(r.get("predicted_ids"), "predicted IDs")
            _require(len(nll) == len(predictions) == len(targets), "Incomplete token metrics")
            _require(
                all(_finite(v) and v >= -_ROW_ATOL for v in nll),
                "Invalid token NLL",
            )
            _require(
                all(type(v) is int and 0 <= v < vocab for v in predictions),
                "Invalid predicted IDs",
            )
            if "supervised_tokens" in r:
                _require(
                    _integer(r["supervised_tokens"], "row supervised tokens", 1) == len(targets),
                    "Row supervised count mismatch",
                )
            if "logits_sha256" in r:
                _require(
                    isinstance(r["logits_sha256"], str)
                    and bool(_HEX64.fullmatch(r["logits_sha256"])),
                    "Invalid logits digest",
                )
        local = defaultdict(list)
        for j, (target, label) in enumerate(zip(targets, labels, strict=True)):
            value = (
                b["token_nll"][j],
                a["token_nll"][j],
                b["predicted_ids"][j] == target,
                a["predicted_ids"][j] == target,
            )
            for key in ("all", label, "family:" + label.split("/")[0]):
                local[key].append(value)
                groups[key].append(value)
        result = {key: _summary(values) for key, values in local.items()}
        for r, prefix in ((b, "base"), (a, "adapter")):
            correct = _integer(r.get("correct_tokens"), "correct tokens")
            _require(correct == result["all"][prefix + "_correct"], "Correct count mismatch")
            loss = _number(r.get("loss"), "row loss")
            _require(
                abs(loss - result["all"][prefix + "_nll"]) < _ROW_ATOL,
                "Row NLL mismatch",
            )
        per_row.append({"index": i, "metrics": result})
    total = {key: _summary(values) for key, values in groups.items()}
    for r, prefix in ((base, "base"), (adapter, "adapter")):
        supervised = _integer(r.get("supervised_tokens"), "global supervised tokens", 1)
        _require(supervised == total["all"]["tokens"], "Global token count mismatch")
        loss = _number(r.get("loss"), "global loss")
        _require(
            abs(loss - total["all"][prefix + "_nll"]) < _GLOBAL_ATOL,
            "Global token-weighted loss mismatch",
        )
    macro_base = math.fsum(r["metrics"]["all"]["base_nll"] for r in per_row) / count
    macro_adapter = math.fsum(r["metrics"]["all"]["adapter_nll"] for r in per_row) / count
    return {
        "schema": "paired-token-evaluation-audit-v1",
        "status": "metrics_consistent_pending_provenance_audit",
        "examples": count,
        "selected_examples_sha256": selection,
        "aggregate": total,
        "macro_example_nll": {
            "base": macro_base,
            "adapter": macro_adapter,
            "delta": macro_adapter - macro_base,
        },
        "rows": per_row,
        "quality_certified": False,
        "limits": [
            "Teacher-forced metrics only, not generation quality or inference speed",
            "Requires separate model, runtime, completed-launch and checkpoint provenance audit",
            "Supplied category labels are checked for consistency, not semantic correctness",
        ],
    }


def audit_files(*, prepared, categories, base, adapter, output):
    """Capture input files, audit their content and exclusively publish a JSON result."""
    destination = Path(output).expanduser().absolute()
    _require(not destination.exists(), "Output already exists")
    paths = {
        name: Path(value).expanduser().resolve(strict=True)
        for name, value in {
            "prepared": prepared,
            "categories": categories,
            "base": base,
            "adapter": adapter,
        }.items()
    }
    captured = {name: path.read_bytes() for name, path in paths.items()}
    examples = [_json(line) for line in captured["prepared"].splitlines() if line.strip()]
    result = compare_token_evaluations(
        examples, *(_json(captured[name]) for name in ("categories", "base", "adapter"))
    )
    result["inputs_sha256"] = {
        str(paths[name]): hashlib.sha256(value).hexdigest() for name, value in captured.items()
    }
    result["input_roles"] = {name: str(path) for name, path in paths.items()}
    result["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    for name, path in paths.items():
        _require(path.read_bytes() == captured[name], "Input changed during audit")
    # Same-directory temporary plus exclusive hard-link prevents partial publication/overwrite.
    fd, temp_name = tempfile.mkstemp(prefix=".token-audit-", dir=destination.parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "categories", "base", "adapter", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_files(**vars(args))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Evaluation audit failed: {exc}\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "examples": result["examples"],
                "tokens": result["aggregate"]["all"]["tokens"],
                "quality_certified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
