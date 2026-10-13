"""CPU-only receipt checks for optional final training-mode fit probes."""

import hashlib
import json
import math
from pathlib import Path

MAX_FINAL_FIT_EXAMPLES = 32
FIT_LOSS_ATOL = 1e-6


def validate_final_fit_examples(value):
    if type(value) is not int or not 0 <= value <= MAX_FINAL_FIT_EXAMPLES:
        raise ValueError("final_fit_examples must be an integer in 0..32 (0 disables the probe)")
    return value


def selection_sha256(rows):
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError("Training fit probe: " + message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def validate_scores(scores, selected):
    """Check complete per-row target alignment before using a fit receipt."""
    _require(isinstance(scores, dict) and bool(selected), "nonempty score receipt required")
    rows = scores.get("rows")
    expected_count = sum(len(row["input_ids"]) - row["loss_start"] for row in selected)
    _require(
        type(scores.get("examples")) is int and scores["examples"] == len(selected),
        "example count differs",
    )
    _require(
        type(scores.get("supervised_tokens")) is int
        and scores["supervised_tokens"] == expected_count,
        "target count differs",
    )
    _require(
        isinstance(rows, list) and len(rows) == len(selected), "complete token metrics required"
    )
    total = 0.0
    for value, example in zip(rows, selected):
        _require(isinstance(value, dict), "invalid row receipt")
        targets = example["input_ids"][example["loss_start"] :]
        _require(
            value.get("target_ids") == targets and all(type(v) is int for v in value["target_ids"]),
            "target IDs differ",
        )
        predictions = value.get("predicted_ids")
        _require(
            isinstance(predictions, list)
            and len(predictions) == len(targets)
            and all(type(token) is int and token >= 0 for token in predictions),
            "predicted IDs are invalid",
        )
        _require(
            type(value.get("supervised_tokens")) is int
            and value["supervised_tokens"] == len(targets),
            "row target count differs",
        )
        nll = value.get("token_nll")
        _require(
            isinstance(nll, list) and len(nll) == len(targets) and all(_finite(v) for v in nll),
            "nonfinite/incomplete token NLL",
        )
        _require(
            _finite(value.get("loss"))
            and abs(value["loss"] - math.fsum(nll) / len(nll)) <= FIT_LOSS_ATOL,
            "row loss differs from token mean",
        )
        digest = value.get("logits_sha256")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            "row logits digest missing",
        )
        total += math.fsum(nll)
    _require(
        _finite(scores.get("loss"))
        and abs(scores["loss"] - total / expected_count) <= FIT_LOSS_ATOL,
        "aggregate loss differs from token mean",
    )
    return rows


def run_final_fit_probe(model, examples, *, count, evaluate, check, metadata):
    """Score current adapters in their training mode without any optimizer call."""
    count = validate_final_fit_examples(count)
    _require(count > 0 and bool(examples), "enabled probe needs examples")
    _require(model.training is True, "final probe requires the current training mode")
    selected = examples[:count]
    before = dict(model.report().get("prefix_cache", {}))
    scores = evaluate(model, selected, check=check, include_token_metrics=True, preserve_mode=True)
    _require(model.training is True, "evaluation changed the model training mode")
    validate_scores(scores, selected)
    after = dict(model.report().get("prefix_cache", {}))
    return {
        "format": "exl3-final-training-fit-probe-v1",
        "scope": "same training rows in training mode; not held-out quality",
        "quality_certified": False,
        **metadata,
        "requested_examples": count,
        "selected_examples_sha256": selection_sha256(selected),
        "training_mode": True,
        "prefix_cache_before": before,
        "prefix_cache_after": after,
        "scores": scores,
    }


def compare_final_fit_probe(probe, fresh, selected_rows, *, checkpoint, expected):
    """Reject mismatches; distinguish numerical parity from proven warm-cache reuse."""
    _require(
        isinstance(probe, dict) and probe.get("format") == "exl3-final-training-fit-probe-v1",
        "unknown receipt format",
    )
    checkpoint = Path(checkpoint).resolve()
    _require(probe.get("checkpoint") == str(checkpoint), "checkpoint differs")
    _require(probe.get("training_mode") is True, "probe did not preserve training mode")
    count = validate_final_fit_examples(probe.get("requested_examples"))
    _require(count > 0, "empty probe")
    _require(count == expected["requested_examples"], "requested probe count differs from project")
    _require(isinstance(probe.get("scores"), dict), "missing scores")
    available = probe["scores"].get("examples")
    _require(
        type(available) is int and 0 < available <= len(selected_rows),
        "fresh evaluation does not cover probe rows",
    )
    selected = selected_rows[:available]
    _require(
        available == min(count, expected["available_examples"]), "probe selection count differs"
    )
    _require(
        probe.get("selected_examples_sha256") == selection_sha256(selected), "selected rows differ"
    )
    for key in (
        "data",
        "input_identity",
        "implementation",
        "seq_length",
        "activation_mode",
        "chunk_size",
    ):
        _require(probe.get(key) == expected[key], f"{key} differs")
    files = probe.get("checkpoint_files_sha256")
    _require(
        isinstance(files, dict)
        and set(files) == {"adapter.safetensors", "adapter_config.json", "trainer_state.json"},
        "missing checkpoint hashes",
    )
    for name, digest in files.items():
        _require(file_sha256(checkpoint / name) == digest, f"checkpoint changed: {name}")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    _require(
        type(probe.get("step")) is int and probe["step"] == state["step"], "checkpoint step differs"
    )
    cold_cache = fresh.get("prefix_cache", {})
    _require(
        cold_cache.get("enabled") is False and cold_cache.get("limit_bytes") == 0,
        "fresh evaluation must disable prefix cache",
    )
    warm_rows = validate_scores(probe["scores"], selected)
    cold_rows = validate_scores(fresh, selected_rows)
    for index, (warm, cold) in enumerate(zip(warm_rows, cold_rows)):
        _require(warm["logits_sha256"] == cold["logits_sha256"], f"row{index} logits mismatch")
        _require(abs(warm["loss"] - cold["loss"]) <= FIT_LOSS_ATOL, f"row{index} loss mismatch")
        _require(
            warm["predicted_ids"] == cold["predicted_ids"], f"row{index} predicted IDs mismatch"
        )
        _require(
            all(abs(a - b) <= FIT_LOSS_ATOL for a, b in zip(warm["token_nll"], cold["token_nll"])),
            f"row{index} token NLL mismatch",
        )
    before, after = probe.get("prefix_cache_before", {}), probe.get("prefix_cache_after", {})
    stats = (before.get("hits"), after.get("hits"), before.get("misses"), after.get("misses"))
    stats_valid = all(type(value) is int and value >= 0 for value in stats)
    all_hits = bool(
        stats_valid
        and before.get("enabled") is True
        and after.get("enabled") is True
        and after.get("hits") - before.get("hits") == available
        and after.get("misses") == before.get("misses")
    )
    return {
        "status": "exact",
        "rows_verified": available,
        "per_row_logits_exact": True,
        "per_row_loss_atol": FIT_LOSS_ATOL,
        "all_probe_rows_were_cache_hits": all_hits,
        "real_model_warm_cache_parity_verified": all_hits,
        "note": "Warm training-mode logits match fresh uncached evaluation."
        if all_hits
        else "Training-mode logits match fresh evaluation, but complete warm-cache reuse was not established.",
        "quality_certified": False,
    }
