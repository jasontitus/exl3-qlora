"""CPU-only inputs for project paired-token auditing."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from .evaluation_audit import _json, validate_category_selection


def snapshot_audit_inputs(model, selected, directory, categories=None):
    """Snapshot exact selection and local tokenizer metadata; never load model tensors.

    Caller must verify its project-bound metadata before and after this operation.
    Only tokenizer-declared EOS IDs receive an EOS label; generation stop IDs do not.
    """
    from .data import load_tokenizer

    model, directory = Path(model), Path(directory)
    config = _json((model / "config.json").read_bytes())
    vocab = config["text_config"]["vocab_size"]
    tokenizer = load_tokenizer(model)  # local_files_only=True, no remote code
    size = len(tokenizer)
    if type(vocab) is not int or vocab <= 0 or type(size) is not int or size <= 0:
        raise ValueError("Invalid model/tokenizer vocabulary dimensions")
    eos = tokenizer.eos_token_id
    eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
    if any(type(value) is not int or not 0 <= value < min(vocab, size) for value in eos):
        raise ValueError("Invalid tokenizer EOS contract")
    eos = sorted(set(eos))
    captured = None
    if categories is not None:
        source = Path(categories).expanduser().resolve(strict=True)
        payload = source.read_bytes()
        labels = _json(payload)
        if not isinstance(labels, dict):
            raise ValueError("Categories must be a JSON object")
        if labels.get("prediction_vocab_size") != vocab or labels.get("tokenizer_size") != size:
            raise ValueError("Category vocabulary differs from project model/tokenizer")
        captured = (source, payload)
        origin = {
            "kind": "supplied_categories_not_semantically_certified",
            "path": str(source),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    else:
        rows, counts = [], Counter()
        for i, row in enumerate(selected):
            ids, start = row["input_ids"], row["loss_start"]
            values = [
                "generic/observed_id/eos" if token in eos else "generic/observed_id/supervised"
                for token in ids[start:]
            ]
            counts.update(values)
            rows.append(
                {
                    "index": i,
                    "input_ids_sha256": hashlib.sha256(
                        json.dumps(ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "loss_start": start,
                    "supervised_tokens": len(values),
                    "categories": values,
                }
            )
        labels = {
            "prediction_vocab_size": vocab,
            "tokenizer_size": size,
            "rows": rows,
            "counts": dict(counts),
        }
        payload = (json.dumps(labels, indent=2, allow_nan=False) + "\n").encode()
        origin = {
            "kind": "observed_token_id_categories",
            "tokenizer_eos_ids": eos,
            "note": "Token ID identity only; no inferred code, rationale, or correctness labels.",
        }
    validate_category_selection(selected, labels)
    prepared_path, labels_path = (
        directory / "audit-selected.jsonl",
        directory / "audit-categories.json",
    )
    with prepared_path.open("x") as stream:
        for row in selected:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    with labels_path.open("xb") as stream:
        stream.write(payload)
    return prepared_path, labels_path, origin, captured


def verify_category_source(captured):
    if captured is not None and captured[0].read_bytes() != captured[1]:
        raise ValueError("Category source changed during comparison")
