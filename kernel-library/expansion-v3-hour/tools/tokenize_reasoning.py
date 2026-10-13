"""Tokenize a bound reviewed reasoning export; never run models or source code."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(W / "loss-fix/apple/src"), str(Path(__file__).resolve().parent)]
import reasoning_data
from exl3_mlx.data import (
    load_tokenizer,
    tokenize_row,
    tokenizer_identity,
    validate_example,
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()


def verify_export(raw_path):
    """Match serialized examples to the reviewed, hash-bound record inputs."""
    p = Path(raw_path).resolve()
    if not p.is_relative_to(W / "kernel-library"):
        raise ValueError("Reasoning export outside library")
    audit_path = p.parent / "audit.json"
    provenance_path = p.parent / "provenance.jsonl"
    raw = p.read_bytes()
    audit_raw, provenance_raw = audit_path.read_bytes(), provenance_path.read_bytes()
    audit = json.loads(audit_raw)
    inputs = audit.get("inputs_sha256")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Reviewed export bindings missing")
    hashes = {
        str(p): sha(raw),
        str(audit_path): sha(audit_raw),
        str(provenance_path): sha(provenance_raw),
    }
    authorized = {}
    for filename, expected in inputs.items():
        source = Path(filename).resolve()
        if not source.is_relative_to(W / "kernel-library"):
            raise ValueError("Export binding outside library")
        data = source.read_bytes()
        if sha(data) != expected:
            raise ValueError("Reviewed export input mismatch")
        hashes[str(source)] = expected
        if source.suffix == ".json":
            value = json.loads(data)
            records = (
                value
                if isinstance(value, list)
                else value.get("records", [])
                if isinstance(value, dict)
                else []
            )
            for record in records:
                if isinstance(record, dict) and {
                    "id",
                    "prompt",
                    "completion",
                    "source_evidence",
                    "rationale_basis",
                } <= set(record):
                    key = record["id"]
                    if key in authorized and canonical(authorized[key]) != canonical(
                        record
                    ):
                        raise ValueError("Conflicting reviewed record identity")
                    authorized[key] = record
    provenance = [
        json.loads(line)
        for line in provenance_raw.decode().splitlines()
        if line.strip()
    ]
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if (
        type(audit.get("records")) is not int
        or len(provenance) != audit["records"]
        or len(rows) != len(provenance)
    ):
        raise ValueError("Export record count mismatch")
    reserved = reasoning_data.reserved_ids()
    seen = set()
    for row, record in zip(rows, provenance, strict=True):
        key = record["id"]
        if (
            key in seen
            or key not in authorized
            or canonical(record) != canonical(authorized[key])
        ):
            raise ValueError("Provenance is not an authorized reviewed record")
        seen.add(key)
        hashes.update(reasoning_data.validate(record, reserved))
        if p.name == "train.messages.jsonl":
            expected = {
                "messages": [
                    {"role": "user", "content": record["prompt"]},
                    {"role": "assistant", "content": record["completion"]},
                ]
            }
        elif p.name == "train.prompt-completion.jsonl":
            expected = {"prompt": record["prompt"], "completion": record["completion"]}
        else:
            raise ValueError("Unsupported reviewed export filename")
        if row != expected:
            raise ValueError("Raw example differs from reviewed positive payload")
    return rows, raw, hashes


def prepare(raw_path, model, output, limit):
    out = Path(output)
    if out.exists():
        raise ValueError("Refuse existing output")
    if type(limit) is not int or not 3 <= limit <= 32768:
        raise ValueError("Unsupported preparation capacity")
    rows, raw, hashes = verify_export(raw_path)
    paths = [
        Path(__file__).resolve(),
        Path(reasoning_data.__file__).resolve(),
        *list((W / "loss-fix/apple/src/exl3_mlx").glob("*.py")),
    ]
    hashes.update({str(p): sha(p.read_bytes()) for p in paths})
    identity = tokenizer_identity(model)
    config_path = Path(model).expanduser().resolve() / "config.json"
    config_raw = config_path.read_bytes()
    hashes[str(config_path)] = sha(config_raw)
    config = json.loads(config_raw)
    vocab = config.get("text_config", config)["vocab_size"]
    if type(vocab) is not int or vocab <= 0:
        raise ValueError("Invalid model vocabulary")
    tok = load_tokenizer(model)
    accepted, stats, excluded = [], [], []
    for index, row in enumerate(rows):
        prepared = tokenize_row(row, tok)
        n = len(prepared["input_ids"])
        stat = {
            "row": index,
            "serialized_tokens": n,
            "supervised_tokens": n - prepared["loss_start"],
            "canonical_raw_sha256": sha(canonical(row)),
        }
        if n > limit:
            excluded.append(
                {**stat, "reason": "complete_item_over_context_no_truncation"}
            )
            continue
        validate_example(
            prepared, seq_length=limit - 1, vocab_size=vocab, truncate=False
        )
        accepted.append(prepared)
        stats.append(stat)
    if (
        any(sha(Path(p).read_bytes()) != digest for p, digest in hashes.items())
        or tokenizer_identity(model) != identity
    ):
        raise ValueError("Input/tokenizer/implementation drift")
    out.mkdir(parents=True)
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in accepted))
    lengths = sorted(s["serialized_tokens"] for s in stats)

    def pct(q):
        return (
            lengths[min(len(lengths) - 1, int(q * (len(lengths) - 1)))]
            if lengths
            else None
        )

    report = {
        "raw": str(Path(raw_path).resolve()),
        "raw_sha256": sha(raw),
        "tokenizer": identity,
        "inputs_sha256": hashes,
        "accepted": len(stats),
        "excluded": excluded,
        "truncated": 0,
        "capacity": limit,
        "rows": stats,
        "supervised_tokens": sum(s["supervised_tokens"] for s in stats),
        "lengths": {
            "p50": pct(0.5),
            "p90": pct(0.9),
            "p99": pct(0.99),
            "max": max(lengths) if lengths else None,
        },
        "scope": "Bound reviewed historical diagnostic text with verified tokenizer/mask; no model training or demonstrated reasoning gain",
    }
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=12288)
    args = parser.parse_args()
    result = prepare(args.raw, args.model, args.output, args.limit)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"rows", "tokenizer", "inputs_sha256"}
            },
            indent=2,
        )
    )
