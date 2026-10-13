"""Strict local SFT import, grouped splitting, and auditable prepared datasets.

Only tokenization uses MLX-LM's existing data helpers. Importing this module does
not import MLX or allocate a model. ``max_length`` includes every serialized token
(BOS, prompt, answer and EOS); no importer operation truncates a record.
"""

import csv
import hashlib
import json
import math
import random
import shutil
import tempfile
from pathlib import Path

from .context import MAX_CONTEXT_TOKENS
from .tool_data import normalize_tool_row

_GROUP_KEYS = ("source_id", "document_id", "group_id")
_FORBIDDEN = {"tool_calls", "functions", "image", "images", "audio", "video", "videos"}
_SPLITS = ("train", "valid", "test")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _parse_json(value):
    return json.loads(value, object_pairs_hook=_json_pairs, parse_constant=_bad_constant)


def _helpers():
    # Lazy so schema/import tests and help do not require a Metal device.
    from .data import load_tokenizer, tokenize_row, tokenizer_identity, validate_example

    return load_tokenizer, tokenize_row, tokenizer_identity, validate_example


def _string(value, name, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be a {'string' if allow_empty else 'nonempty string'}")
    return value


def _messages(value):
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a nonempty list")
    messages = []
    for index, message in enumerate(value):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(
                f"Message {index + 1}: use role/content text only; tools/images unsupported"
            )
        role = message["role"]
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Message {index + 1}: unsupported role {role!r}")
        messages.append({"role": role, "content": _string(message["content"], "message content")})
    start = 1 if messages[0]["role"] == "system" else 0
    if len(messages) - start < 2 or (len(messages) - start) % 2:
        raise ValueError("Conversations must contain user/assistant pairs and end with an answer")
    for i, message in enumerate(messages[start:]):
        if message["role"] != ("user" if i % 2 == 0 else "assistant"):
            raise ValueError(
                "After an optional initial system message, alternate user and assistant"
            )
    return messages


def _normalize(row, prompt_column, response_column):
    if not isinstance(row, dict) or not row:
        raise ValueError("Each record must be a nonempty object")
    if _FORBIDDEN & set(row):
        raise ValueError("Unsupported top-level tool or multimodal fields")
    tool_row = "tools" in row
    if tool_row and (prompt_column is not None or "messages" not in row):
        raise ValueError("Tool records require explicit messages and declared tools")
    if prompt_column is not None:
        selected = "mapped_prompt_completion"
        used = {prompt_column, response_column}
        normalized = {
            "prompt": _string(row.get(prompt_column), prompt_column),
            "completion": _string(row.get(response_column), response_column),
        }
    else:
        candidates = []
        for name, keys in (
            ("prompt_completion", {"prompt", "completion"}),
            ("question_answer", {"question", "answer"}),
            ("messages", {"messages"}),
            ("alpaca", {"instruction", "output"}),
            ("sharegpt", {"conversations"}),
        ):
            if keys & set(row):
                candidates.append(name)
        if len(candidates) != 1:
            raise ValueError(
                "Ambiguous or unsupported record format; choose explicit prompt/response columns "
                "or use prompt/completion, question/answer, messages, Alpaca, or ShareGPT"
            )
        selected = candidates[0]
        if selected in ("prompt_completion", "question_answer"):
            prompt_key, answer_key = (
                ("prompt", "completion")
                if selected == "prompt_completion"
                else ("question", "answer")
            )
            used = {prompt_key, answer_key}
            normalized = {
                "prompt": _string(row.get(prompt_key), prompt_key),
                "completion": _string(row.get(answer_key), answer_key),
            }
        elif selected == "alpaca":
            used = {"instruction", "input", "output"}
            instruction = _string(row.get("instruction"), "instruction")
            context = _string(row.get("input", ""), "input", allow_empty=True)
            normalized = {
                "prompt": instruction + ("\n\n" + context if context else ""),
                "completion": _string(row.get("output"), "output"),
            }
        elif selected == "messages":
            used = {"messages"}
            if tool_row:
                used.add("tools")
                normalized = normalize_tool_row({"messages": row["messages"], "tools": row["tools"]})
                selected = "terminal_tool_call"
            else:
                normalized = {"messages": _messages(row["messages"])}
        else:
            used = {"conversations"}
            conversation = row["conversations"]
            if not isinstance(conversation, list) or not conversation:
                raise ValueError("ShareGPT conversations must be a nonempty list")
            converted = []
            for message in conversation:
                if not isinstance(message, dict) or set(message) != {"from", "value"}:
                    raise ValueError("ShareGPT messages must contain exactly from/value")
                role = {"human": "user", "gpt": "assistant", "system": "system"}.get(
                    message["from"]
                )
                if role is None:
                    raise ValueError(f"Unsupported ShareGPT role: {message['from']!r}")
                converted.append({"role": role, "content": message["value"]})
            normalized = {"messages": _messages(converted)}
    if "system" in row:
        system = {"role": "system", "content": _string(row["system"], "system")}
        used.add("system")
        messages = normalized.get("messages") or [
            {"role": "user", "content": normalized["prompt"]},
            {"role": "assistant", "content": normalized["completion"]},
        ]
        if messages[0]["role"] == "system":
            raise ValueError("System instructions occur both at top level and inside messages")
        normalized = (
            normalize_tool_row({"messages": [system, *messages], "tools": normalized["tools"]})
            if tool_row else {"messages": _messages([system, *messages])}
        )
    metadata = {key: value for key, value in row.items() if key not in used}
    return normalized, metadata, selected



def tool_row_ends_with_template(row, example, tokenizer):
    if "tools" not in row:
        return example["input_ids"][-1] == tokenizer.eos_token_id
    # Check the entire exact template output, not just an assumed EOS suffix.
    actual = tokenizer.apply_chat_template(row["messages"], tools=row["tools"], return_dict=False)
    return example["input_ids"] == actual

def _read_file_unchecked(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            headers = reader.fieldnames
            if (
                not headers
                or any(not h or not h.strip() for h in headers)
                or len(set(headers)) != len(headers)
            ):
                raise ValueError(f"{path}: CSV needs nonempty, unique headers")
            for number, row in enumerate(reader, 1):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{path}: record {number} has missing or extra CSV fields")
                yield number, row
    elif suffix == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if len(line) > 16 << 20:
                    raise ValueError(f"{path}:{number}: row exceeds 16 MiB")
                if line.strip():
                    try:
                        yield number, _parse_json(line)
                    except (ValueError, TypeError) as error:
                        raise ValueError(f"{path}:{number}: {error}") from error
    elif suffix == ".json":
        value = _parse_json(path.read_text(encoding="utf-8-sig"))
        rows = value if isinstance(value, list) else [value]
        yield from enumerate(rows, 1)
    else:
        raise ValueError(f"Unsupported source extension {suffix!r}; use CSV, JSON or JSONL")


def _read_file(path):
    try:
        yield from _read_file_unchecked(path)
    except (csv.Error, UnicodeError) as error:
        raise ValueError(f"{path}: invalid CSV or UTF-8 input: {error}") from error


def _source_files(source):
    if not source.is_dir():
        if not source.is_file():
            raise FileNotFoundError(source)
        return [(None, source)]
    files = []
    for split, aliases in (
        ("train", ("train",)),
        ("valid", ("valid", "validation")),
        ("test", ("test",)),
    ):
        candidates = [
            source / f"{alias}{suffix}"
            for alias in aliases
            for suffix in (".jsonl", ".json", ".csv")
            if (source / f"{alias}{suffix}").exists()
        ]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous {split} split: use one file/extension, found {candidates}")
        if candidates:
            if not candidates[0].is_file():
                raise ValueError(f"Split is not a regular file: {candidates[0]}")
            files.append((split, candidates[0]))
        elif split == "train":
            raise ValueError(
                "A dataset directory must contain train.jsonl, train.json or train.csv"
            )
    return files


def _group_keys(row, group_column):
    keys = (group_column,) if group_column is not None else _GROUP_KEYS
    groups = []
    for key in keys:
        if key not in row:
            if group_column is not None:
                raise ValueError(f"Missing group column {key!r}")
            continue
        value = row[key]
        if type(value) not in (str, int) or not str(value).strip():
            raise ValueError(f"Group field {key!r} must be a nonempty string or integer")
        groups.append((key, str(value).strip()))
    return groups


class _UnionFind:
    def __init__(self, size):
        self.parents = list(range(size))

    def find(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def _assign_splits(records, *, explicit, seed, demo):
    union, seen = _UnionFind(len(records)), {}
    duplicates, exact = {}, {}
    for index, record in enumerate(records):
        example = record["example"]
        full = tuple(example["input_ids"])
        prompt = full[: example["loss_start"]]
        # The three automatic group fields are aliases. Equal IDs across these
        # fields conservatively join records, even when imported schemas differ.
        edges = [("tokens", full), ("prompt", prompt)] + [
            ("group", value) for _, value in record["groups"]
        ]
        for edge in edges:
            if edge in seen:
                union.union(index, seen[edge])
            else:
                seen[edge] = index
        if full in exact:
            original = exact[full]
            if records[original]["example"]["loss_start"] != example["loss_start"]:
                raise ValueError("Identical tokens have conflicting loss masks")
            duplicates[index] = original
        else:
            exact[full] = index
    components = {}
    for index in range(len(records)):
        components.setdefault(union.find(index), []).append(index)
    assignments = {}
    if explicit:
        for root, members in components.items():
            splits = {records[index]["source_split"] for index in members}
            if len(splits) != 1:
                locations = [records[index]["origin"] for index in members]
                raise ValueError(
                    "Explicit split overlap: shared group, identical prompt, or duplicate tokens "
                    f"across {sorted(splits)} at {locations}"
                )
            assignments[root] = next(iter(splits))
    else:
        roots = list(components)
        if not demo and len(roots) < 3:
            raise ValueError(
                f"Only {len(roots)} independent groups after grouping/deduplication; need 3 for "
                "train/validation/test. Add independent examples or explicitly use demo=True."
            )
        if demo:
            assignments = dict.fromkeys(roots, "train")
        else:
            random.Random(seed).shuffle(roots)
            heldout = max(1, int(len(roots) * 0.1))
            assignments.update(dict.fromkeys(roots[:heldout], "valid"))
            assignments.update(dict.fromkeys(roots[heldout : 2 * heldout], "test"))
            assignments.update(dict.fromkeys(roots[2 * heldout :], "train"))
    for index, record in enumerate(records):
        root = union.find(index)
        record.update(
            split=assignments[root],
            group_id=f"group-{root + 1:06d}",
            duplicate_of=f"row-{duplicates[index] + 1:06d}" if index in duplicates else None,
        )
    return len(components), len(duplicates)


def _stats(values):
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[math.ceil(len(ordered) * 0.5) - 1],
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
    }


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def prepare_sft_data(
    source,
    model,
    output,
    *,
    max_length=None,
    seed=8,
    demo=False,
    prompt_column=None,
    response_column=None,
    group_column=None,
):
    """Import local examples atomically; return the dataset manifest and its hash.

    A file is grouped and split approximately 80/10/10 by group count. A directory
    preserves explicit train/valid(or validation)/test splits, including absent
    optional splits. Demo file imports use training only. CSV row numbers identify
    logical records rather than physical lines when quoted cells contain newlines.
    """
    if max_length is not None and (
        type(max_length) is not int or not 2 <= max_length <= MAX_CONTEXT_TOKENS
    ):
        raise ValueError(
            "max_length must be an integer in 2..32768, including BOS/prompt/answer/EOS"
        )
    if type(seed) is not int or type(demo) is not bool:
        raise ValueError("seed must be an integer and demo must be a boolean")
    if (prompt_column is None) != (response_column is None):
        raise ValueError("Specify both prompt_column and response_column")
    for name, value in (
        ("prompt_column", prompt_column),
        ("response_column", response_column),
        ("group_column", group_column),
    ):
        if value is not None:
            _string(value, name)
    if prompt_column is not None and prompt_column == response_column:
        raise ValueError("Prompt and response columns must differ")
    source, model, output = (Path(p).expanduser().resolve() for p in (source, model, output))
    if output.exists():
        raise FileExistsError(f"Choose a new prepared dataset directory: {output}")
    files = _source_files(source)
    source_hashes = {str(path): _sha256(path) for _, path in files}
    records = []
    for split, path in files:
        for number, row in _read_file(path):
            origin = f"{path}:{number}"
            try:
                normalized, metadata, format_name = _normalize(row, prompt_column, response_column)
                groups = _group_keys(row, group_column)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{origin}: {error}") from error
            records.append(
                {
                    "row_id": f"row-{len(records) + 1:06d}",
                    "origin": origin,
                    "source_file": str(path),
                    "source_row": number,
                    "source_split": split,
                    "normalized": normalized,
                    "metadata": metadata,
                    "input_format": format_name,
                    "groups": groups,
                }
            )
    if not records:
        raise ValueError("Source contains no examples")
    if source.is_dir() and not any(r["source_split"] == "train" for r in records):
        raise ValueError("Explicit training split must be nonempty")
    load_tokenizer, tokenize_row, tokenizer_identity, validate_example = _helpers()
    tokenizer_hashes = tokenizer_identity(model)
    tokenizer = load_tokenizer(model)
    config_hash = _sha256(model / "config.json")
    config = _parse_json((model / "config.json").read_text())
    vocab = config["text_config"]["vocab_size"]
    limit = max_length if max_length is not None else MAX_CONTEXT_TOKENS
    for record in records:
        try:
            example = tokenize_row(record["normalized"], tokenizer)
            total = len(example["input_ids"])
            if total > limit:
                raise ValueError(
                    f"{total} serialized tokens exceed max_length={limit}; shorten this example. "
                    "Import never truncates."
                )
            record["example"] = validate_example(
                example, seq_length=max(2, limit - 1), vocab_size=vocab, truncate=False
            )
            if not tool_row_ends_with_template(record["normalized"], example, tokenizer):
                raise ValueError("Tokenized assistant answer has an invalid template ending")
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{record['origin']}: {error}") from error
    groups, duplicates = _assign_splits(records, explicit=source.is_dir(), seed=seed, demo=demo)
    retained = [r for r in records if r["duplicate_of"] is None]
    split_records = {name: [r for r in retained if r["split"] == name] for name in _SPLITS}
    if not split_records["train"]:
        raise ValueError("Training split is empty after deduplication")
    observed_max = max(len(r["example"]["input_ids"]) for r in retained)
    max_length = max_length if max_length is not None else observed_max
    counts = {name: len(rows) for name, rows in split_records.items()}
    scope = (
        "demonstration"
        if demo
        else "held_out_test"
        if counts["test"]
        else "development"
        if counts["valid"]
        else "training_only"
    )
    preview = []
    for split, rows in split_records.items():
        for record in rows[:2]:
            example = record["example"]
            ids, start = example["input_ids"], example["loss_start"]
            preview.append(
                {
                    "split": split,
                    "row_id": record["row_id"],
                    "source": record["origin"],
                    "context": tokenizer.decode(ids[:start]),
                    "supervised": tokenizer.decode(ids[start:]),
                    "total_tokens": len(ids),
                    "supervised_tokens": len(ids) - start,
                    "input_ids": ids,
                    "loss_start": start,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".sft-data-", dir=output.parent))
    try:
        for split, rows in split_records.items():
            _write_jsonl(stage / f"{split}.jsonl", [r["example"] for r in rows])
            _write_jsonl(stage / f"raw-{split}.jsonl", [r["normalized"] for r in rows])
        provenance = []
        for record in records:
            provenance.append(
                {
                    key: value
                    for key, value in record.items()
                    if key not in ("example", "normalized")
                }
                | {
                    "normalized_sha256": _digest(record["normalized"]),
                    "tokens_sha256": _digest(record["example"]),
                    "retained": record["duplicate_of"] is None,
                }
            )
        _write_jsonl(stage / "provenance.jsonl", provenance)
        hashes = {path.name: _sha256(path) for path in sorted(stage.iterdir())}
        report = {
            "format": "exl3-mlx-dataset-v1",
            "importer": "sft-local-v1",
            "source": str(source),
            "source_files_sha256": source_hashes,
            "source_sha256": _digest(source_hashes),
            "tokenizer": tokenizer_hashes,
            "model_config_sha256": config_hash,
            "max_length": max_length,
            "seq_length": max(2, max_length - 1),
            "length_definition": "total serialized tokens including BOS, prompt and complete template-terminated assistant target",
            "observed_max_length": observed_max,
            "seed": seed,
            "demo": demo,
            "counts": counts,
            "input_rows": len(records),
            "independent_groups": groups,
            "duplicates_removed": duplicates,
            "truncated_rows": 0,
            "group_counts": {
                name: len({r["group_id"] for r in rows}) for name, rows in split_records.items()
            },
            "supervised_tokens": {
                name: sum(len(r["example"]["input_ids"]) - r["example"]["loss_start"] for r in rows)
                for name, rows in split_records.items()
            },
            "length_stats": _stats([len(r["example"]["input_ids"]) for r in retained]),
            "mask": "last assistant answer/tool-call turn including exact template termination; prior turns and schemas are context",
            "terminal_tool_call_rows": sum(r["input_format"] == "terminal_tool_call" for r in retained),
            "split_policy": "preserved explicit splits"
            if source.is_dir()
            else "training-only demonstration"
            if demo
            else "seeded 80/10/10 by connected source/prompt/duplicate group count",
            "evaluation_scope": scope,
            "quality_evaluation_available": bool(counts["test"]) and not demo,
            "quality_note": "Independent held-out data is available; model quality is not certified."
            if counts["test"] and not demo
            else "No independent final quality evaluation is available; do not claim model quality from this dataset.",
            "conversion": {
                "prompt_column": prompt_column,
                "response_column": response_column,
                "group_column": group_column,
                "alpaca_instruction_input_separator": "\n\n",
            },
            "files": {f"{split}.jsonl": hashes[f"{split}.jsonl"] for split in _SPLITS},
            "artifacts_sha256": hashes,
            "preview": preview,
        }
        (stage / "dataset.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if source_hashes != {str(path): _sha256(path) for _, path in files}:
            raise ValueError("Source files changed during preparation")
        if tokenizer_identity(model) != tokenizer_hashes:
            raise ValueError("Tokenizer files changed during preparation")
        if _sha256(model / "config.json") != config_hash:
            raise ValueError("Model config changed during preparation")
        if output.exists():
            raise FileExistsError(f"Output appeared during preparation: {output}")
        stage.rename(output)
        return {
            **report,
            "output": str(output),
            "manifest_sha256": _sha256(output / "dataset.json"),
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage)
