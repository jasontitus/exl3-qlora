"""MLX-LM data formats, with explicit boundaries and auditable loss masks."""

import hashlib
import json
import random
from pathlib import Path

from mlx_lm.tuner.datasets import ChatDataset, CompletionsDataset, TextDataset
from transformers import PreTrainedTokenizerFast

from .context import MAX_CONTEXT_TOKENS

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tokenizer_identity(root):
    return {name: sha256(Path(root) / name) for name in TOKENIZER_FILES}


def load_tokenizer(root):
    # This is a tokenizer-only local load, without remote code or model creation.
    return PreTrainedTokenizerFast.from_pretrained(str(root), local_files_only=True)


def training_data_identity(path, root):
    identity = {"data_sha256": sha256(path)}
    # Raw chat can tokenize differently if only its Jinja template changes. Raw
    # file bytes and tokenizer.json alone do not establish exact continuation.
    if any("input_ids" not in row for _, row in read_jsonl(path)):
        identity["raw_data_tokenizer"] = tokenizer_identity(root)
    return identity


def read_jsonl(path):
    with Path(path).open() as f:
        for number, line in enumerate(f, 1):
            if len(line) > 16 << 20:
                raise ValueError(f"{path}:{number}: row exceeds 16 MiB")
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                yield number, row


def validate_example(row, *, seq_length, vocab_size, truncate=False):
    if not 2 <= seq_length <= MAX_CONTEXT_TOKENS:
        raise ValueError("seq_length must be in 2..32768")
    if set(row) - {"input_ids", "loss_start"}:
        raise ValueError("Prepared examples use input_ids and optional loss_start only")
    ids, start = row.get("input_ids"), row.get("loss_start", 1)
    if (
        not isinstance(ids, list)
        or len(ids) < 2
        or any(type(t) is not int or not 0 <= t < vocab_size or t == 129264 for t in ids)
    ):
        raise ValueError("Invalid text token IDs")
    if type(start) is not int or not 1 <= start < len(ids):
        raise ValueError("loss_start must name the first supervised token (1..len-1)")
    if len(ids) > seq_length + 1:
        if not truncate:
            raise ValueError(
                f"{len(ids) - 1} targets exceed seq_length={seq_length}; use --truncate"
            )
        ids = ids[: seq_length + 1]
    if start >= len(ids):
        raise ValueError("Truncation removed all supervised targets; shorten the prompt")
    return {"input_ids": ids, "loss_start": start}


def tokenize_row(row, tokenizer, *, all_tokens=False):
    if "input_ids" in row:
        return row
    if set(row) == {"text"} and isinstance(row["text"], str) and row["text"].strip():
        ids, _ = TextDataset([], tokenizer).process(row)
        if ids[0] != tokenizer.bos_token_id:
            ids.insert(0, tokenizer.bos_token_id)
        return {"input_ids": ids, "loss_start": 1}
    if set(row) == {"prompt", "completion"} and all(isinstance(v, str) for v in row.values()):
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
        dataset = CompletionsDataset([], tokenizer, "prompt", "completion", not all_tokens)
    elif set(row) == {"messages"} and isinstance(row["messages"], list):
        messages = row["messages"]
        dataset = ChatDataset([], tokenizer, mask_prompt=not all_tokens)
    else:
        raise ValueError("Use text, prompt/completion, messages, or input_ids/loss_start")
    if (
        not messages
        or messages[-1].get("role") != "assistant"
        or any(
            set(m) != {"role", "content"}
            or m["role"] not in ("system", "user", "assistant")
            or not isinstance(m["content"], str)
            for m in messages
        )
    ):
        raise ValueError(
            "Use text messages ending in an assistant answer; tools/images unsupported"
        )
    ids, offset = dataset.process(row)
    if not all_tokens:
        prefix = tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, return_dict=False
        )
        if ids[:offset] != prefix:
            raise ValueError("Chat template prefix is inconsistent; supply explicit IDs/loss_start")
    return {"input_ids": ids, "loss_start": max(1, offset)}


def load_dataset(path, root, *, seq_length, truncate=False):
    path, root = Path(path), Path(root)
    manifest_path = path.parent / "dataset.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "exl3-mlx-dataset-v1":
            raise ValueError("Unsupported prepared dataset format")
        if manifest["tokenizer"] != tokenizer_identity(root):
            raise ValueError("Dataset tokenizer identity mismatch")
        if path.name not in manifest["files"] or sha256(path) != manifest["files"][path.name]:
            raise ValueError("Prepared dataset checksum mismatch")
    vocab = json.loads((root / "config.json").read_text())["text_config"]["vocab_size"]
    tokenizer = None
    examples = []
    for number, row in read_jsonl(path):
        try:
            if "input_ids" not in row:
                tokenizer = tokenizer or load_tokenizer(root)
                row = tokenize_row(row, tokenizer)
            examples.append(
                validate_example(row, seq_length=seq_length, vocab_size=vocab, truncate=truncate)
            )
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    if not examples:
        raise ValueError(f"Empty dataset: {path}")
    return examples


def prepare(args):
    import shutil
    import tempfile

    root, output = Path(args.model).expanduser(), Path(args.output).expanduser()
    if output.exists():
        raise FileExistsError(f"Choose a new dataset directory: {output}")
    if not 0 <= args.valid_fraction < 1 or not 0 <= args.test_fraction < 1:
        raise ValueError("Split fractions must be in [0,1)")
    if args.valid_fraction + args.test_fraction >= 1:
        raise ValueError("Reserve a nonempty training split")
    tokenizer = load_tokenizer(root)
    vocab = json.loads((root / "config.json").read_text())["text_config"]["vocab_size"]
    rows, seen, duplicates, truncated = [], set(), 0, 0
    for number, row in read_jsonl(args.data):
        try:
            tokenized = tokenize_row(row, tokenizer, all_tokens=args.all_tokens)
            truncated += len(tokenized["input_ids"]) > args.seq_length + 1
            example = validate_example(
                tokenized, seq_length=args.seq_length, vocab_size=vocab, truncate=args.truncate
            )
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"Line {number}: {error}") from error
        # Deduplicate before splitting, including duplicates produced by truncation.
        key = tuple(example["input_ids"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        rows.append(example)
    random.Random(args.seed).shuffle(rows)
    nvalid = max(1, int(len(rows) * args.valid_fraction)) if args.valid_fraction else 0
    ntest = max(1, int(len(rows) * args.test_fraction)) if args.test_fraction else 0
    if len(rows) <= nvalid + ntest:
        raise ValueError("Not enough unique examples for nonempty requested splits")
    splits = {
        "valid": rows[:nvalid],
        "test": rows[nvalid : nvalid + ntest],
        "train": rows[nvalid + ntest :],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".dataset-", dir=output.parent))
    try:
        for name, examples in splits.items():
            (stage / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in examples))
        report = {
            "format": "exl3-mlx-dataset-v1",
            "tokenizer": tokenizer_identity(root),
            "source_sha256": sha256(args.data),
            "seq_length": args.seq_length,
            "seed": args.seed,
            "duplicates_removed": duplicates,
            "truncated_rows": truncated,
            "mask": "all tokens" if args.all_tokens else "last assistant answer for chat",
            "counts": {k: len(v) for k, v in splits.items()},
            "supervised_tokens": {
                k: sum(len(r["input_ids"]) - r["loss_start"] for r in v) for k, v in splits.items()
            },
            "files": {f"{k}.jsonl": sha256(stage / f"{k}.jsonl") for k in splits},
        }
        (stage / "dataset.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return report
