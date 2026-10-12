"""CPU-only importer contracts with mocked tokenizer helpers; no model allocation."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "src/exl3_mlx/sft_data.py"
_SPEC = importlib.util.spec_from_file_location("exl3_mlx.sft_data_cpu_tests", _PATH)
sft = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sft)


class Tokenizer:
    eos_token_id = 1

    @staticmethod
    def decode(ids):
        parts, raw = [], bytearray()
        for token in ids:
            if token < 16:
                if raw:
                    parts.append(raw.decode("utf-8"))
                    raw.clear()
                parts.append(
                    {0: "<bos>", 1: "<eos>", 2: "<user>", 3: "<assistant>", 4: "<system>"}[token]
                )
            else:
                raw.append(token - 16)
        if raw:
            parts.append(raw.decode("utf-8"))
        return "".join(parts)


def tokenize(row, tokenizer):
    messages = row.get("messages") or [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["completion"]},
    ]
    ids, start = [0], None
    for message in messages:
        ids.append({"user": 2, "assistant": 3, "system": 4}[message["role"]])
        if message["role"] == "assistant":
            start = len(ids)
        ids.extend(byte + 16 for byte in message["content"].encode("utf-8"))
    return {"input_ids": ids + [tokenizer.eos_token_id], "loss_start": start}


def validate(row, *, seq_length, vocab_size, truncate):
    assert truncate is False
    assert 2 <= seq_length <= 32768
    ids, start = row["input_ids"], row["loss_start"]
    if len(ids) > seq_length + 1 or not 1 <= start < len(ids):
        raise ValueError("Invalid token count/mask")
    if any(type(t) is not int or not 0 <= t < vocab_size for t in ids):
        raise ValueError("Invalid token ID")
    return row


@pytest.fixture
def setup(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 300}}))
    tokenizer = Tokenizer()
    monkeypatch.setattr(
        sft,
        "_helpers",
        lambda: (
            lambda _: tokenizer,
            tokenize,
            lambda _: {
                "tokenizer.json": "fixture",
                "tokenizer_config.json": "fixture",
                "chat_template.jinja": "fixture",
            },
            validate,
        ),
    )
    return tmp_path, model


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def row(prompt="Question", answer="Answer", **metadata):
    return {"prompt": prompt, "completion": answer, **metadata}


def run(setup, records, *, output="prepared", **kwargs):
    temp, model = setup
    source = write_rows(temp / "source.jsonl", records)
    return sft.prepare_sft_data(source, model, temp / output, **kwargs)


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_canonical_formats_normalize_to_identical_inputs():
    variants = [
        row("Hi", "Hello"),
        {"question": "Hi", "answer": "Hello"},
        {"instruction": "Hi", "input": "", "output": "Hello"},
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        },
        {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]},
    ]
    ids = []
    for variant in variants:
        canonical, _, _ = sft._normalize(variant, None, None)
        ids.append(tokenize(canonical, Tokenizer()))
    assert all(value == ids[0] for value in ids)


def test_atomic_output_manifest_preview_metadata_and_exact_dedup(setup):
    records = [row(f"Q{i}", f"A{i}", document_id=f"doc{i}", note={"i": i}) for i in range(4)]
    records.append(row("Q0", "A0", source_id="another-source", note="duplicate"))
    report = run(setup, records)
    output = Path(report["output"])
    assert report["counts"] == {"train": 2, "valid": 1, "test": 1}
    assert report["input_rows"] == 5
    assert report["duplicates_removed"] == 1
    assert report["independent_groups"] == 4
    assert report["quality_evaluation_available"] is True
    assert report["evaluation_scope"] == "held_out_test"
    assert report["seq_length"] == report["max_length"] - 1
    assert report["truncated_rows"] == 0
    assert set(report["files"]) == {"train.jsonl", "valid.jsonl", "test.jsonl"}
    assert set(report["artifacts_sha256"]) == {
        "train.jsonl",
        "valid.jsonl",
        "test.jsonl",
        "raw-train.jsonl",
        "raw-valid.jsonl",
        "raw-test.jsonl",
        "provenance.jsonl",
    }
    for name, digest in report["artifacts_sha256"].items():
        assert sft._sha256(output / name) == digest
    assert sft._sha256(output / "dataset.json") == report["manifest_sha256"]
    provenance = read(output / "provenance.jsonl")
    assert provenance[-1]["duplicate_of"] == "row-000001"
    assert provenance[-1]["retained"] is False
    assert provenance[0]["metadata"]["note"] == {"i": 0}
    assert all(
        "note" not in record
        for split in ("train", "valid", "test")
        for record in read(output / f"raw-{split}.jsonl")
    )
    assert all(
        p["supervised"].endswith("<eos>") and "<assistant>" in p["context"]
        for p in report["preview"]
    )


def test_csv_preserves_bom_quoted_multiline_unicode_and_column_mapping(setup):
    temp, model = setup
    source = temp / "input.csv"
    source.write_text(
        '\ufeffrequest,response,doc\n"Say café, please","sí\nexacto",sourceA\n', encoding="utf-8"
    )
    report = sft.prepare_sft_data(
        source,
        model,
        temp / "prepared",
        demo=True,
        prompt_column="request",
        response_column="response",
        group_column="doc",
    )
    assert read(Path(report["output"]) / "raw-train.jsonl") == [
        row("Say café, please", "sí\nexacto")
    ]
    assert report["counts"] == {"train": 1, "valid": 0, "test": 0}
    assert report["evaluation_scope"] == "demonstration"
    assert not report["quality_evaluation_available"]


def test_alpaca_context_and_sharegpt_system_preserved(setup):
    records = [
        {"instruction": "Classify", "input": "A\nB", "output": "two", "system": "Be short"},
        {
            "conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}],
            "system": "Be nice",
        },
    ]
    report = run(setup, records, demo=True)
    raw = read(Path(report["output"]) / "raw-train.jsonl")
    assert raw[0]["messages"][0] == {"role": "system", "content": "Be short"}
    assert raw[0]["messages"][1]["content"] == "Classify\n\nA\nB"
    assert raw[1]["messages"][0]["content"] == "Be nice"


def test_final_assistant_mask_preserves_earlier_context(setup):
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Earlier"},
        {"role": "user", "content": "Second"},
        {"role": "assistant", "content": "Final"},
    ]
    report = run(setup, [{"messages": messages}], demo=True)
    preview = report["preview"][0]
    assert "Earlier" in preview["context"]
    assert preview["supervised"] == "Final<eos>"


@pytest.mark.parametrize(
    "bad",
    [
        {"prompt": "Q", "completion": "A", "messages": []},
        {"prompt": "Q", "completion": "A", "question": "Q"},
        {"prompt": "Q", "completion": " "},
        {"prompt": "Q", "completion": None},
        {"question": "Q"},
        {"instruction": "Q", "input": None, "output": "A"},
        {"prompt": "Q", "completion": "A", "tools": []},
        {"messages": [{"role": "assistant", "content": "A"}]},
        {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": ""}]},
        {"messages": [{"role": "user", "content": "Q"}, {"role": "user", "content": "A"}]},
        {"conversations": [{"from": "tool", "value": "X"}]},
        {"conversations": [{"from": "human", "value": ["image"]}, {"from": "gpt", "value": "A"}]},
    ],
)
def test_invalid_ambiguous_empty_or_tool_records_fail_before_tokenization(setup, monkeypatch, bad):
    def unexpected():
        raise AssertionError("Tokenizer should not be touched")

    monkeypatch.setattr(sft, "_helpers", unexpected)
    with pytest.raises(ValueError):
        run(setup, [bad], demo=True)
    assert not (setup[0] / "prepared").exists()


def test_normal_auto_requires_three_components_demo_never_copies_training(setup):
    with pytest.raises(ValueError, match="independent groups"):
        run(setup, [row("Same", "one"), row("Same", "two"), row("Same", "three")])
    report = run(setup, [row("Same", "one"), row("Same", "two")], demo=True)
    output = Path(report["output"])
    assert output.joinpath("valid.jsonl").read_text() == ""
    assert output.joinpath("test.jsonl").read_text() == ""
    assert report["independent_groups"] == 1


def test_group_prompt_and_duplicate_unions_are_transitive(setup):
    records = [
        row("A", "1", document_id="docA"),
        row("Bridge", "2", document_id="docA"),
        row("Bridge", "3", document_id="docB"),
        row("B", "4", document_id="docB"),
        row("C", "5", document_id="docC"),
        row("D", "6", document_id="docD"),
    ]
    report = run(setup, records)
    provenance = read(Path(report["output"]) / "provenance.jsonl")
    assert report["independent_groups"] == 3
    assert len({record["split"] for record in provenance[:4]}) == 1
    assert len({record["group_id"] for record in provenance[:4]}) == 1


def test_seeded_splitting_reproducible(setup):
    records = [row(f"Q{i}", "Answer") for i in range(15)]
    first = run(setup, records, output="one", seed=42)
    second = run(setup, records, output="two", seed=42)
    assert first["files"] == second["files"]
    assert first["artifacts_sha256"] == second["artifacts_sha256"]


@pytest.mark.parametrize("valid_name", ["valid.jsonl", "validation.json", "valid.csv"])
def test_explicit_train_validation_without_test_preserves_split(setup, valid_name):
    temp, model = setup
    source = temp / "explicit"
    source.mkdir()
    write_rows(source / "train.jsonl", [row("Train", "T")])
    valid = source / valid_name
    if valid.suffix == ".json":
        valid.write_text(json.dumps([row("Valid", "V")]))
    elif valid.suffix == ".csv":
        valid.write_text("prompt,completion\nValid,V\n")
    else:
        write_rows(valid, [row("Valid", "V")])
    report = sft.prepare_sft_data(source, model, temp / "prepared")
    assert report["counts"] == {"train": 1, "valid": 1, "test": 0}
    assert report["evaluation_scope"] == "development"
    assert not report["quality_evaluation_available"]
    assert read(Path(report["output"]) / "raw-valid.jsonl") == [row("Valid", "V")]


def test_explicit_train_only_scope(setup):
    temp, model = setup
    source = temp / "explicit"
    source.mkdir()
    (source / "train.json").write_text(json.dumps([row()]))
    report = sft.prepare_sft_data(source, model, temp / "prepared")
    assert report["evaluation_scope"] == "training_only"
    assert report["counts"] == {"train": 1, "valid": 0, "test": 0}


@pytest.mark.parametrize(
    "train,valid",
    [
        (row("A", "1", document_id="same"), row("B", "2", document_id="same")),
        (row("Same", "1", document_id="first"), row("Same", "2", document_id="other")),
        (row("Same", "1"), row("Same", "1")),
        (row("A", "1", source_id="alias"), row("B", "2", document_id="alias")),
    ],
)
def test_explicit_split_leakage_fails(setup, train, valid):
    temp, model = setup
    source = temp / "explicit"
    source.mkdir()
    write_rows(source / "train.jsonl", [train])
    write_rows(source / "test.jsonl", [valid])
    with pytest.raises(ValueError, match="Explicit split overlap"):
        sft.prepare_sft_data(source, model, temp / "prepared")
    assert not (temp / "prepared").exists()


def test_split_alias_and_extension_ambiguity_rejected(setup):
    temp, model = setup
    source = temp / "explicit"
    source.mkdir()
    write_rows(source / "train.jsonl", [row()])
    write_rows(source / "valid.jsonl", [row("V", "A")])
    (source / "validation.json").write_text("[]")
    with pytest.raises(ValueError, match="Ambiguous valid"):
        sft.prepare_sft_data(source, model, temp / "prepared")


def test_total_length_exact_boundary_and_no_truncation(setup):
    # Fake template is BOS + USER + prompt + ASSISTANT + answer + EOS.
    report = run(setup, [row("Q", "A")], demo=True, max_length=6)
    assert report["max_length"] == 6
    assert report["seq_length"] == 5
    with pytest.raises(ValueError, match="6 serialized tokens exceed max_length=5"):
        run(setup, [row("Q", "A")], output="too-short", demo=True, max_length=5)
    with pytest.raises(ValueError, match="exceed max_length=32768"):
        run(setup, [row("Q", "A" * 32764)], output="too-long", demo=True)


def test_invalid_args_and_explicit_group_missing(setup):
    for kwargs in (
        {"max_length": True},
        {"max_length": 32769},
        {"seed": True},
        {"demo": 1},
        {"prompt_column": "x"},
        {"prompt_column": "prompt", "response_column": "prompt"},
    ):
        with pytest.raises(ValueError):
            run(setup, [row()], **kwargs)
    with pytest.raises(ValueError, match="Missing group"):
        run(setup, [row()], demo=True, group_column="family")


@pytest.mark.parametrize(
    "contents",
    [
        "prompt,prompt,completion\nQ,Q,A\n",
        "prompt,completion\nQ\n",
        "prompt,completion\nQ,A,extra\n",
    ],
)
def test_bad_csv_headers_or_width_fail(setup, contents):
    temp, model = setup
    source = temp / "bad.csv"
    source.write_text(contents)
    with pytest.raises(ValueError):
        sft.prepare_sft_data(source, model, temp / "prepared", demo=True)


def test_duplicate_json_keys_and_nonfinite_metadata_rejected(setup):
    temp, model = setup
    source = temp / "bad.jsonl"
    for contents in (
        '{"prompt":"Q","prompt":"other","completion":"A"}',
        '{"prompt":"Q","completion":"A","metric":NaN}',
    ):
        source.write_text(contents)
        with pytest.raises(ValueError):
            sft.prepare_sft_data(source, model, temp / "prepared", demo=True)


def test_partial_write_cleanup_and_existing_output_untouched(setup, monkeypatch):
    original = sft._write_jsonl
    calls = []

    def fail_second(path, rows):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk failure")
        original(path, rows)

    monkeypatch.setattr(sft, "_write_jsonl", fail_second)
    with pytest.raises(OSError, match="disk failure"):
        run(setup, [row()], demo=True)
    assert not list(setup[0].glob(".sft-data-*"))
    assert not (setup[0] / "prepared").exists()
    (setup[0] / "prepared").mkdir()
    (setup[0] / "prepared/keep").write_text("keep")
    with pytest.raises(FileExistsError):
        run(setup, [row()], demo=True)
    assert (setup[0] / "prepared/keep").read_text() == "keep"


def test_source_mutation_before_publication_rejected(setup, monkeypatch):
    original = sft._write_jsonl
    changed = False

    def mutate_source(path, rows):
        nonlocal changed
        original(path, rows)
        if not changed:
            (setup[0] / "source.jsonl").write_text(json.dumps(row("Changed", "answer")))
            changed = True

    monkeypatch.setattr(sft, "_write_jsonl", mutate_source)
    with pytest.raises(ValueError, match="Source files changed"):
        run(setup, [row()], demo=True)
    assert not (setup[0] / "prepared").exists()
    assert not list(setup[0].glob(".sft-data-*"))


def test_unterminated_csv_is_actionable_value_error(setup):
    temp, model = setup
    source = temp / "bad.csv"
    source.write_text('prompt,completion\n"unterminated,A\n')
    with pytest.raises(ValueError, match="invalid CSV"):
        sft.prepare_sft_data(source, model, temp / "prepared", demo=True)


def test_module_import_is_cpu_only():
    # This module can also be collected beside GPU tests; inspect its imports directly.
    assert sft._helpers.__name__ == "_helpers"
    assert "mlx" not in sft.__dict__
    assert not any(name.startswith("sft_data_cpu_tests.mlx") for name in sys.modules)


def test_infer_long_example_preserves_all_tokens(setup):
    report = run(setup, [row("Q" * 1000, "A" * 1300)], demo=True)
    assert report["max_length"] == 2304
    assert report["seq_length"] == 2303
