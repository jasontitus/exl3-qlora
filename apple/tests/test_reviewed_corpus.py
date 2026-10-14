"""Portable reviewed intake through actual local tokenizer and standard project."""

import hashlib
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from exl3_mlx import reviewed_corpus as intake
from exl3_mlx.agent_edits import TOOL, full_commit_images, render_command, replay, unified_patch
from exl3_mlx.data import tokenize_row, tokenizer_identity
from exl3_mlx.sft_data import _normalize


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(intake.canonical(value) + b"\n")
    return {"path": str(path), "sha256": intake.sha(path.read_bytes())}


def seal(root, manifest):
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = intake.sha(intake.canonical(manifest))
    write(root / "corpus.json", manifest)


def blob(text, mode="100644"):
    raw = text.encode()
    return {
        "content": text,
        "mode": mode,
        "sha256": intake.sha(raw),
        "blob": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
    }


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    vocab = {chr(i): i for i in range(256)}
    vocab.update({"[UNK]": 256, "[EOS]": 257})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.chat_template = '{% for m in messages %}{{ m.role }}:{{ m.content or "" }}{% if m.tool_calls %}{{ m.tool_calls|tojson }}{% endif %}{% if m.role == "assistant" %}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}'
    tokenizer.save_pretrained(model)
    write(
        model / "config.json",
        {
            "model_type": "deepseek_v41",
            "quantization": {"mode": "mxfp8"},
            "text_config": {"num_hidden_layers": 40, "index_topk": 512, "vocab_size": 258},
        },
    )
    for name in (
        "expert-manifest-v2.json",
        "model.safetensors.index.json",
        "engram/engram-manifest.json",
        "engram-token-map.u32.json",
        "engram/engram-residents.safetensors",
    ):
        write(model / name, {})
    source = {
        "repository": "https://github.com/example/kernels",
        "head": {"commit": "1" * 40},
        "commits": ["1" * 40, "2" * 40],
        "lineage_keys": ["change:one"],
        "protected": False,
        "files": {
            "kernel.metal": {
                "role": "changed",
                "before": blob("float x = 1;"),
                "after": blob("float x = 2;"),
            },
            "run.sh": {
                "role": "changed",
                "before": blob("echo old\n", "100755"),
                "after": blob("echo new\n", "100755"),
            },
        },
    }
    before, after = full_commit_images(source)
    cmd = render_command(unified_patch(before, after))
    raw = {
        "tools": [TOOL],
        "messages": [
            {
                "role": "user",
                "content": "Make this explicit kernel and host edit. float x = 1; echo old",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "edit1",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": {"cmd": cmd}},
                    }
                ],
            },
        ],
    }
    raw = json.loads(intake.canonical(raw))  # Tokenize the exact reviewed serialization.
    normalized, _, _ = _normalize(raw, None, None)
    prepared = tokenize_row(normalized, tokenizer)
    vals = {
        "source": source,
        "raw": raw,
        "prepared": prepared,
        "replay": replay(cmd, before, after),
    }
    bindings = {}
    for role, value in vals.items():
        path = root / f"{role}.json"
        write(path, value)
        bindings[role] = {"path": path.name, "sha256": intake.sha(path.read_bytes())}
    review = {
        "format": "exact-edit-review-v1",
        "status": "approved",
        "reviewer": "fixture reviewer assertion",
        "cases": [{"id": "one", "inputs": bindings}],
    }
    write(root / "review.json", review)
    policy = {
        k: []
        for k in (
            "protected_commits",
            "protected_patch_ids",
            "protected_lineages",
            "heldout_repositories",
            "lineage_aliases",
        )
    }
    write(root / "policy.json", policy)
    for name in ("valid.jsonl", "valid.prepared.jsonl", "test.jsonl", "test.prepared.jsonl"):
        (root / name).write_bytes(b"")

    def bind(name):
        return {"path": name, "sha256": intake.sha((root / name).read_bytes())}

    manifest = {
        "format": intake.FORMAT,
        "max_length": 4096,
        "tokenizer": tokenizer_identity(model),
        "model_config_sha256": intake.sha((model / "config.json").read_bytes()),
        "policy": bind("policy.json"),
        "entries": [
            {
                "id": "one",
                "kind": "exact_edit",
                "split": "train",
                "inputs": bindings,
                "review": bind("review.json"),
            }
        ],
        "frozen_splits": {
            s: {"raw": bind(s + ".jsonl"), "prepared": bind(s + ".prepared.jsonl")}
            for s in ("valid", "test")
        },
    }
    seal(root, manifest)
    return root, model, manifest, vals


def test_cli_to_standard_project_and_frozen_bytes(bundle, tmp_path):
    from exl3_mlx.ktune_cli import main

    root, model, manifest, vals = bundle
    out, project = tmp_path / "out", tmp_path / "project"
    assert (
        main(
            [
                "prepare-reviewed",
                "--source",
                str(root / "corpus.json"),
                "--model",
                str(model),
                "--output",
                str(out),
                "--project",
                str(project),
            ]
        )
        == 0
    )
    for split in ("valid", "test"):
        for suffix in (".jsonl", ".prepared.jsonl"):
            assert (out / (split + suffix)).read_bytes() == (root / (split + suffix)).read_bytes()
    assert intake.lines((project / "data/train.jsonl").read_bytes()) == [vals["prepared"]]
    report = json.loads((project / "project.json").read_bytes())
    assert report["provenance_binding"]["state"] == "complete"
    assert not (project / "attempts").exists()


@pytest.mark.parametrize(
    "mutation", ["diagnostic", "oversize", "protected", "review", "token", "frozen", "duplicate"]
)
def test_rejects_entire_import(bundle, tmp_path, mutation):
    root, model, m, vals = bundle
    if mutation == "diagnostic":
        m["entries"][0]["kind"] = "diagnostic"
    elif mutation == "oversize":
        m["max_length"] = 3
    elif mutation == "protected":
        p = json.loads((root / "policy.json").read_bytes())
        p["protected_commits"] = ["1" * 40]
        write(root / "policy.json", p)
        m["policy"]["sha256"] = intake.sha((root / "policy.json").read_bytes())
    elif mutation == "review":
        p = json.loads((root / "review.json").read_bytes())
        p["status"] = "pending"
        write(root / "review.json", p)
        m["entries"][0]["review"]["sha256"] = intake.sha((root / "review.json").read_bytes())
    elif mutation == "token":
        vals["prepared"]["input_ids"][0] = 7
        write(root / "prepared.json", vals["prepared"])
        m["entries"][0]["inputs"]["prepared"]["sha256"] = intake.sha(
            (root / "prepared.json").read_bytes()
        )
        r = json.loads((root / "review.json").read_bytes())
        r["cases"][0]["inputs"] = m["entries"][0]["inputs"]
        write(root / "review.json", r)
        m["entries"][0]["review"]["sha256"] = intake.sha((root / "review.json").read_bytes())
    elif mutation == "frozen":
        (root / "test.jsonl").write_bytes(b"{}\n")
        m["frozen_splits"]["test"]["raw"]["sha256"] = intake.sha((root / "test.jsonl").read_bytes())
    else:
        m["entries"].append(dict(m["entries"][0]))
    seal(root, m)
    with pytest.raises((ValueError, KeyError)):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_input_mutation_during_tokenization_fails(bundle, tmp_path, monkeypatch):
    from exl3_mlx import data

    root, model, m, vals = bundle
    original = data.tokenize_row

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (root / "review.json").write_bytes((root / "review.json").read_bytes() + b" ")
        return result

    monkeypatch.setattr(data, "tokenize_row", mutate)
    with pytest.raises(ValueError, match="changed"):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_atomic_publication_does_not_replace_empty_existing_directory(tmp_path):
    stage, dest = tmp_path / "stage", tmp_path / "dest"
    stage.mkdir()
    dest.mkdir()
    (stage / "receipt").write_text("complete")
    with pytest.raises(FileExistsError):
        intake.publish_new(stage, dest)
    assert list(dest.iterdir()) == [] and (stage / "receipt").exists()


def test_embedded_template_identity_and_new_file_invalidation(bundle):
    root, model, m, vals = bundle
    template = (model / "chat_template.jinja").read_text()
    (model / "chat_template.jinja").unlink()
    config = json.loads((model / "tokenizer_config.json").read_bytes())
    config["chat_template"] = template
    write(model / "tokenizer_config.json", config)
    before = tokenizer_identity(model)
    assert "chat_template.jinja" not in before
    (model / "chat_template.jinja").write_text(template)
    assert tokenizer_identity(model) != before
    (model / "chat_template.jinja").unlink()
    config.pop("chat_template")
    write(model / "tokenizer_config.json", config)
    with pytest.raises(ValueError, match="template"):
        tokenizer_identity(model)


def add_case(bundle, case_id, split):
    import copy

    root, model, manifest, vals = bundle
    values = copy.deepcopy(vals)
    source = values["source"]
    source["head"]["commit"] = hashlib.sha1(case_id.encode()).hexdigest()
    source["commits"] = [source["head"]["commit"]]
    source["lineage_keys"] = ["change:" + case_id]
    source["files"]["kernel.metal"]["before"] = blob(f"float {case_id} = 10;")
    source["files"]["kernel.metal"]["after"] = blob(f"float {case_id} = 20;")
    before, after = full_commit_images(source)
    command = render_command(unified_patch(before, after))
    values["raw"]["messages"][0]["content"] = "Specific complete edit " + case_id
    values["raw"]["messages"][-1]["tool_calls"][0]["function"]["arguments"]["cmd"] = command
    values["raw"] = json.loads(intake.canonical(values["raw"]))
    from exl3_mlx.data import load_tokenizer

    values["prepared"] = tokenize_row(
        _normalize(values["raw"], None, None)[0], load_tokenizer(model)
    )
    values["replay"] = replay(command, before, after)
    bindings = {}
    for role, value in values.items():
        path = root / f"{case_id}-{role}.json"
        write(path, value)
        bindings[role] = {"path": path.name, "sha256": intake.sha(path.read_bytes())}
    review = {
        "format": "exact-edit-review-v1",
        "status": "approved",
        "reviewer": "test assertion",
        "cases": [{"id": case_id, "inputs": bindings}],
    }
    path = root / f"{case_id}-review.json"
    write(path, review)
    manifest["entries"].append(
        {
            "id": case_id,
            "kind": "exact_edit",
            "split": split,
            "inputs": bindings,
            "review": {"path": path.name, "sha256": intake.sha(path.read_bytes())},
        }
    )
    if split in ("valid", "test"):
        for suffix, value, role in (
            (".jsonl", values["raw"], "raw"),
            (".prepared.jsonl", values["prepared"], "prepared"),
        ):
            path = root / (split + suffix)
            path.write_bytes(
                path.read_bytes() + json.dumps(value, ensure_ascii=False).encode() + b"\n"
            )
            manifest["frozen_splits"][split][role]["sha256"] = intake.sha(path.read_bytes())
    seal(root, manifest)
    return values


def test_nonempty_frozen_splits_order_and_explicit_sizes(bundle, tmp_path):
    root, model, manifest, vals = bundle
    add_case(bundle, "second", "train")
    add_case(bundle, "development", "valid")
    add_case(bundle, "testingone", "test")
    add_case(bundle, "testingtwo", "test")
    result = intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "out")
    assert result["counts"] == {"train": 2, "valid": 1, "test": 2}
    for split in ("valid", "test"):
        for suffix in (".jsonl", ".prepared.jsonl"):
            assert (tmp_path / "out" / (split + suffix)).read_bytes() == (
                root / (split + suffix)
            ).read_bytes()


def test_transitive_alias_protection_and_cross_split_rejected(bundle, tmp_path):
    root, model, manifest, vals = bundle
    add_case(bundle, "testcase", "test")
    policy = json.loads((root / "policy.json").read_bytes())
    policy["lineage_aliases"] = [
        ["change:one", "alias:bridge"],
        ["alias:bridge", "change:testcase"],
    ]
    write(root / "policy.json", policy)
    manifest["policy"]["sha256"] = intake.sha((root / "policy.json").read_bytes())
    seal(root, manifest)
    with pytest.raises(ValueError, match="crosses explicit splits"):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "invalid", [True, 1, {}, [], {"default": 5}, [{"name": "default", "template": False}], " "]
)
def test_invalid_embedded_templates_refused(bundle, invalid):
    _, model, _, _ = bundle
    (model / "chat_template.jinja").unlink()
    config = json.loads((model / "tokenizer_config.json").read_bytes())
    config["chat_template"] = invalid
    write(model / "tokenizer_config.json", config)
    with pytest.raises(ValueError, match="template"):
        tokenizer_identity(model)


def test_mixed_diagnostic_corpus_not_silently_filtered(bundle, tmp_path):
    import copy

    root, model, m, _ = bundle
    extra = copy.deepcopy(m["entries"][0])
    extra.update(id="reasoning", kind="diagnostic")
    m["entries"].append(extra)
    seal(root, m)
    with pytest.raises(ValueError, match="diagnostic/mixed"):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_symlink_source_payload_refused(bundle, tmp_path):
    root, model, _, _ = bundle
    payload = (root / "raw.json").read_bytes()
    (root / "raw.json").unlink()
    outside = tmp_path / "outside.json"
    outside.write_bytes(payload)
    (root / "raw.json").symlink_to(outside)
    with pytest.raises(ValueError, match="Symlink/path escape"):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "out")


@pytest.mark.parametrize("bad", [None, [], 1])
def test_nonobject_tokenizer_config_refused(bundle, bad):
    _, model, _, _ = bundle
    write(model / "tokenizer_config.json", bad)
    with pytest.raises(ValueError, match="configuration must be an object"):
        tokenizer_identity(model)


@pytest.mark.parametrize("named", [False, True])
def test_embedded_templates_load_with_actual_resolution(bundle, named):
    from exl3_mlx.data import load_tokenizer

    _, model, _, vals = bundle
    template = (model / "chat_template.jinja").read_text()
    (model / "chat_template.jinja").unlink()
    config = json.loads((model / "tokenizer_config.json").read_bytes())
    config["chat_template"] = [{"name": "default", "template": template}] if named else template
    write(model / "tokenizer_config.json", config)
    tokenizer_identity(model)
    tokenizer = load_tokenizer(model)
    normalized, _, _ = _normalize(vals["raw"], None, None)
    assert tokenize_row(normalized, tokenizer) == vals["prepared"]
    if named:
        config["chat_template"].append({"name": "default", "template": template + "bad"})
        write(model / "tokenizer_config.json", config)
        with pytest.raises(ValueError, match="template"):
            tokenizer_identity(model)


@pytest.mark.parametrize(
    "damage",
    [
        "review_null",
        "review_list",
        "case_null",
        "case_missing",
        "pair_null",
        "blob_missing",
        "source_null",
        "raw_null",
        "replay_null",
        "entry_missing",
    ],
)
def test_malformed_nested_cli_is_actionable_and_unpublished(bundle, tmp_path, capsys, damage):
    from exl3_mlx.ktune_cli import main

    root, model, manifest, vals = bundle
    review = json.loads((root / "review.json").read_bytes())
    if damage == "review_null":
        review = None
    elif damage == "review_list":
        review = []
    elif damage == "case_null":
        review["cases"] = [None]
    elif damage == "case_missing":
        review["cases"] = [{"id": "one"}]
    elif damage == "entry_missing":
        del manifest["entries"][0]["inputs"]
    else:
        role = damage.split("_")[0]
        role = "source" if role in {"pair", "blob"} else role
        value = vals[role]
        if damage == "pair_null":
            value["files"]["kernel.metal"] = None
        elif damage == "blob_missing":
            del value["files"]["kernel.metal"]["before"]["content"]
        else:
            value = None
        write(root / f"{role}.json", value)
        manifest["entries"][0]["inputs"][role]["sha256"] = intake.sha(
            (root / f"{role}.json").read_bytes()
        )
        review["cases"][0]["inputs"] = manifest["entries"][0]["inputs"]
    write(root / "review.json", review)
    manifest["entries"][0]["review"]["sha256"] = intake.sha((root / "review.json").read_bytes())
    seal(root, manifest)
    output = tmp_path / "out"
    assert (
        main(
            [
                "prepare-reviewed",
                "--source",
                str(root / "corpus.json"),
                "--model",
                str(model),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    error = capsys.readouterr().err
    assert "Error:" in error and ("expected object" in error or "missing required fields" in error)
    assert not output.exists()
