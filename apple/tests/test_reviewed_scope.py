"""File-scoped replay retains full-source protection and explicit admission."""

import copy
import json

import pytest
import test_reviewed_corpus as fixtures

from exl3_mlx import reviewed_corpus as intake
from exl3_mlx.data import load_tokenizer, tokenize_row
from exl3_mlx.sft_data import _normalize

bundle = fixtures.bundle


def scoped(values, name="README.md", text="new documentation\n"):
    values = copy.deepcopy(values)
    pair = {
        "role": "changed",
        "before": fixtures.blob("old documentation\n"),
        "after": fixtures.blob(text),
    }
    source = values["source"]
    selected = sorted(source["files"])
    source["files"][name] = pair
    omission = {
        "classification": "documentation",
        "before_sha256": pair["before"]["sha256"],
        "after_sha256": pair["after"]["sha256"],
        "reason": "Fixture reviewed prose only.",
    }
    source["edit_scope"] = {
        "format": "reviewed-file-edit-scope-v1",
        "coverage": "complete_selected_file_edits",
        "selected_paths": selected,
        "omitted_paths": {name: omission},
    }
    return values


def publish(bundle, values, *, kind=intake.MIXED_FORMAT, split="train", rid="one"):
    root, model, manifest, _ = bundle
    values["prepared"] = tokenize_row(
        _normalize(values["raw"], None, None)[0], load_tokenizer(model)
    )
    inputs = {}
    for role, value in values.items():
        path = root / (rid + "-" + role + ".json")
        fixtures.write(path, value)
        inputs[role] = {"path": path.name, "sha256": intake.sha(path.read_bytes())}
    review = {
        "format": "exact-edit-review-v1",
        "status": "approved",
        "reviewer": "fixture assertion",
        "cases": [{"id": rid, "inputs": inputs}],
    }
    rp = root / (rid + "-review.json")
    fixtures.write(rp, review)
    entry = {
        "id": rid,
        "kind": "exact_edit",
        "split": split,
        "inputs": inputs,
        "review": {"path": rp.name, "sha256": intake.sha(rp.read_bytes())},
    }
    if rid == "one":
        manifest["entries"] = [entry]
    else:
        manifest["entries"].append(entry)
    manifest["format"] = kind
    if split != "train":
        for role, suffix in [("raw", ".jsonl"), ("prepared", ".prepared.jsonl")]:
            p = root / (split + suffix)
            p.write_bytes(intake.canonical(values[role]) + b"\n")
            manifest["frozen_splits"][split][role] = {
                "path": p.name,
                "sha256": intake.sha(p.read_bytes()),
            }
    fixtures.seal(root, manifest)


def test_scoped_mixed_import_keeps_full_source(bundle, tmp_path):
    values = scoped(bundle[3])
    publish(bundle, values)
    result = intake.prepare_reviewed_corpus(bundle[0] / "corpus.json", bundle[1], tmp_path / "out")
    assert result["counts"] == {"train": 1, "valid": 0, "test": 0}
    record = json.loads((tmp_path / "out/provenance.jsonl").read_text())
    assert "README.md" in record["source"]["files"]
    replay = record["replay"]
    assert replay["full_commit_replay"] is False
    assert replay["selected_stable_patch_id"] != replay["full_stable_patch_id"]


def test_v1_refuses_explicit_scope(bundle):
    v = scoped(bundle[3])
    with pytest.raises(ValueError, match="mixed v2"):
        intake.source_edit(v["source"], v["raw"], v["replay"])


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "overlap",
        "extra",
        "empty",
        "wrong_hash",
        "code",
        "new_doc",
        "mode",
        "hidden_context",
        "historical",
        "bad_schema",
    ],
)
def test_scope_corruptions_refused(bundle, damage):
    v = scoped(bundle[3])
    s = v["source"]
    scope = s["edit_scope"]
    if damage == "missing":
        scope["omitted_paths"] = {}
    elif damage == "overlap":
        scope["selected_paths"].append("README.md")
        scope["selected_paths"].sort()
    elif damage == "extra":
        scope["selected_paths"].append("absent")
    elif damage == "empty":
        scope["selected_paths"] = []
    elif damage == "wrong_hash":
        scope["omitted_paths"]["README.md"]["after_sha256"] = "0" * 64
    elif damage == "code":
        s["files"]["omitted.metal"] = s["files"].pop("README.md")
        scope["omitted_paths"]["omitted.metal"] = scope["omitted_paths"].pop("README.md")
    elif damage == "new_doc":
        s["files"]["README.md"]["before"] = None
    elif damage == "mode":
        s["files"]["README.md"]["after"]["mode"] = "100755"
    elif damage == "hidden_context":
        s["files"]["README.md"]["role"] = "context"
        s.pop("edit_scope")
    elif damage == "historical":
        v["replay"]["omitted_documentation"] = {"wrong.md": {}}
    elif damage == "bad_schema":
        scope["omitted_paths"]["README.md"] = None
    with pytest.raises(ValueError):
        intake.source_edit(s, v["raw"], v["replay"], allow_scoped=True)


def test_historical_omission_hashes_bound(bundle):
    v = scoped(bundle[3])
    d = copy.deepcopy(v["source"]["edit_scope"]["omitted_paths"]["README.md"])
    d.pop("classification")
    v["replay"]["omitted_documentation"] = {"README.md": d}
    intake.source_edit(v["source"], v["raw"], v["replay"], allow_scoped=True)
    d["before_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Historical omission"):
        intake.source_edit(v["source"], v["raw"], v["replay"], allow_scoped=True)


@pytest.mark.parametrize("split", ["train", "test"])
def test_same_selected_edit_with_different_docs_is_detected(bundle, tmp_path, split):
    first = scoped(bundle[3])
    publish(bundle, first)
    second = scoped(bundle[3], text="different docs\n")
    second["source"]["head"]["commit"] = "3" * 40
    second["source"]["commits"] = ["3" * 40, "4" * 40]
    second["source"]["lineage_keys"] = []
    second["raw"]["messages"][0]["content"] += " Another source snapshot."
    publish(bundle, second, split=split, rid="two")
    if split == "test":
        bundle[2]["multi_views"] = [["one", "two"]]
        fixtures.seal(bundle[0], bundle[2])
    with pytest.raises(ValueError, match="multi_views|crosses explicit splits"):
        intake.prepare_reviewed_corpus(bundle[0] / "corpus.json", bundle[1], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_selected_patch_identity_is_protected(bundle, tmp_path):
    v = scoped(bundle[3])
    publish(bundle, v)
    current, _, _ = intake.source_edit(v["source"], v["raw"], v["replay"], allow_scoped=True)
    root, model, manifest, _ = bundle
    p = root / "policy.json"
    policy = json.loads(p.read_bytes())
    policy["protected_patch_ids"] = [current["selected_stable_patch_id"]]
    fixtures.write(p, policy)
    manifest["policy"]["sha256"] = intake.sha(p.read_bytes())
    fixtures.seal(root, manifest)
    with pytest.raises(ValueError, match="Protected source"):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "out")


def test_malformed_role_cli_is_actionable_and_unpublished(bundle, tmp_path, capsys):
    from exl3_mlx.ktune_cli import main

    values = scoped(bundle[3])
    values["source"]["files"]["README.md"]["role"] = ["changed"]
    publish(bundle, values)
    root, model, _, _ = bundle
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
    assert "Unknown source role" in capsys.readouterr().err
    assert not output.exists()
