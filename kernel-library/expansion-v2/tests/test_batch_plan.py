import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "batch_plan", Path(__file__).parents[1] / "tools/batch_plan.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def commit(n):
    return f"{n:040x}"


def config(tmp_path, inventories):
    sources = []
    for n, (repo, rows) in enumerate(inventories):
        path = tmp_path / f"history{n}.jsonl"
        path.write_bytes(b"".join(m.encode(r) + b"\n" for r in rows))
        sources.append({"repository": repo, "history": path.name})
    path = tmp_path / "config.json"
    path.write_bytes(m.encode({"sources": sources}))
    return path


def row(n, **extra):
    return {
        "commit": commit(n),
        "parents": [commit(n + 10000)],
        "message": "Metal performance: faster fused kernel\nReported 10% improvement.",
        "paths": ["src/kernel.metal"],
        **extra,
    }


def write_manifest(tmp_path, candidates=(), protected=()):
    value = {
        "format": "exl3-mlx-upstream-commits-v1",
        "repositories": {"repo": {"url": "https://github.com/a/repo"}},
        "candidates": list(candidates),
        "protected_keys": list(protected),
    }
    value["manifest_sha256"] = m.sha(m.encode(value))
    path = tmp_path / "manifest.json"
    path.write_bytes(m.encode(value))
    return path


def candidate(n):
    return {"id": "case-" + str(n), "repository": "repo", "commit": commit(n)}


def test_fair_quotas_deterministic_and_full_attrition(tmp_path):
    path = config(
        tmp_path,
        [
            ("https://github.com/a/repo", [row(i) for i in range(20)]),
            ("https://github.com/b/repo", [row(i) for i in range(100, 120)]),
        ],
    )
    a = m.select(path, per_source=10, total=6)
    b = m.select(path, per_source=10, total=6)
    assert a == b
    assert sorted(v["selected"] for v in a["counts"]["by_source"].values()) == [3, 3]
    assert a["counts"]["history_rows"] == 40
    assert sum(a["counts"]["reasons"].values()) == 40
    assert all(not x["training_eligible"] for x in a["selected"])


def test_source_order_does_not_change_selected_commit_order(tmp_path):
    path = config(
        tmp_path,
        [
            ("https://github.com/a/repo", [row(i) for i in range(10)]),
            ("https://github.com/b/repo", [row(i) for i in range(100, 110)]),
        ],
    )
    a = m.select(path, total=5)
    value = json.loads(path.read_text())
    value["sources"].reverse()
    path.write_bytes(m.encode(value))
    b = m.select(path, total=5)
    assert [(x["repository"], x["commit"]) for x in a["selected"]] == [
        (x["repository"], x["commit"]) for x in b["selected"]
    ]


def test_prior_commit_exact_and_llama_protection(tmp_path):
    path = config(tmp_path, [("https://github.com/a/repo", [row(1), row(2)]), (m.LLAMA, [row(3)])])
    prior = write_manifest(tmp_path, [candidate(1)])
    result = m.select(path, [prior])
    assert [r["commit"] for r in result["selected"]] == [commit(2)]
    assert result["counts"]["reasons"]["protected_identity"] == 1
    assert result["counts"]["reasons"]["already_in_prior_manifest"] == 1


def test_protected_parent_never_selected(tmp_path):
    path = config(tmp_path, [("https://github.com/a/repo", [row(1, parents=[commit(7)])])])
    prior = write_manifest(tmp_path, protected=["commit:" + commit(7)])
    assert m.select(path, [prior])["counts"]["reasons"] == {"protected_identity": 1}


def test_cross_repository_alias_single_commit(tmp_path):
    path = config(
        tmp_path, [("https://github.com/a/repo", [row(1)]), ("https://github.com/b/repo", [row(1)])]
    )
    x = m.select(path)
    assert x["counts"]["selected_unique_commits"] == 1
    assert x["counts"]["reasons"] == {"selected_for_review": 1, "shared_commit_alias": 1}


def test_exact_message_spans_and_no_claim_promotion(tmp_path):
    message = "日本語\nMetal faster projection 12%\nNo local reproduction.\n"
    path = config(tmp_path, [("https://github.com/a/repo", [row(1, message=message)])])
    x = m.select(path)["selected"][0]
    for evidence in x["source_claim_lines"]:
        assert message[evidence["start_character"] : evidence["end_character"]] == evidence["text"]
    assert x["qualification"].startswith("discovery_only")


@pytest.mark.parametrize(
    "change",
    [
        {"commit": "abc"},
        {"parents": ["abc"]},
        {"message": None},
        {"paths": None},
        {"source_url": "https://github.com/wrong/repo"},
    ],
)
def test_invalid_source_fails_closed(tmp_path, change):
    path = config(tmp_path, [("https://github.com/a/repo", [row(1, **change)])])
    with pytest.raises(ValueError):
        m.select(path)


@pytest.mark.parametrize(
    "kwargs", [{"per_source": True}, {"per_source": 0}, {"total": 501}, {"seed": ""}]
)
def test_bad_quota_rejected(tmp_path, kwargs):
    path = config(tmp_path, [("https://github.com/a/repo", [row(1)])])
    with pytest.raises(ValueError):
        m.select(path, **kwargs)


def test_integrity_and_drift(tmp_path):
    p = tmp_path / "a"
    p.write_bytes(b"abc")
    inputs = m.Inputs()
    inputs.read(p)
    p.write_bytes(b"abd")
    with pytest.raises(ValueError, match="checksum"):
        inputs.verify()
    with pytest.raises(ValueError, match="Duplicate"):
        m.strict(b'{"a":1,"a":2}')
    with pytest.raises(ValueError, match="Nonfinite"):
        m.strict(b'{"a":NaN}')


def test_publication_never_overwrites(tmp_path):
    path = tmp_path / "out.json"
    m.publish({"x": 1}, path)
    with pytest.raises(FileExistsError):
        m.publish({"x": 2}, path)
    assert json.loads(path.read_text()) == {"x": 1}
    assert not list(tmp_path.glob(".pending-*"))


def published(tmp_path):
    mp = write_manifest(tmp_path, [candidate(1), candidate(2)])
    out = tmp_path / "corpus"
    out.mkdir()
    values = {
        "train.jsonl": m.encode({"case_id": "case-1", "messages": [], "tools": []}) + b"\n",
        "test.jsonl": b"",
        "valid.jsonl": b"",
        "audit.json": m.encode(
            [{"id": "case-1", "split": "train", "total_tokens": 4, "target_tokens": 2}]
        ),
        "exclusions.json": m.encode([{"id": "case-2", "reason": "too_long"}]),
    }
    for name, raw in values.items():
        (out / name).write_bytes(raw)
    report = {
        "format": "exl3-agent-kernel-corpus-v1",
        "truncated": 0,
        "counts": {"train": 1, "test": 0, "valid": 0},
        "context_ceiling": 12,
        "artifacts_sha256": {name: m.sha(raw) for name, raw in values.items()},
        "source_manifests": {str(mp): m.sha(mp.read_bytes())},
        "source_evidence": {},
        "helper_source_sha256": {},
    }
    (out / "ktune-import.json").write_bytes(m.encode(report))
    return mp, out


def test_reconcile_accounting_only_never_claims_tokens_checked(tmp_path):
    mp, out = published(tmp_path)
    x = m.reconcile([mp], [out])
    c = x["corpora"][0]
    assert not c["actual_published_tokens_verified"]
    assert [r["status"] for r in c["dispositions"]] == ["published_train", "excluded"]


def test_reconcile_artifact_corruption_rejected(tmp_path):
    mp, out = published(tmp_path)
    (out / "train.jsonl").write_bytes(b"")
    with pytest.raises(ValueError, match="checksum"):
        m.reconcile([mp], [out])


def test_reconcile_missing_disposition_rejected(tmp_path):
    mp, out = published(tmp_path)
    raw = m.encode([])
    (out / "exclusions.json").write_bytes(raw)
    r = json.loads((out / "ktune-import.json").read_text())
    r["artifacts_sha256"]["exclusions.json"] = m.sha(raw)
    (out / "ktune-import.json").write_bytes(m.encode(r))
    with pytest.raises(ValueError, match="disappeared"):
        m.reconcile([mp], [out])
