"""Pure CPU tests for portable complete-row comparisons and control evidence."""

import copy
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from exl3_mlx import logits_audit as audit


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, separators=(",", ":")).encode()


def tensor_bytes(a=(1.0, 2.0), b=(0.0, 0.0)):
    header = {
        "layer.lora_a": {"dtype": "F32", "shape": [2, 1], "data_offsets": [0, 8]},
        "layer.lora_b": {"dtype": "F32", "shape": [1, 2], "data_offsets": [8, 16]},
    }
    raw = encode(header)
    return struct.pack("<Q", len(raw)) + raw + struct.pack("<4f", *a, *b)


class Fixture:
    def __init__(self, root):
        self.root = root
        self.manifest = {"format": "observed-logits-audit-v1", "conditions": [], "comparisons": []}
        self.reference, self.reference_fingerprint = self.adapter("reference", b=(0.25, 0.5))
        self.zero, zero_fp = self.adapter("zero")
        for name, role, fingerprint in (
            ("base", "baseline", "0" * 64),
            ("zero", "zero_b", zero_fp),
            ("adapt", "adapter", self.reference_fingerprint),
        ):
            identity = {k: sha(k.encode()) for k in audit._SHARED}
            identity.update(
                request_sha256=sha(name.encode()), model_instance=1, adapter_fingerprint=fingerprint
            )
            c = {
                "id": name,
                "role": role,
                "identity": identity,
                "vocab_size": 5,
                "observations": [],
            }
            if role != "baseline":
                c["adapter"] = self.zero if role == "zero_b" else self.reference
            if role == "zero_b":
                c["zero_b_reference"] = self.reference
            for position, values in ((3, [0, 1, 2, 3, 1]), (4, [0, 1, 2, 3, 4])):
                if role == "adapter":
                    values[0] = 0.25
                self.observation(c, position, values)
            self.manifest["conditions"].append(c)
        self.manifest["comparisons"] = [
            {"left": "base", "right": "zero", "purpose": "zero_b_control", "positions": [3, 4]},
            {"left": "base", "right": "adapt", "purpose": "adapter_effect", "positions": [3, 4]},
        ]

    def file(self, name, raw):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        return {"path": name, "sha256": sha(raw)}

    def adapter(self, name, a=(1.0, 2.0), b=(0.0, 0.0)):
        raw = tensor_bytes(a, b)
        config = encode(
            {
                "format": "test-private-lora",
                "rank": 1,
                "alpha": 2,
                "activation_mode": "checkpoint",
                "adapter_sha256": sha(raw),
            }
        )
        descriptors = {
            "weights": self.file(name + "/adapter.safetensors", raw),
            "config": self.file(name + "/adapter_config.json", config),
        }
        return descriptors, sha(audit._DOMAIN + hashlib.sha256(raw).digest() + config)

    def observation(self, c, position, values):
        prefix = [0, 1, 2, 3][:position]
        raw = struct.pack("<5f", *values)
        metadata = {
            "format": "selected_raw_logits_v1",
            "position": position,
            "prefix_ids_le_u32_sha256": sha(struct.pack("<" + "I" * position, *prefix)),
            "request_sha256": c["identity"]["request_sha256"],
            "model_instance": 1,
            "adapter_fingerprint": c["identity"]["adapter_fingerprint"],
            "vocab_size": 5,
            "serialized_dtype": "F32_LE",
            "source_dtype": "float32",
            "raw_file": f"logits-{position}.f32",
            "raw_bytes": 20,
            "raw_sha256": sha(raw),
            "all_finite": True,
            "observer_changes_timing": True,
            "rows_in_forward": position,
            "argmax": max(range(5), key=values.__getitem__),
        }
        prefix_path = f"{c['id']}/logits-{position}"
        c["observations"].append(
            {
                "metadata": self.file(prefix_path + ".json", encode(metadata)),
                "values": self.file(prefix_path + ".f32", raw),
                "prefix_ids": prefix,
            }
        )

    def metadata(self, condition=0, row=0):
        return self.manifest["conditions"][condition]["observations"][row]["metadata"]

    def mutate_metadata(self, key, value, condition=0):
        d = self.metadata(condition)
        data = json.loads((self.root / d["path"]).read_bytes())
        data[key] = value
        d.update(self.file(d["path"], encode(data)))

    def run(self, **kwargs):
        return audit.audit_logit_observations(self.manifest, self.root, **kwargs)

    def publish(self, **kwargs):
        path = self.root / "manifest.json"
        path.write_bytes(encode(self.manifest))
        return audit.audit_logit_files(manifest=path, output=self.root / "audit.json", **kwargs)


def test_zero_control_and_adapter_effect_with_no_quality_promotion(tmp_path):
    f = Fixture(tmp_path)
    r = f.publish()
    assert r["status"] == "observations_consistent_required_comparisons_passed"
    assert r["comparisons"][0]["all_bitwise_equal"]
    assert r["comparisons"][0]["equality_required_by_purpose"]
    assert not r["comparisons"][1]["all_within_tolerance"]
    assert not r["comparisons"][1]["equality_required_by_purpose"]
    assert r["conditions"][1]["zero_b_control"]["a_unchanged"]
    for key in (
        "quality_certified",
        "device_qualified",
        "trainer_forward_parity",
        "base_payload_independently_verified",
    ):
        assert r[key] is False
    assert r["uncompared_observations"] == []


def test_numerical_oracle_and_explicit_symmetric_tolerance():
    a, b = struct.pack("<4f", 0, -0.0, 2, -4), struct.pack("<4f", -0.0, 0.0, 3, -2)
    exact = audit.compare_logit_values(a, b)
    assert exact["values"] == 4 and not exact["bitwise_equal"]
    assert exact["numeric_different_values"] == 2
    assert exact["max_absolute_error"] == 2
    assert exact["mean_absolute_error"] == 0.75
    assert exact["max_symmetric_relative_error"] == 0.5
    assert exact["outside_tolerance_values"] == 2
    assert exact["argmax_equal"]
    assert audit.compare_logit_values(a, b, rtol=0.5)["within_tolerance"]
    assert audit.compare_logit_values(a, b, atol=2)["within_tolerance"]
    assert not audit.compare_logit_values(a, b, atol=0.999)["within_tolerance"]


@pytest.mark.parametrize("value", [-1, True, float("inf"), float("nan"), "0", None, 10**1000])
def test_invalid_tolerances(value):
    for name in ("atol", "rtol"):
        with pytest.raises(ValueError):
            audit.compare_logit_values(struct.pack("<f", 0), struct.pack("<f", 0), **{name: value})


@pytest.mark.parametrize(
    "left,right",
    [
        (b"", b""),
        (b"123", b"123"),
        (b"1234", b"12345678"),
        (struct.pack("<f", float("nan")), b"0000"),
        (struct.pack("<f", float("inf")), b"0000"),
    ],
)
def test_incomplete_nonfinite_rows(left, right):
    with pytest.raises(ValueError):
        audit.compare_logit_values(left, right)


@pytest.mark.parametrize("key", audit._SHARED)
def test_shared_identity_must_match(tmp_path, key):
    f = Fixture(tmp_path)
    f.manifest["conditions"][1]["identity"][key] = "f" * 64
    with pytest.raises(ValueError, match="Different"):
        f.publish()
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("position", True),
        ("position", 4),
        ("raw_bytes", 16),
        ("argmax", 0),
        ("vocab_size", 6),
        ("request_sha256", "f" * 64),
        ("model_instance", 2),
        ("adapter_fingerprint", "f" * 64),
        ("prefix_ids_le_u32_sha256", "f" * 64),
        ("all_finite", 1),
        ("observer_changes_timing", False),
        ("rows_in_forward", 0),
        ("source_dtype", []),
        ("source_dtype", "int32"),
        ("raw_file", "../logits-3.f32"),
    ],
)
def test_row_metadata_mutations_fail(tmp_path, key, value):
    f = Fixture(tmp_path)
    f.mutate_metadata(key, value)
    with pytest.raises(ValueError):
        f.run()


def test_published_raw_mutation_and_resealed_nonfinite_fail(tmp_path):
    f = Fixture(tmp_path)
    row = f.manifest["conditions"][0]["observations"][0]
    (tmp_path / row["values"]["path"]).write_bytes(struct.pack("<5f", 0, 1, 2, 3, 9))
    with pytest.raises(ValueError, match="hash"):
        f.run()
    raw = struct.pack("<5f", 0, 1, 2, 3, float("nan"))
    row["values"] = f.file(row["values"]["path"], raw)
    f.mutate_metadata("raw_sha256", sha(raw))
    with pytest.raises(ValueError, match="Nonfinite"):
        f.run()


def test_different_prefixes_not_silently_intersected(tmp_path):
    f = Fixture(tmp_path)
    row = f.manifest["conditions"][1]["observations"][0]
    row["prefix_ids"][0] = 4
    f.mutate_metadata("prefix_ids_le_u32_sha256", sha(struct.pack("<3I", 4, 1, 2)), 1)
    with pytest.raises(ValueError, match="Different prefixes|prefix-extension"):
        f.run()


def test_resealed_matched_pairs_cannot_hide_impossible_same_request_trajectory(tmp_path):
    f = Fixture(tmp_path)
    # Each position and both comparison partners agree individually, but this
    # later prefix cannot extend the previously observed prefix for that request.
    for condition in f.manifest["conditions"]:
        row = condition["observations"][1]
        row["prefix_ids"][0] = 4
        descriptor = row["metadata"]
        metadata = json.loads((tmp_path / descriptor["path"]).read_bytes())
        metadata["prefix_ids_le_u32_sha256"] = sha(struct.pack("<4I", 4, 1, 2, 3))
        descriptor.update(f.file(descriptor["path"], encode(metadata)))
    with pytest.raises(ValueError, match="prefix-extension trajectory"):
        f.publish()
    assert not (tmp_path / "audit.json").exists()


def test_nonconsecutive_selected_positions_are_valid_prefix_extensions(tmp_path):
    f = Fixture(tmp_path)
    for condition in f.manifest["conditions"]:
        row = condition["observations"][0]
        row["prefix_ids"] = [0]
        descriptor = row["metadata"]
        metadata = json.loads((tmp_path / descriptor["path"]).read_bytes())
        metadata.update(
            position=1,
            rows_in_forward=1,
            raw_file="logits-1.f32",
            prefix_ids_le_u32_sha256=sha(struct.pack("<I", 0)),
        )
        descriptor.update(f.file(descriptor["path"], encode(metadata)))
    for comparison in f.manifest["comparisons"]:
        comparison["positions"] = [1, 4]
    assert f.run()["required_comparisons_passed"] is True


@pytest.mark.parametrize("bad", [-1, True, 5, 1.0])
def test_invalid_prefix_ids(tmp_path, bad):
    f = Fixture(tmp_path)
    f.manifest["conditions"][0]["observations"][0]["prefix_ids"][0] = bad
    with pytest.raises(ValueError, match="prefix"):
        f.run()


def test_missing_selected_position_and_uncompared_explicit(tmp_path):
    f = Fixture(tmp_path)
    f.manifest["comparisons"][0]["positions"] = [3, 5]
    with pytest.raises(ValueError, match="Missing selected"):
        f.run()
    for pair in f.manifest["comparisons"]:
        pair["positions"] = [3]
    assert len(f.run()["uncompared_observations"]) == 3


def test_zero_control_rejects_changed_a_nonzero_b_and_config(tmp_path):
    f = Fixture(tmp_path)
    zero = f.manifest["conditions"][1]
    original = copy.deepcopy(zero)
    for a, b, message in [((9, 2), (0, 0), "A tensor"), ((1, 2), (1, 0), "nonzero")]:
        zero["adapter"], fingerprint = f.adapter("bad-control", a=a, b=b)
        zero["identity"]["adapter_fingerprint"] = fingerprint
        with pytest.raises(ValueError, match=message):
            f.run()
    f.manifest["conditions"][1] = original
    descriptor = original["adapter"]["config"]
    config = json.loads((tmp_path / descriptor["path"]).read_bytes())
    config["alpha"] = 4
    descriptor.update(f.file(descriptor["path"], encode(config)))
    raw = (tmp_path / original["adapter"]["weights"]["path"]).read_bytes()
    original["identity"]["adapter_fingerprint"] = sha(
        audit._DOMAIN + hashlib.sha256(raw).digest() + encode(config)
    )
    with pytest.raises(ValueError, match="config changed"):
        f.run()


def test_cache_parity_requires_same_adapter(tmp_path):
    f = Fixture(tmp_path)
    f.manifest["comparisons"][1]["purpose"] = "cache_parity"
    with pytest.raises(ValueError, match="same adapter"):
        f.run()
    c = copy.deepcopy(f.manifest["conditions"][2])
    c["id"] = "fresh"
    f.manifest["conditions"].append(c)
    f.manifest["comparisons"][1].update(left="adapt", right="fresh")
    assert f.run()["comparisons"][1]["all_bitwise_equal"]


@pytest.mark.parametrize("bad", ["../escape", "/absolute", "base/../x", "base//x", "base\\x"])
def test_portable_paths_fail_closed(tmp_path, bad):
    f = Fixture(tmp_path)
    f.metadata()["path"] = bad
    with pytest.raises(ValueError):
        f.run()


def test_symlink_file_and_parent_rejected(tmp_path):
    f = Fixture(tmp_path)
    p = tmp_path / f.metadata()["path"]
    target = p.with_name("moved.json")
    p.rename(target)
    p.symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        f.run()
    p.unlink()
    target.rename(p)
    moved = tmp_path / "moved-base"
    (tmp_path / "base").rename(moved)
    (tmp_path / "base").symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        f.run()


@pytest.mark.parametrize("where", ["manifest", "metadata", "config"])
def test_duplicate_json_keys_rejected_without_output(tmp_path, where):
    f = Fixture(tmp_path)
    if where == "manifest":
        path = tmp_path / "manifest.json"
        path.write_bytes(b'{"format":"wrong",' + encode(f.manifest)[1:])
    else:
        d = f.metadata() if where == "metadata" else f.reference["config"]
        raw = (tmp_path / d["path"]).read_bytes()
        key = "position" if where == "metadata" else "rank"
        d.update(f.file(d["path"], ('{"' + key + '":0,').encode() + raw[1:]))
        path = tmp_path / "manifest.json"
        path.write_bytes(encode(f.manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        audit.audit_logit_files(manifest=path, output=tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


def test_no_overwrite_and_input_drift(tmp_path, monkeypatch):
    f = Fixture(tmp_path)
    f.publish()
    with pytest.raises(ValueError, match="Output already"):
        f.publish()
    original = audit.compare_logit_values
    touched = False

    def drifting(*args, **kwargs):
        nonlocal touched
        result = original(*args, **kwargs)
        if len(args[0]) == 20 and not touched:
            p = tmp_path / f.metadata()["path"]
            p.write_bytes(p.read_bytes() + b" ")
            touched = True
        return result

    monkeypatch.setattr(audit, "compare_logit_values", drifting)
    with pytest.raises(ValueError, match="hash|drift|size limit"):
        f.run()


def test_cli_existing_entry_point_without_mlx_import(tmp_path):
    f = Fixture(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(encode(f.manifest))
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    code = "from exl3_mlx.evaluation_audit import main; main(); import sys; assert 'mlx' not in sys.modules"
    p = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "logits",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "audit.json"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["device_qualified"] is False


def test_cli_publishes_failed_control_then_exits_nonzero(tmp_path):
    f = Fixture(tmp_path)
    row = f.manifest["conditions"][1]["observations"][0]
    raw = struct.pack("<5f", 0.5, 1, 2, 3, 1)
    row["values"] = f.file(row["values"]["path"], raw)
    f.mutate_metadata("raw_sha256", sha(raw), 1)
    path = tmp_path / "manifest.json"
    path.write_bytes(encode(f.manifest))
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    command = [sys.executable, "-m", "exl3_mlx.evaluation_audit", "logits", "--manifest", str(path)]
    failed = subprocess.run(
        command + ["--output", str(tmp_path / "failed.json")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 1
    report = json.loads((tmp_path / "failed.json").read_bytes())
    assert report["observation_evidence_consistent"] is True
    assert report["required_comparisons_passed"] is False
    assert report["status"].endswith("_failed")
    passed = subprocess.run(
        command + ["--output", str(tmp_path / "explicit-tolerance.json"), "--atol", "0.5"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert passed.returncode == 0, passed.stderr
    assert json.loads((tmp_path / "explicit-tolerance.json").read_bytes())["atol"] == 0.5


def test_adapter_effect_only_is_explicitly_descriptive(tmp_path):
    f = Fixture(tmp_path)
    f.manifest["comparisons"] = f.manifest["comparisons"][1:]
    result = f.run()
    assert result["required_comparisons"] == 0
    assert result["status"] == "observations_consistent_descriptive_only"
    assert not result["comparisons"][0]["all_within_tolerance"]
    assert not result["comparisons"][0]["improvement_claimed"]


@pytest.mark.parametrize("bad", [None, [], 1, "x"])
def test_malformed_nested_json_types(tmp_path, bad):
    for where in ("identity", "role", "observations", "adapter"):
        root = tmp_path / where
        root.mkdir()
        f = Fixture(root)
        f.manifest["conditions"][1][where] = bad
        with pytest.raises(ValueError):
            f.run()


def test_safetensors_parser_spans_pairs_and_values():
    raw = tensor_bytes()
    n = struct.unpack_from("<Q", raw)[0]
    header = json.loads(raw[8 : 8 + n])
    for change in ("dtype", "span", "shape", "key"):
        h = copy.deepcopy(header)
        if change == "dtype":
            h["layer.lora_a"]["dtype"] = "BF16"
        if change == "span":
            h["layer.lora_b"]["data_offsets"] = [4, 12]
        if change == "shape":
            h["layer.lora_b"]["shape"] = [2, 1]
        if change == "key":
            h["other.lora_b"] = h.pop("layer.lora_b")
        encoded = encode(h)
        with pytest.raises(ValueError):
            audit._tensors(struct.pack("<Q", len(encoded)) + encoded + raw[8 + n :])
    with pytest.raises(ValueError, match="Nonfinite"):
        audit._tensors(tensor_bytes(a=(math.nan, 1)))
    with pytest.raises(ValueError, match="Trailing"):
        audit._tensors(raw + b"\0\0\0\0")
