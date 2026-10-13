"""Synthetic tiny safetensors only; no real checkpoint, MLX or GPU access."""

import copy
import importlib.util
import json
import os
import struct
from pathlib import Path
from unittest.mock import patch

import pytest

SPEC = importlib.util.spec_from_file_location(
    "affine_reader_test", Path(__file__).parents[1] / "tools/affine_row_reader.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def fixture(root, bits=4, dtype="BF16", group=32):
    prefix = "language_model.model.layers.47.self_attn.o_proj"
    name = "trunk.safetensors"
    rows, cols = 5, 256
    payload = bytearray()
    header = {}
    expected = {i: {} for i in range(rows)}
    # Include unrelated physical content and a non-default tensor order.
    header["unrelated"] = {"dtype": "U8", "shape": [11], "data_offsets": [0, 11]}
    payload.extend(b"01234567890")
    for kind, dt, width, size in [
        ("biases", dtype, cols // group, 4 if dtype == "F32" else 2),
        ("weight", "U32", cols * bits // 32, 4),
        ("scales", dtype, cols // group, 4 if dtype == "F32" else 2),
    ]:
        start = len(payload)
        for i in range(rows):
            raw = bytes((j * 7 + i * 17 + len(kind)) % 256 for j in range(width * size))
            expected[i][kind] = raw
            payload.extend(raw)
        header[prefix + "." + kind] = {
            "dtype": dt,
            "shape": [rows, width],
            "data_offsets": [start, len(payload)],
        }
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    (root / name).write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    config = {
        "quantization": {
            "bits": 8,
            "group_size": 64,
            "mode": "affine",
            prefix: {"bits": bits, "group_size": group, "mode": "affine"},
        }
    }
    index = {"weight_map": {key: name for key in header}}
    (root / "config.json").write_text(json.dumps(config))
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    return {
        "root": root,
        "prefix": prefix,
        "shard": name,
        "expected": expected,
        "header": header,
        "config": config,
        "index": index,
        "header_bytes": len(raw),
    }


def bind(f, rows=(4, 0, 4), **kwargs):
    return m.bind_rows(f["root"], prefix=f["prefix"], rows=rows, **kwargs)


def reader(f, identity=None, **kwargs):
    value = bind(f) if identity is None else identity
    return m.AffineRowReader(
        f["root"], value, expected_sha256=m.identity_sha256(value), **kwargs
    )


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("dtype", ["F16", "BF16", "F32"])
def test_exact_raw_rows_all_packed_rates_and_dtypes(tmp_path, bits, dtype):
    f = fixture(tmp_path, bits, dtype)
    identity = bind(f)
    value = reader(f, identity).read([0, 4, 0])
    assert value["rows"] == (0, 4) and value["request_to_unique"] == (0, 1, 0)
    assert value["payloads"] == (f["expected"][0], f["expected"][4])
    assert all(
        isinstance(raw, bytes)
        for payload in value["payloads"]
        for raw in payload.values()
    )
    r = value["receipt"]
    assert (
        r["selected_payload_checksums_verified"]
        and not r["full_file_rehashed"]
        and not r["decoded"]
    )
    assert identity["full_file_hashed"] is False
    assert r["payload_bytes_read"] == sum(
        len(raw) for p in value["payloads"] for raw in p.values()
    )
    assert r["opened_header_verification_bytes"] == 8 + f["header_bytes"]


@pytest.mark.parametrize("group", [32, 64, 128])
def test_group_geometry_and_per_weight_override(tmp_path, group):
    f = fixture(tmp_path, bits=3, group=group)
    p = reader(f).read([4])["receipt"]["plan"]
    assert p["bits"] == 3 and p["group_size"] == group and p["columns"] == 256


def test_bind_and_read_only_selected_spans_no_whole_payload_scan(tmp_path):
    f = fixture(tmp_path)
    for binding in (True, False):
        identity = None if binding else bind(f)
        r = None if binding else reader(f, identity)
        with patch.object(m.os, "pread", wraps=os.pread) as reads:
            result = bind(f) if binding else r.read([4, 0, 4])
        assert reads.call_count == 7  # one opened-header check plus 2x3 payloads
        actual = [(c.args[2], c.args[1]) for c in reads.call_args_list]
        spans = (
            result["plan"]["spans"] if binding else result["receipt"]["plan"]["spans"]
        )
        assert actual == [(0, 8 + f["header_bytes"])] + [
            (s["offset"], s["bytes"]) for s in spans
        ]
        assert (
            sum(n for _, n in actual[1:])
            < (tmp_path / f["shard"]).stat().st_size - 8 - f["header_bytes"]
        )


def test_mutable_caller_identity_and_results_do_not_alter_reader(tmp_path):
    f = fixture(tmp_path)
    identity = bind(f)
    r = reader(f, identity)
    identity["plan"]["spans"][0]["sha256"] = "0" * 64
    result = r.read([4])
    result["payloads"][0]["weight"] = b"changed"
    result["receipt"]["plan"]["spans"][0]["sha256"] = "0" * 64
    assert r.read([4])["payloads"][0] == f["expected"][4]


@pytest.mark.parametrize("rows", [[], [True], [-1], [5], [0.5], [1]])
def test_invalid_or_unbound_rows_before_payload_reads(tmp_path, rows):
    f = fixture(tmp_path)
    r = reader(f)
    with patch.object(m.os, "pread") as reads, pytest.raises(ValueError):
        r.read(rows)
    reads.assert_not_called()


def test_bound_budgets_reject_before_payload_read(tmp_path):
    f = fixture(tmp_path)
    for args in (
        {"max_read_bytes": 1},
        {"max_rows": 1},
        {"max_rows": True},
        {"max_read_bytes": m.MAX_READ_BYTES + 1},
        {"max_file_bytes": 1},
    ):
        with patch.object(m.os, "pread") as reads, pytest.raises(ValueError):
            bind(f, **args)
        reads.assert_not_called()
    r = reader(f, bind(f, [0, 4]), max_rows=2)
    with patch.object(m.os, "pread") as reads, pytest.raises(ValueError):
        r.read([0, 4, 0])
    reads.assert_not_called()


@pytest.mark.parametrize(
    "change", ["split", "physical-owner", "extra", "missing", "path"]
)
def test_index_ownership_refuses_wrong_or_unsafe_owners(tmp_path, change):
    f = fixture(tmp_path)
    index = f["index"]
    key = f["prefix"] + ".scales"
    if change == "split":
        index["weight_map"][key] = "other.safetensors"
    if change == "physical-owner":
        index["weight_map"]["unrelated"] = "other.safetensors"
    if change == "extra":
        index["weight_map"]["not-physical"] = f["shard"]
    if change == "missing":
        del index["weight_map"][key]
    if change == "path":
        index["weight_map"][key] = "../elsewhere.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        bind(f)


def test_requested_wrong_shard_and_conflicting_config_reject(tmp_path):
    f = fixture(tmp_path)
    with pytest.raises(ValueError):
        bind(f, shard_name="other.safetensors")
    f["config"]["quantization_config"] = {"bits": 2}
    (tmp_path / "config.json").write_text(json.dumps(f["config"]))
    with pytest.raises(ValueError):
        bind(f)


def test_identity_pin_geometry_and_hash_field_tampering(tmp_path):
    f = fixture(tmp_path)
    good = bind(f)
    bad = copy.deepcopy(good)
    bad["plan"]["spans"][0]["offset"] += 4
    with pytest.raises(ValueError, match="digest mismatch"):
        m.AffineRowReader(tmp_path, bad, expected_sha256=m.identity_sha256(good))
    with pytest.raises(ValueError, match="geometry"):
        reader(f, bad)
    bad = copy.deepcopy(good)
    bad["plan"]["spans"][0]["sha256"] = "invalid"
    with pytest.raises(ValueError, match="SHA256"):
        reader(f, bad)
    bad = copy.deepcopy(good)
    bad["binding_payload_bytes_read"] = 0
    with pytest.raises(ValueError, match="accounting"):
        reader(f, bad)


def test_payload_mutation_stat_guard_and_selected_hash_backstop(tmp_path):
    f = fixture(tmp_path)
    identity = bind(f)
    r = reader(f, identity)
    offset = identity["plan"]["spans"][0]["offset"]
    path = tmp_path / f["shard"]
    with path.open("r+b") as stream:
        stream.seek(offset)
        stream.write(b"!")
    with pytest.raises(ValueError, match="changed|replaced"):
        r.read([4])
    with (
        patch.object(m, "_unchanged"),
        patch.object(
            m, "_open_bound", side_effect=lambda *_: os.open(path, os.O_RDONLY)
        ),
        pytest.raises(ValueError, match="checksum"),
    ):
        r.read([4])


def test_open_descriptor_toctou_and_replacement(tmp_path):
    f = fixture(tmp_path)
    identity = bind(f)
    r = reader(f, identity)
    original = m.os.open
    path = tmp_path / f["shard"]
    other = tmp_path / "replacement"
    other.write_bytes(path.read_bytes())

    def replace_open(p, *args, **kwargs):
        other.replace(path)
        return original(p, *args, **kwargs)

    with (
        patch.object(m.os, "open", side_effect=replace_open),
        pytest.raises(ValueError, match="while opening"),
    ):
        r.read([4])


def test_opened_header_hash_even_if_stat_guard_misses_change(tmp_path):
    f = fixture(tmp_path)
    identity = bind(f)
    shard = identity["shard"]
    path = tmp_path / f["shard"]
    with path.open("r+b") as stream:
        stream.seek(10)
        stream.write(b"!")
    with (
        patch.object(m.inventory, "regular_identity", return_value=shard["stat"]),
        patch.object(m.inventory, "identity", return_value=shard["stat"]),
        pytest.raises(ValueError, match="header checksum"),
    ):
        m._open_bound(tmp_path, shard)


@pytest.mark.parametrize("when", ["middle", "last"])
def test_mid_read_mutation_detected(tmp_path, when):
    f = fixture(tmp_path)
    r = reader(f)
    original = m.os.pread
    path = tmp_path / f["shard"]
    calls = 0

    def mutate(fd, count, offset):
        nonlocal calls
        calls += 1
        raw = original(fd, count, offset)
        if calls == (2 if when == "middle" else 4):
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"!")
        return raw

    with (
        patch.object(m.os, "pread", side_effect=mutate),
        pytest.raises(ValueError, match="changed|checksum"),
    ):
        r.read([4])


def test_short_read_and_symlink_refused(tmp_path):
    f = fixture(tmp_path)
    r = reader(f)
    original = m.os.pread

    def short(fd, count, offset):
        return (
            original(fd, count, offset)[:-1] if offset else original(fd, count, offset)
        )

    with (
        patch.object(m.os, "pread", side_effect=short),
        pytest.raises(ValueError, match="Truncated"),
    ):
        r.read([4])
    path = tmp_path / f["shard"]
    actual = tmp_path / "actual"
    path.rename(actual)
    path.symlink_to(actual)
    with pytest.raises(ValueError, match="regular"):
        bind(f)


def test_metadata_and_implementation_changes_refused(tmp_path):
    f = fixture(tmp_path)
    r = reader(f)
    with (
        patch.object(m, "implementation_identity", return_value={}),
        pytest.raises(ValueError, match="implementation"),
    ):
        r.read([4])
    (tmp_path / "config.json").write_text(json.dumps(f["config"]) + " ")
    with pytest.raises(ValueError, match="changed"):
        r.read([4])


def test_load_identity_strict_duplicate_json_and_cli_no_payload_export(
    tmp_path, capsys
):
    f = fixture(tmp_path)
    output = tmp_path / "identity.json"
    assert (
        m.main(
            [
                "bind",
                str(tmp_path),
                "--prefix",
                f["prefix"],
                "--rows",
                "4",
                "0",
                "4",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    pin = json.loads(capsys.readouterr().out)["canonical_sha256"]
    assert (
        m.main(
            [
                "read",
                str(tmp_path),
                "--identity",
                str(output),
                "--identity-sha256",
                pin,
                "--rows",
                "0",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["decoded"] is False and "payloads" not in receipt
    assert m.load_identity(output, expected_sha256=pin)["format"] == m.FORMAT
    output.write_text('{"format":1,"format":2}')
    with pytest.raises(ValueError, match="duplicate"):
        m.load_identity(output, expected_sha256=pin)


@pytest.mark.parametrize("mapping", [[1, 0], [0], [False, 1], [0, 2]])
def test_bound_request_mapping_must_preserve_unique_first_occurrence_order(
    tmp_path, mapping
):
    f = fixture(tmp_path)
    bad = bind(f)
    bad["plan"]["request_to_unique"] = mapping
    with pytest.raises(ValueError, match="mapping"):
        reader(f, bad)
