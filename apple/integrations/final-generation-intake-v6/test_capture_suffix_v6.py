import importlib.util, json, copy
from pathlib import Path
import pytest

p = Path(__file__).with_name("capture_suffix_v6.py")
sp = importlib.util.spec_from_file_location("cap", p)
m = importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)


class Tokenizer:
    def get_vocab(self):
        return {str(i): i for i in range(1000)}

    def apply_chat_template(self, *a, **kw):
        return list(range(774))

    def decode(self, ids, **kw):
        assert kw == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return "synthetic<｜end▁of▁sentence｜>"


@pytest.fixture
def records(monkeypatch):
    req = {
        "format": "mlx-stream-adapter-generation-request-v1",
        "mode": "adapter",
        "max_new_tokens": 4096,
        "max_context_tokens": 8192,
        "policy": m.POLICY,
        "tf32": False,
        "model_root": "/base",
        "engine": "/engine",
        "runtime_manifest": "/runtime",
        "payload_manifest": "/payload",
        "adapter_dir": "/adapter",
        "eos_token_ids": [1],
        "arithmetic_profile": "mlx_stream_served_transfer_v1",
        "serving_profile": {
            "name": "served_transfer_v1",
            "warmup": "serial_owned_v1",
            "draft_policy": "disabled_weights_retained",
            "user_settings_policy": "ignored_fixed_defaults_v1",
            "embedding_host_rows": False,
            "cross_request_prefix_cache": False,
        },
        "prompt_ids_file": "/prompt-ids",
        "inputs_sha256": {
            "/engine": m.ENGINE,
            "/runtime": "b" * 64,
            "/payload": "c" * 64,
            "/adapter/adapter.safetensors": "d" * 64,
            "/adapter/adapter_config.json": "e" * 64,
        },
    }
    req["execution"] = {
        "deadline_seconds": 1800,
        "environment": {"PATH": "/usr/bin:/bin"},
        "lease_request": "/lease-request",
        "lease_state": "/lease-state",
        "wrapper": "/wrapper",
        "process_memory_gib": 40,
        "native_box_ceiling_gib": 80,
        "physical_memory_gib": 128,
        "reserve_gib": 32,
    }
    q = {
        "format": "private-cached-serving-device-admission-v1",
        "status": "qualified",
        "checks": dict.fromkeys(
            [
                "native_construction",
                "zero_b_invariance",
                "nonzero_adapter_projection",
                "cached_fresh_logits",
                "context_capacity",
                "resource_and_lease",
            ],
            True,
        ),
        "GPU_executed": True,
        "trainer_forward_parity": False,
        "engine_sha256": m.ENGINE,
        "model_vocab_size": 1000,
        "max_context": 8192,
        "max_new_tokens": 4096,
        "samples_per_condition": 1,
        "retry_policy": "none",
        "model_root": "/base",
        "runtime_manifest_sha256": "b" * 64,
        "payload_manifest_sha256": "c" * 64,
        "eos_token_ids": [1],
        "adapters": {
            "d" * 64: {
                "condition": "final458",
                "checkpoint_step": 458,
                "config_sha256": "e" * 64,
                "rank": 4,
                "alpha": 8,
                "layer": 39,
                "parameters": 78848,
                "fingerprint": "f" * 64,
            }
        },
    }
    prompt = {"messages": [], "tools": []}
    prepared = {
        "input_ids": list(range(774)),
        "tokenizer": {"tokenizer.json": "a" * 64},
    }
    native = {
        "format": "mlx-stream-adapter-construction-v1",
        "request_sha256": "",
        "model_instance": 1,
        "resolved_tier": {"path": "fixture"},
        "adapter_fingerprint": "f" * 64,
        "loaded_embedding_vocab": 1000,
        "loaded_head_vocab": 1000,
    }
    native.update({k: v for k, v in req["serving_profile"].items() if k != "name"})
    native["trainer_forward_parity"] = False
    res = {
        "format": "mlx-stream-adapter-generation-result-v1",
        "status": "completed",
        "max_new_tokens": 4096,
        "prompt_ids": list(range(774)),
        "policy": m.POLICY,
        "model_instance": 1,
        "resolved_tier": {"path": "fixture"},
        "adapter_sha256": "d" * 64,
        "engine_sha256": m.ENGINE,
        "runtime_manifest_sha256": "b" * 64,
        "payload_manifest_sha256": "c" * 64,
        "loaded_embedding_vocab": 1000,
        "loaded_head_vocab": 1000,
        "eos_token_ids": [1],
        "output_ids": [4, 1],
        "stop_reason": "eos",
        "raw_text": "synthetic<｜end▁of▁sentence｜>",
        "arithmetic_profile": "mlx_stream_served_transfer_v1",
        "serving_profile": req["serving_profile"],
        "tokenizer_sha256": "a" * 64,
    }
    res["execution_profile"] = {
        k: copy.deepcopy(req["execution"][k])
        for k in ("deadline_seconds", "environment", "process_memory_gib", "native_box_ceiling_gib", "physical_memory_gib", "reserve_gib")
    }
    res.update(
        dict.fromkeys(
            ["chat_reformatted", "prompt_truncated", "tf32", "trainer_forward_parity"],
            False,
        )
    )
    audit = {
        "status": "receipt_consistent",
        "normal_eos": True,
        "truncated": False,
        "model_instance": 1,
        "resolved_tier": {"path": "fixture"},
        "logits_observer": {"enabled": False},
    }
    d = dict(
        request=req,
        qualification=q,
        prompt=prompt,
        prepared=prepared,
        construction=native,
        result=res,
        audit=audit,
        prompt_ids=list(range(774)),
    )

    def bind():
        enc = lambda v: json.dumps(v, sort_keys=True).encode()
        req["inputs_sha256"]["/prompt-ids"] = m.sha(enc(d["prompt_ids"]))
        rh = m.sha(enc(req))
        native["request_sha256"] = res["request_sha256"] = audit["request_sha256"] = rh
        nh = m.sha(enc(native))
        res["native_construction_sha256"] = audit["native_construction_sha256"] = nh
        audit["child_report_sha256"] = m.sha(enc(res))
        raw = {k: enc(v) for k, v in d.items()}
        monkeypatch.setattr(m, "PROMPT", m.sha(raw["prompt"]))
        monkeypatch.setattr(m, "PREPARED", m.sha(raw["prepared"]))
        return raw

    return d, bind


def test_consistent_fabricated_records(records):
    d, bind = records
    raw = bind()
    suffix, receipt = m.validate(d, raw, Tokenizer())
    assert (
        suffix
        == b"synthetic<\xef\xbd\x9cend\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>"
        and receipt["tokens"] == 2
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("qualification", "status", "linked"),
        ("qualification", "GPU_executed", False),
        ("qualification", "max_context", 1024),
        ("qualification", "engine_sha256", "0" * 64),
        ("qualification", "retry_policy", "retry"),
        ("result", "output_ids", [4]),
        ("result", "output_ids", [1, 4, 1]),
        ("result", "output_ids", [True, 1]),
        ("result", "output_ids", [1000, 1]),
        ("result", "output_ids", list(range(774)) + [1]),
        ("result", "raw_text", "different"),
        ("result", "stop_reason", "max_tokens"),
        ("result", "prompt_ids", [2]),
        ("result", "max_new_tokens", 4095),
        ("result", "tf32", True),
        ("request", "policy", {}),
        ("result", "runtime_manifest_sha256", "0" * 64),
        ("construction", "model_instance", 2),
        ("construction", "adapter_fingerprint", "0" * 64),
        ("audit", "normal_eos", False),
        ("audit", "logits_observer", None),
    ],
)
def test_coherent_bad_records_rejected(records, section, key, value):
    d, bind = records
    d[section][key] = value
    raw = bind()
    with pytest.raises(ValueError):
        m.validate(d, raw, Tokenizer())


def test_failed_capture_preserves_exact_original_records(
    records, tmp_path, monkeypatch
):
    from exl3_mlx import data

    d, bind = records
    d["result"]["raw_text"] = "bad"
    raw = bind()
    files = {}
    for role, b in raw.items():
        p = tmp_path / (role + ".json")
        p.write_bytes(b)
        files[role] = {"path": str(p), "sha256": m.sha(b)}
    evidence = tmp_path / "evidence"
    evidence.write_bytes(b"fabricated CPU fixture, not device evidence")
    d["qualification"]["evidence"] = [
        {"path": str(evidence), "sha256": m.sha(evidence.read_bytes())}
    ]
    raw = bind()
    for role, b in raw.items():
        Path(files[role]["path"]).write_bytes(b)
        files[role]["sha256"] = m.sha(b)
    manifest = {
        "format": m.FORMAT,
        "files": files,
        "tokenizer_dir": str(tmp_path),
        "tokenizer_identity": {"tokenizer.json": "a" * 64},
    }
    p = tmp_path / "manifest"
    p.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        data, "tokenizer_identity", lambda p: {"tokenizer.json": "a" * 64}
    )
    out = tmp_path / "out"
    r = m.capture(
        p, m.sha(p.read_bytes()), out, tokenizer_factory=lambda p: Tokenizer()
    )
    assert (
        not r["accepted"]
        and (out / "result.captured.json").read_bytes() == raw["result"]
    )
    assert not (out / "generation.raw").exists()


def test_strict_duplicate_json():
    with pytest.raises(ValueError):
        m.strict(b'{"x":1,"x":2}')
    with pytest.raises(ValueError):
        m.strict(b'{"x":NaN}')


@pytest.mark.parametrize(
    "field,value",
    [
        ("warmup", "other"),
        ("draft_policy", "enabled"),
        ("user_settings_policy", "user"),
        ("embedding_host_rows", True),
        ("cross_request_prefix_cache", True),
        ("trainer_forward_parity", True),
    ],
)
def test_construction_coherent_mutation(records, field, value):
    d, bind = records
    d["construction"][field] = value
    with pytest.raises(ValueError, match="construction policy"):
        m.validate(d, bind(), Tokenizer())


def test_missing_prompt_file(records):
    d, bind = records
    del d["request"]["prompt_ids_file"]
    with pytest.raises(ValueError, match="prompt file"):
        m.validate(d, bind(), Tokenizer())


def test_wrong_prompt_file_pin(records):
    d, bind = records
    d["request"]["prompt_ids_file"] = "/unrelated"
    with pytest.raises(ValueError, match="prompt file"):
        m.validate(d, bind(), Tokenizer())


def test_prompt_file_ids_differ(records):
    d, bind = records
    d["prompt_ids"][0] = 5
    with pytest.raises(ValueError, match="prompt IDs"):
        m.validate(d, bind(), Tokenizer())


@pytest.mark.parametrize(
    "field,value",
    [("rank", 8), ("checkpoint_step", 17), ("alpha", True), ("fingerprint", None)],
)
def test_control_parent_basis(records, field, value):
    d, bind = records
    q = d["qualification"]
    parent = copy.deepcopy(q["adapters"]["d" * 64])
    parent[field] = value
    q["adapters"]["0" * 64] = parent
    q["adapters"]["d" * 64].update(condition="zeroB-final458", derived_from="0" * 64)
    with pytest.raises(ValueError, match="zero-B parent"):
        m.validate(d, bind(), Tokenizer())


@pytest.mark.parametrize("positions", [[True], [773], [4870], [774, 774]])
def test_observer_invalid_positions(records, positions):
    d, bind = records
    spec = {"format": "selected_raw_logits_v1", "positions": positions}
    for role in ("request", "result", "construction"):
        d[role]["logits_observer"] = spec
    d["audit"]["logits_observer"] = {
        "enabled": True,
        "rows_verified": len(positions),
        "rows": [{} for _ in positions],
    }
    with pytest.raises(ValueError, match="observer positions"):
        m.validate(d, bind(), Tokenizer())


@pytest.mark.parametrize("value", [None, True, 1024, 10000])
def test_request_context(records, value):
    d, bind = records
    d["request"]["max_context_tokens"] = value
    with pytest.raises(ValueError, match="context"):
        m.validate(d, bind(), Tokenizer())


def test_execution_profile(records):
    d, bind = records
    d["result"]["execution_profile"] = {}
    with pytest.raises(ValueError, match="Execution profile"):
        m.validate(d, bind(), Tokenizer())


@pytest.mark.parametrize(
    "field,value",
    [
        ("process_memory_gib", True),
        ("process_memory_gib", 129),
        ("reserve_gib", 7),
        ("deadline_seconds", 86401),
        ("environment", {}),
        ("environment", {"PATH": "/bin", "MLX_ENABLE_TF32": "1"}),
    ],
)
def test_execution_schema_negatives(records, field, value):
    d, bind = records
    d["request"]["execution"][field] = value
    d["result"]["execution_profile"] = {
        k: d["request"]["execution"][k]
        for k in ("deadline_seconds", "environment", "process_memory_gib", "native_box_ceiling_gib", "physical_memory_gib", "reserve_gib")
    }
    with pytest.raises(ValueError, match="execution"):
        m.validate(d, bind(), Tokenizer())


@pytest.mark.parametrize(
    "field", ["model_instance", "loaded_head_vocab", "loaded_embedding_vocab"]
)
def test_native_integer_types(records, field):
    d, bind = records
    d["construction"][field] = float(d["construction"][field])
    with pytest.raises(ValueError):
        m.validate(d, bind(), Tokenizer())


def test_nonhash_config(records):
    d, bind = records
    d["qualification"]["adapters"]["d" * 64]["config_sha256"] = "unhashed"
    d["request"]["inputs_sha256"]["/adapter/adapter_config.json"] = "unhashed"
    with pytest.raises(ValueError, match="config digest"):
        m.validate(d, bind(), Tokenizer())


@pytest.mark.parametrize("count", [9, 1000])
def test_oversize_observer_envelope_reads_nothing(records, monkeypatch, count):
    d, bind = records
    d["audit"]["logits_observer"] = {
        "enabled": True,
        "rows": [{}] * count,
        "rows_verified": count,
    }
    monkeypatch.setattr(m, "read", lambda *a, **kw: pytest.fail("unexpected file read"))
    with pytest.raises(ValueError, match="capture bound"):
        m.observer_envelope(d)


def test_disabled_rows_no_read(records, monkeypatch):
    d, bind = records
    d["audit"]["logits_observer"]["rows"] = [{}]
    monkeypatch.setattr(m, "read", lambda *a, **kw: pytest.fail("unexpected file read"))
    with pytest.raises(ValueError, match="Disabled"):
        m.observer_envelope(d)


def test_zero_b_parent_nonhash(records):
    d, bind = records
    q = d["qualification"]
    q["adapters"]["bad"] = copy.deepcopy(q["adapters"]["d" * 64])
    q["adapters"]["d" * 64].update(condition="zeroB-final458", derived_from="bad")
    with pytest.raises(ValueError, match="parent digest"):
        m.validate(d, bind(), Tokenizer())

@pytest.mark.parametrize("field", ["process_memory_gib", "native_box_ceiling_gib", "physical_memory_gib"])
@pytest.mark.parametrize("value", [True, 0, 129, 128.0])
def test_new_profile_integer_fields(records, field, value):
    d, bind = records
    d["request"]["execution"][field] = value
    with pytest.raises(ValueError, match="execution"):
        m.validate(d, bind(), Tokenizer())

@pytest.mark.parametrize("field", ["process_memory_gib", "native_box_ceiling_gib", "physical_memory_gib"])
def test_missing_current_fields(records, field):
    d, bind = records
    del d["request"]["execution"][field]
    with pytest.raises(ValueError, match="execution"):
        m.validate(d, bind(), Tokenizer())

def test_obsolete_only_profile_refused(records):
    d, bind = records
    e=d["request"]["execution"]
    for k in ["process_memory_gib", "native_box_ceiling_gib", "physical_memory_gib"]:del e[k]
    e["memory_cap_gib"]=40
    with pytest.raises(ValueError, match="execution"):
        m.validate(d, bind(), Tokenizer())

@pytest.mark.parametrize("field,value", [("process_memory_gib",100), ("native_box_ceiling_gib",128)])
def test_resource_geometry(records, field, value):
    d, bind = records
    d["request"]["execution"]["physical_memory_gib"] = 96
    d["request"]["execution"][field] = value
    with pytest.raises(ValueError, match="execution"):
        m.validate(d, bind(), Tokenizer())

def test_old_engine_rejected(records):
    d,bind=records
    d["qualification"]["engine_sha256"]="ea5f55e4e7ff2cf8fd5905151ba8c39466ac05d2a5712b438fea4c8c37e04bd3"
    with pytest.raises(ValueError,match="binary"):
        m.validate(d,bind(),Tokenizer())

@pytest.mark.parametrize("value", [None, True, "32", 32.0, 7, 65])
def test_reserve_type_before_arithmetic(records, value):
    d,bind=records
    d["request"]["execution"]["reserve_gib"]=value
    with pytest.raises(ValueError,match="execution"):
        m.validate(d,bind(),Tokenizer())
