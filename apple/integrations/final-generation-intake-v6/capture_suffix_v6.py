"""Private CPU-only suffix identity check; qualification attestations remain external."""

import argparse
import hashlib
import json
import os
import re
import stat
import math
import struct
from pathlib import Path

FORMAT = "private-native-suffix-capture-v1"
ENGINE = "ebc54335731cf3227c04855082828d1c6ad390ca2eb8053a8f1e032e1d33fe2e"
PROMPT = "2f9c298e94c1a78500d292aec8b11b33c7c3f1c31a4d5642362c2ca67850130d"
PREPARED = "dcc3686625d1c3394c816e6810adcdad01fa821aa30db81b0f6f5107ebdacf06"
POLICY = {
    "sampling": "greedy",
    "temperature": 0,
    "top_p": 1,
    "repetition_penalty": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "suppressed_token_ids": [],
    "logit_processors": [],
    "prompt_format": "exact_token_ids",
    "truncation": "reject",
    "draft": False,
    "cross_request_prefix_cache": False,
}
ROLES = (
    "request",
    "result",
    "audit",
    "construction",
    "qualification",
    "prompt",
    "prepared",
    "prompt_ids",
)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def exact(a, b):
    return json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(
        b, sort_keys=True, separators=(",", ":")
    )


def strict(raw):
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, "Duplicate JSON key")
            d[k] = v
        return d

    def bad(x):
        raise ValueError("Nonfinite JSON")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def read(path, limit=8 << 20):
    path = Path(path).absolute()
    require(path.resolve(strict=True) == path, "Noncanonical input path")
    a = path.lstat()
    require(
        stat.S_ISREG(a.st_mode) and a.st_size <= limit, "Nonregular/oversized input"
    )
    identity = lambda s: (
        s.st_dev,
        s.st_ino,
        s.st_mode,
        s.st_size,
        s.st_mtime_ns,
        s.st_ctime_ns,
    )
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        b = os.fstat(fd)
        require(
            stat.S_ISREG(b.st_mode) and identity(a) == identity(b), "Input replaced"
        )
        with os.fdopen(fd, "rb", closefd=False) as f:
            data = f.read(limit + 1)
        c = os.fstat(fd)
    finally:
        os.close(fd)
    require(
        len(data) <= limit and identity(b) == identity(c) == identity(path.lstat()),
        "Input drift",
    )
    return data


def validate(records, raws, tokenizer):
    req, res, audit, native, q, prompt, prepared, prompt_ids = (
        records[k] for k in ROLES
    )
    require(
        all(isinstance(records[k], dict) for k in ROLES if k != "prompt_ids"),
        "Expected object records",
    )
    require(
        sha(raws["prompt"]) == PROMPT and sha(raws["prepared"]) == PREPARED,
        "Wrong fixed prompt",
    )
    ids = prepared["input_ids"]
    require(exact(prompt_ids, ids), "Captured request prompt IDs mismatch")
    require(
        isinstance(req.get("prompt_ids_file"), str)
        and req.get("inputs_sha256", {}).get(req["prompt_ids_file"])
        == sha(raws["prompt_ids"]),
        "Request prompt file pin mismatch",
    )
    require(len(ids) == 774, "Unexpected fixed prompt length")
    require(
        exact(
            tokenizer.apply_chat_template(
                prompt["messages"],
                tools=prompt["tools"],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            ),
            ids,
        ),
        "Template/prompt ID drift",
    )
    require(
        q.get("format") == "private-cached-serving-device-admission-v1"
        and q.get("status") == "qualified",
        "Engine not device-qualified",
    )
    checks = {
        "native_construction",
        "zero_b_invariance",
        "nonzero_adapter_projection",
        "cached_fresh_logits",
        "context_capacity",
        "resource_and_lease",
    }
    require(
        isinstance(q.get("checks"), dict)
        and set(q["checks"]) == checks
        and all(v is True for v in q["checks"].values()),
        "Missing device checks",
    )
    require(
        q.get("GPU_executed") is True and q.get("trainer_forward_parity") is False,
        "Invalid qualification scope",
    )
    require(q.get("engine_sha256") == ENGINE, "Wrong qualified observer binary")
    require(
        type(q.get("model_vocab_size")) is int and q["model_vocab_size"] > 0,
        "Invalid model vocabulary",
    )
    require(
        q.get("max_context", 0) >= 774 + 4096 and q.get("max_new_tokens", 0) >= 4096,
        "Unqualified generation capacity",
    )
    require(
        type(q.get("samples_per_condition")) is int
        and q.get("samples_per_condition") == 1
        and q.get("retry_policy") == "none",
        "Sampling admission mismatch",
    )
    require(
        req.get("format") == "mlx-stream-adapter-generation-request-v1"
        and req.get("mode") == "adapter",
        "Expected adapter request, including zero-B control",
    )
    require(
        req.get("max_new_tokens") == 4096 and type(req["max_new_tokens"]) is int,
        "Wrong generation cap",
    )
    require(
        exact(req.get("policy"), POLICY) and req.get("tf32") is False,
        "Wrong generation policy",
    )
    require(req.get("model_root") == q.get("model_root"), "Base root mismatch")
    require(
        type(req.get("max_context_tokens")) is int
        and 4870 <= req["max_context_tokens"] <= q["max_context"],
        "Request context unqualified",
    )
    execution = req.get("execution")
    require(
        isinstance(execution, dict)
        and set(execution)
        == {
            "lease_request",
            "lease_state",
            "wrapper",
            "deadline_seconds",
            "environment",
            "process_memory_gib",
            "native_box_ceiling_gib",
            "physical_memory_gib",
            "reserve_gib",
        },
        "Missing supervised execution profile",
    )
    require(
        all(
            isinstance(execution[k], str) and Path(execution[k]).is_absolute()
            for k in ("lease_request", "lease_state", "wrapper")
        ),
        "Invalid execution paths",
    )
    env = execution["environment"]
    require(
        isinstance(env, dict)
        and "PATH" in env
        and set(env) <= {"PATH", "HOME", "TMPDIR"}
        and all(isinstance(v, str) and "\x00" not in v for v in env.values()),
        "Invalid execution environment",
    )
    require(
        type(execution["deadline_seconds"]) is int
        and 1 <= execution["deadline_seconds"] <= 86400
        and all(type(execution[k]) is int and 1 <= execution[k] <= 128
                for k in ("process_memory_gib", "native_box_ceiling_gib", "physical_memory_gib"))
        and type(execution["reserve_gib"]) is int
        and 8 <= execution["reserve_gib"] <= 64
        and execution["process_memory_gib"] + execution["reserve_gib"] <= execution["physical_memory_gib"]
        and execution["native_box_ceiling_gib"] <= execution["physical_memory_gib"],
        "Invalid execution resource bounds",
    )
    profile = {
        k: execution.get(k)
        for k in ("deadline_seconds", "environment", "process_memory_gib", "native_box_ceiling_gib", "physical_memory_gib", "reserve_gib")
    }
    require(exact(res.get("execution_profile"), profile), "Execution profile mismatch")
    require(
        type(profile["deadline_seconds"]) is int
        and profile["deadline_seconds"] > 0
        and isinstance(profile["environment"], dict)
        and all(
            type(profile[k]) in (int, float) and profile[k] > 0
            for k in ("process_memory_gib", "native_box_ceiling_gib", "physical_memory_gib", "reserve_gib")
        ),
        "Invalid execution bounds",
    )

    require(
        res.get("format") == "mlx-stream-adapter-generation-result-v1"
        and res.get("status") == "completed"
        and not res.get("error"),
        "Incomplete native result",
    )
    require(
        res.get("request_sha256")
        == sha(raws["request"])
        == audit.get("request_sha256")
        == native.get("request_sha256"),
        "Request binding mismatch",
    )
    require(
        audit.get("status") == "receipt_consistent"
        and audit.get("normal_eos") is True
        and audit.get("truncated") is False,
        "Unverified/unfinished generation",
    )
    require(
        audit.get("child_report_sha256") == sha(raws["result"]),
        "Result audit hash mismatch",
    )
    require(
        audit.get("native_construction_sha256")
        == sha(raws["construction"])
        == res.get("native_construction_sha256"),
        "Native construction binding mismatch",
    )
    require(
        native.get("format") == "mlx-stream-adapter-construction-v1",
        "Wrong construction format",
    )
    require(
        exact(res.get("prompt_ids"), ids), "Prompt IDs changed or prefix duplicated"
    )
    require(
        res.get("max_new_tokens") == 4096 and exact(res.get("policy"), POLICY),
        "Native settings mismatch",
    )
    for flag in (
        "chat_reformatted",
        "prompt_truncated",
        "tf32",
        "trainer_forward_parity",
    ):
        require(res.get(flag) is False, "Unsupported native flag: " + flag)
    for key in ("engine_sha256", "runtime_manifest_sha256", "payload_manifest_sha256"):
        require(
            isinstance(q.get(key), str)
            and re.fullmatch("[0-9a-f]{64}", q[key])
            and res.get(key) == q[key],
            "Unqualified engine/runtime/base identity",
        )
    require(
        req["inputs_sha256"].get(req["engine"]) == ENGINE
        and req["inputs_sha256"].get(req["runtime_manifest"])
        == q["runtime_manifest_sha256"]
        and req["inputs_sha256"].get(req["payload_manifest"])
        == q["payload_manifest_sha256"],
        "Request runtime/base mismatch",
    )
    adapter = res.get("adapter_sha256")
    require(
        isinstance(adapter, str) and re.fullmatch("[0-9a-f]{64}", adapter),
        "Invalid adapter digest",
    )
    approved = q.get("adapters", {}).get(adapter)
    require(
        isinstance(approved, dict)
        and approved.get("condition") in ("final458", "zeroB-final458")
        and approved.get("checkpoint_step") == 458,
        "Unbound final/control adapter",
    )
    require(
        isinstance(approved.get("config_sha256"), str)
        and re.fullmatch("[0-9a-f]{64}", approved["config_sha256"]),
        "Invalid config digest",
    )
    require(
        req["inputs_sha256"].get(str(Path(req["adapter_dir"]) / "adapter.safetensors"))
        == adapter,
        "Adapter payload binding mismatch",
    )
    require(
        req["inputs_sha256"].get(str(Path(req["adapter_dir"]) / "adapter_config.json"))
        == approved.get("config_sha256"),
        "Adapter config binding mismatch",
    )
    require(
        approved.get("rank") == 4
        and approved.get("alpha") == 8
        and approved.get("layer") == 39
        and approved.get("parameters") == 78848,
        "Adapter basis contract mismatch",
    )
    require(
        isinstance(approved.get("fingerprint"), str)
        and re.fullmatch("[0-9a-f]{64}", approved["fingerprint"])
        and native.get("adapter_fingerprint") == approved["fingerprint"],
        "Native adapter fingerprint mismatch",
    )
    for key in ("checkpoint_step", "rank", "alpha", "layer", "parameters"):
        require(type(approved.get(key)) is int, "Invalid adapter basis type")
    if approved["condition"] == "zeroB-final458":
        require(
            isinstance(approved.get("derived_from"), str)
            and re.fullmatch("[0-9a-f]{64}", approved["derived_from"])
            and approved["derived_from"] != adapter,
            "Invalid zero-B parent digest",
        )
        parent = q.get("adapters", {}).get(approved.get("derived_from"))
        require(
            isinstance(parent, dict)
            and parent.get("condition") == "final458"
            and all(
                type(parent.get(k)) is int and parent[k] == v
                for k, v in {
                    "checkpoint_step": 458,
                    "rank": 4,
                    "alpha": 8,
                    "layer": 39,
                    "parameters": 78848,
                }.items()
            )
            and all(
                isinstance(parent.get(k), str)
                and re.fullmatch("[0-9a-f]{64}", parent[k])
                for k in ("config_sha256", "fingerprint")
            ),
            "Unbound zero-B parent",
        )
    require(
        type(native.get("model_instance")) is int
        and type(audit.get("model_instance")) is int,
        "Invalid native owner type",
    )
    require(
        type(res.get("model_instance")) is int
        and res["model_instance"] > 0
        and res["model_instance"]
        == native.get("model_instance")
        == audit.get("model_instance"),
        "Native model owner mismatch",
    )
    require(
        exact(res.get("resolved_tier"), native.get("resolved_tier"))
        and exact(res.get("resolved_tier"), audit.get("resolved_tier"))
        and res.get("resolved_tier"),
        "Native route mismatch",
    )
    require(
        exact(res.get("logits_observer"), req.get("logits_observer"))
        and exact(native.get("logits_observer"), req.get("logits_observer")),
        "Observer policy mismatch",
    )
    observer = audit.get("logits_observer")
    require(isinstance(observer, dict), "Missing observer audit")
    spec = req.get("logits_observer")
    if spec is None:
        require(
            observer.get("enabled") is False and not observer.get("rows"),
            "Invalid disabled observer audit",
        )
    else:
        require(
            isinstance(spec, dict)
            and set(spec) == {"format", "positions"}
            and spec["format"] == "selected_raw_logits_v1",
            "Invalid observer spec",
        )
        positions = spec["positions"]
        require(
            isinstance(positions, list)
            and 1 <= len(positions) <= 8
            and all(type(x) is int and 774 <= x < 4870 for x in positions)
            and positions == sorted(set(positions)),
            "Invalid observer positions",
        )
        require(
            observer.get("enabled") is True
            and type(observer.get("rows_verified")) is int
            and observer["rows_verified"] == len(positions)
            and len(observer.get("rows", [])) == len(positions),
            "Incomplete observer audit",
        )
        for position, row in zip(positions, observer["rows"]):
            require(
                isinstance(row, dict)
                and type(row.get("position")) is int
                and row["position"] == position,
                "Observer row mismatch",
            )
            metadata_raw = read(row["metadata_path"], 65536)
            values_raw = read(row["raw_path"], 4 << 20)
            require(
                sha(metadata_raw) == row.get("metadata_sha256")
                and sha(values_raw) == row.get("raw_sha256"),
                "Observer file drift",
            )
            meta = strict(metadata_raw)
            offset = position - len(ids)
            require(offset < len(res["output_ids"]), "Observer missing generated token")
            prefix = ids + res["output_ids"][:offset]
            ph = sha(struct.pack("<" + "I" * len(prefix), *prefix))
            fields = {
                "format": "selected_raw_logits_v1",
                "request_sha256": sha(raws["request"]),
                "position": position,
                "prefix_ids_le_u32_sha256": ph,
                "vocab_size": q["model_vocab_size"],
                "serialized_dtype": "F32_LE",
                "raw_file": Path(row["raw_path"]).name,
                "raw_bytes": q["model_vocab_size"] * 4,
                "all_finite": True,
                "model_instance": res["model_instance"],
                "adapter_fingerprint": approved["fingerprint"],
                "observer_changes_timing": True,
                "raw_sha256": sha(values_raw),
            }
            require(
                all(exact(meta.get(k), v) for k, v in fields.items()),
                "Observer metadata binding mismatch",
            )
            require(
                len(values_raw) == q["model_vocab_size"] * 4
                and meta.get("source_dtype") in {"float32", "float16", "bfloat16"}
                and type(meta.get("rows_in_forward")) is int
                and 1 <= meta["rows_in_forward"] <= position,
                "Observer dimensions mismatch",
            )
            values = [x[0] for x in struct.iter_unpack("<f", values_raw)]
            require(all(math.isfinite(x) for x in values), "Nonfinite observer values")
            argmax = max(range(len(values)), key=values.__getitem__)
            require(
                type(meta.get("argmax")) is int
                and argmax == meta["argmax"] == res["output_ids"][offset],
                "Observer greedy mismatch",
            )
            require(
                all(
                    exact(row.get(k), v)
                    for k, v in {
                        "prefix_ids_le_u32_sha256": ph,
                        "vocab_size": len(values),
                        "source_dtype": meta["source_dtype"],
                        "argmax": argmax,
                    }.items()
                ),
                "Observer audit mismatch",
            )
    require(
        res.get("arithmetic_profile")
        == req.get("arithmetic_profile")
        == "mlx_stream_served_transfer_v1",
        "Arithmetic identity mismatch",
    )
    require(
        exact(res.get("serving_profile"), req.get("serving_profile"))
        and isinstance(req.get("serving_profile"), dict),
        "Serving policy mismatch",
    )
    serving = {
        "name": "served_transfer_v1",
        "warmup": "serial_owned_v1",
        "draft_policy": "disabled_weights_retained",
        "user_settings_policy": "ignored_fixed_defaults_v1",
        "embedding_host_rows": False,
        "cross_request_prefix_cache": False,
    }
    require(exact(req.get("serving_profile"), serving), "Wrong served profile")
    for key, value in {
        **{k: v for k, v in serving.items() if k != "name"},
        "trainer_forward_parity": False,
    }.items():
        require(
            exact(native.get(key), value), "Native construction policy mismatch: " + key
        )
    eos = req.get("eos_token_ids")
    require(
        isinstance(eos, list)
        and eos
        and all(type(x) is int for x in eos)
        and len(set(eos)) == len(eos),
        "Invalid EOS contract",
    )
    require(
        exact(res.get("eos_token_ids"), eos) and exact(q.get("eos_token_ids"), eos),
        "EOS contract drift",
    )
    valid = set(tokenizer.get_vocab().values())
    vocab = q["model_vocab_size"]
    require(all(x in valid and 0 <= x < vocab for x in eos), "EOS out of bounds")
    for key in ("loaded_embedding_vocab", "loaded_head_vocab"):
        require(
            type(res.get(key)) is int
            and type(native.get(key)) is int
            and res.get(key) == vocab == native.get(key),
            "Loaded vocabulary mismatch",
        )
    output = res.get("output_ids")
    require(
        isinstance(output, list)
        and 1 <= len(output) <= 4096
        and all(type(x) is int and x in valid and 0 <= x < vocab for x in output),
        "Invalid output IDs",
    )
    require(
        res.get("stop_reason") == "eos"
        and [i for i, x in enumerate(output) if x in eos] == [len(output) - 1],
        "Missing/early EOS or truncated generation",
    )
    require(output[: len(ids)] != ids, "Duplicated prompt prefix")
    decoded = tokenizer.decode(
        output, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    require(
        isinstance(res.get("raw_text"), str) and decoded == res["raw_text"],
        "Native/tokenizer decoded suffix drift",
    )
    require(
        decoded.endswith("<｜end▁of▁sentence｜>")
        and "<｜Assistant｜>" not in decoded
        and "<｜User｜>" not in decoded,
        "Unexpected native suffix boundary",
    )
    return decoded.encode("utf-8"), {
        "condition": approved["condition"],
        "tokens": len(output),
        "output_ids_sha256": sha(json.dumps(output, separators=(",", ":")).encode()),
        "model_vocab_size": vocab,
        "adapter_sha256": adapter,
        "qualification_sha256": sha(raws["qualification"]),
    }


def observer_envelope(records):
    audit = records["audit"].get("logits_observer")
    require(isinstance(audit, dict), "Invalid observer audit envelope")
    rows = audit.get("rows", [])
    require(isinstance(rows, list) and len(rows) <= 8, "Observer row capture bound")
    spec = records["request"].get("logits_observer")
    if spec is None:
        require(
            audit.get("enabled") is False and not rows, "Disabled observer has rows"
        )
        return []
    require(
        isinstance(spec, dict)
        and set(spec) == {"format", "positions"}
        and spec["format"] == "selected_raw_logits_v1",
        "Invalid observer spec",
    )
    pos = spec["positions"]
    require(
        isinstance(pos, list)
        and 1 <= len(pos) <= 8
        and all(type(x) is int and 774 <= x < 4870 for x in pos)
        and pos == sorted(set(pos)),
        "Invalid observer positions",
    )
    require(
        audit.get("enabled") is True
        and type(audit.get("rows_verified")) is int
        and audit["rows_verified"] == len(pos) == len(rows),
        "Incomplete observer envelope",
    )
    require(
        all(
            isinstance(row, dict)
            and type(row.get("position")) is int
            and row["position"] == position
            for row, position in zip(rows, pos)
        ),
        "Observer position envelope mismatch",
    )
    return rows


def capture(manifest_path, expected, output, *, tokenizer_factory=None):
    manifest_raw = read(manifest_path)
    require(sha(manifest_raw) == expected, "Manifest hash mismatch")
    m = strict(manifest_raw)
    require(
        isinstance(m, dict)
        and set(m) == {"format", "files", "tokenizer_dir", "tokenizer_identity"}
        and m["format"] == FORMAT,
        "Invalid capture manifest",
    )
    require(
        isinstance(m["files"], dict) and set(m["files"]) == set(ROLES),
        "Incomplete input inventory",
    )
    raws = {}
    paths = {}
    for role, entry in m["files"].items():
        require(
            isinstance(entry, dict) and set(entry) == {"path", "sha256"},
            "Invalid file binding",
        )
        p = Path(entry["path"])
        p = p if p.is_absolute() else Path(manifest_path).absolute().parent / p
        b = read(p)
        require(sha(b) == entry["sha256"], "Captured input digest mismatch")
        raws[role] = b
        paths[role] = p
    out = Path(output).absolute()
    require(out.parent.resolve(strict=True) == out.parent, "Unsafe output parent")
    out.mkdir(exist_ok=False)
    for role, b in raws.items():
        (out / (role + ".captured.json")).write_bytes(b)
    receipt = {
        "format": FORMAT,
        "accepted": False,
        "manifest_sha256": expected,
        "input_sha256": {k: sha(v) for k, v in raws.items()},
        "GPU_executed": False,
        "trainer_forward_parity": False,
        "scope": "Hash-bound receipt and decoded-suffix consistency only; external device admission is not independently authenticated",
    }
    try:
        records = {k: strict(v) for k, v in raws.items()}
        # Validate local tokenizer inventory before loading; never import model payloads.
        from exl3_mlx.data import load_tokenizer, tokenizer_identity

        path = Path(m["tokenizer_dir"])
        require(
            path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(),
            "Noncanonical tokenizer directory",
        )
        names = {
            "tokenizer.json": 32 << 20,
            "tokenizer_config.json": 4 << 20,
            "chat_template.jinja": 1 << 20,
        }
        require(
            isinstance(m["tokenizer_identity"], dict)
            and set(m["tokenizer_identity"]) == set(names),
            "Expected fixed three-file tokenizer",
        )
        captured_tokenizer = out / "tokenizer"
        captured_tokenizer.mkdir()
        tokenizer_raw = {
            name: read(path / name, limit) for name, limit in names.items()
        }
        for name, b in tokenizer_raw.items():
            require(
                sha(b) == m["tokenizer_identity"][name],
                "Tokenizer capture digest mismatch",
            )
            (captured_tokenizer / name).write_bytes(b)
        identity = tokenizer_identity(captured_tokenizer)
        require(
            identity == m["tokenizer_identity"] == records["prepared"]["tokenizer"],
            "Tokenizer identity mismatch",
        )
        require(
            records["result"].get("tokenizer_sha256") == identity["tokenizer.json"],
            "Native tokenizer mismatch",
        )
        evidence = records["qualification"].get("evidence")
        require(
            isinstance(evidence, list) and 1 <= len(evidence) <= 32,
            "Missing device qualification evidence",
        )
        evidence_pins = {}
        for ref in evidence:
            require(
                isinstance(ref, dict) and set(ref) == {"path", "sha256"},
                "Invalid qualification evidence binding",
            )
            ep = Path(ref["path"])
            require(ep.is_absolute(), "Qualification evidence path must be absolute")
            eb = read(ep)
            require(sha(eb) == ref["sha256"], "Qualification evidence drift")
            evidence_pins[str(ep)] = ref["sha256"]
        tok = (tokenizer_factory or load_tokenizer)(captured_tokenizer)
        require(
            str(paths["prompt_ids"]) == records["request"].get("prompt_ids_file"),
            "Captured prompt file path mismatch",
        )
        observer_pins = {}
        for i, row in enumerate(observer_envelope(records)):
            for kind, limit in (("metadata", 65536), ("raw", 4 << 20)):
                rp = row[kind + "_path"]
                rb = read(rp, limit)
                require(sha(rb) == row[kind + "_sha256"], "Observer capture drift")
                (out / f"observer-{i}-{kind}.captured").write_bytes(rb)
                observer_pins[rp] = (sha(rb), limit)
        decoded, info = validate(records, raws, tok)
        require(
            all(sha(read(p, limit)) == h for p, (h, limit) in observer_pins.items()),
            "Observer changed during capture",
        )
        receipt["observer_input_sha256"] = {p: h for p, (h, _) in observer_pins.items()}
        require(
            all(
                read(path / name, names[name]) == b
                and read(captured_tokenizer / name, names[name]) == b
                for name, b in tokenizer_raw.items()
            ),
            "Tokenizer drift",
        )
        require(
            all(sha(read(Path(ep))) == h for ep, h in evidence_pins.items()),
            "Qualification evidence changed",
        )
        receipt["qualification_evidence_sha256"] = evidence_pins
        require(
            read(manifest_path) == manifest_raw
            and all(read(paths[k]) == v for k, v in raws.items()),
            "Input drift",
        )
        (out / "generation.raw").write_bytes(decoded)
        receipt.update(accepted=True, raw_sha256=sha(decoded), **info)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        receipt["rejection"] = str(exc)
    receipt["receipt_sha256"] = sha(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    )
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    try:
        r = capture(a.manifest, a.manifest_sha256, a.output)
    except (ValueError, OSError) as e:
        p.exit(2, str(e) + "\n")
    print(json.dumps(r, indent=2))
    raise SystemExit(0 if r["accepted"] else 1)
