"""Independent CPU source/receipt reconstruction; does not compile or execute shaders."""

import datetime
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

R = Path(__file__).resolve().parent
repo = R.parent / "engines-a/mirrors/ds4.git"
hashes = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def bind(p):
    p = Path(p).resolve()
    raw = p.read_bytes()
    hashes[str(p)] = sha(raw)
    return raw


def git(ref, path):
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", ref + ":" + path],
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"},
    )


def prelude(host):
    text = host.decode()
    start = text.index("static const char *ds4_gpu_source =") + len(
        "static const char *ds4_gpu_source ="
    )
    end = text.index("\nstatic NSString *ds4_gpu_full_source", start)
    out = []
    for line in text[start:end].strip().splitlines():
        v = line.strip().removesuffix(";")
        if not v:
            continue
        assert v[0] == v[-1] == '"'
        body = v[1:-1]
        i = 0
        s = ""
        while i < len(body):
            if body[i] == "\\":
                i += 1
                assert body[i] in ["n", '"']
                s += "\n" if body[i] == "n" else '"'
            else:
                s += body[i]
            i += 1
        out.append(s)
    return "".join(out).encode()


rows = []
for filename, complete in [
    ("ds4-compiler-receipt.json", False),
    ("ds4-complete-compiler-receipt.json", True),
]:
    report = json.loads(bind(R / filename))
    for row in report["results"]:
        cid = row["case_id"]
        side = row["side"]
        ref = row["revision"]
        source = json.loads(
            bind(R.parent / "new-core-32768" / (cid + ".provenance.json"))
        )["source"]
        assert ref == source["parent" if side == "before" else "head"]["commit"]
        host = git(ref, "ds4_metal.m")
        assert sha(host) == row["host_sha256"]
        p = prelude(host)
        assert sha(p) == row["prelude_sha256"]
        scoped = R / "ds4-scoped" / cid / side
        assert bind(scoped / "runtime-host.m.txt") == host
        assert bind(scoped / "runtime-prelude.metal") == p
        if complete:
            body = host.decode().split("required_sources = @[", 1)[1].split("];", 1)[0]
            specs = re.findall(
                r'@\[@"(DS4_[A-Z0-9_]+)",\s*@"(metal/[^"\n]+\.metal)"\]', body
            )
            names = [x[1] for x in specs]
            assert len(names) == 19 and names == row["exact_source_order"]
            assert set(names) == set(row["source_sha256"])
            # Match actual appendFormat, including its final newline for EACH loaded file.
            tu = p
            for name in names:
                raw = git(ref, name)
                assert sha(raw) == row["source_sha256"][name]
                assert bind(R / "ds4-complete" / cid / side / name) == raw
                tu += b"\n// appended " + name.encode() + b"\n" + raw + b"\n"
            assert row["exit_code"] == 0
        else:
            raw = git(ref, row["file"])
            assert raw == source["files"][row["file"]][side]["content"].encode()
            assert sha(raw) == row["source_sha256"]
            assert bind(scoped / "kernel-source.metal") == raw
            tu = p + b"\n" + raw
            assert row["exit_code"] == (0 if row["file"] == "metal/cpy.metal" else 1)
        argv = row["argv"]
        assert argv[:5] == ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0"]
        assert bind(argv[argv.index("-c") + 1]) == tu
        assert sha(tu) == row["translation_unit_sha256"]
        air = Path(argv[argv.index("-o") + 1])
        air_sha = None
        if row["exit_code"] == 0:
            ir = bind(air)
            assert len(ir) > 100
            air_sha = sha(ir)
        rows.append(
            {
                "case_id": cid,
                "side": side,
                "configuration": "complete_default_runtime"
                if complete
                else "prelude_plus_single_file",
                "exit_code": row["exit_code"],
                "translation_unit_sha256": sha(tu),
                "output_air_sha256": air_sha,
                "source_files": 19 if complete else 1,
            }
        )
for name in [
    "compile_ds4.py",
    "compile_ds4_complete.py",
    "ds4-complete-initial-whitespace-variant.json",
]:
    bind(R / name)
version = subprocess.check_output(
    ["xcrun", "-sdk", "macosx", "metal", "--version"], text=True
).strip()
result = {
    "schema": "ds4-compiler-independent-review-v1",
    "status": "passed_scoped_audit",
    "passed": True,
    "reviewer": "adversarial_code",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "results": rows,
    "counts": {
        "compile_invocations_reviewed": 6,
        "successful": 4,
        "failed_missing_dependency_scopes": 2,
        "complete_runtime_pairs": 1,
        "complete_files_per_side": 19,
    },
    "compiler_observed_at_review": version,
    "findings": [
        {
            "severity": "medium",
            "status": "addressed",
            "description": "Initial complete concat omitted the trailing newline in each runtime appendFormat. Parent preserved the initial whitespace-variant receipt, corrected composition and reran both complete compiles; independently reconstructed exact corrected bytes now match.",
        }
    ],
    "scope": [
        "Pinned parent/head revisions, complete host bytes, independently decoded C-string prelude and exact ordered19-file default concatenation verified against Git.",
        "Both cpy scoped translation units compile successfully. Both isolated RoPE scopes fail for missing dependency context; these failures remain recorded. The complete default runtime pair compiles successfully.",
        "Successful AIR artifacts are now hash-bound by this independent receipt. Compiler exit/status and diagnostics are reviewed author-run receipts; this reviewer did not rerun compiler invocations or create a Metal device.",
    ],
    "limits": [
        "Metal4 default preprocessor configuration only; no tensor-feature variants, function-constant pipeline creation, host linkage, runtime compile-option equivalence, GPU dispatch, numerical parity or performance qualification.",
        "The full runtime source compilation does not compile the commit’s Objective-C/C host edits and does not validate the new RoPE fence or stale-state behavior.",
        "Reconstruction scripts are one-off evidence tools, not hardened general build systems. Future runs must revalidate cached prelude/host inputs against pinned Git as this audit did.",
    ],
    "inputs_sha256": hashes,
    "review_script_sha256": sha(Path(__file__).read_bytes()),
}
(R / "ds4-independent-review.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            "status": result["status"],
            "counts": result["counts"],
            "sha256": sha((R / "ds4-independent-review.json").read_bytes()),
        }
    )
)
