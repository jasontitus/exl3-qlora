"""Local closure contracts with mocked compiler only; no Metal/GPU invocation."""

import json
import subprocess
from pathlib import Path

import pytest

from exl3_mlx import metal_compile
from exl3_mlx import metal_compile_closure as mc


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    primary = root / "main.metal"
    primary.write_text('#include "inc/a.h"\nkernel void k() {}\n')
    header = root / "a.h"
    header.write_text("#include <metal_stdlib>\n")
    value = {
        "format": mc.FORMAT,
        "standard": "metal4.0",
        "sdk_mpp": False,
        "units": [
            {
                "id": "before",
                "revision": "a" * 40,
                "primary": "main.metal",
                "files": {
                    "main.metal": {"path": str(primary), "sha256": mc.sha(primary.read_bytes())},
                    "inc/a.h": {"path": str(header), "sha256": mc.sha(header.read_bytes())},
                },
            }
        ],
    }
    manifest = root / "manifest.json"
    out = tmp_path / "result"
    compiler = tmp_path / "metal"
    compiler.write_bytes(b"fake executable")
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    calls = []

    def run(argv, env, cwd, timeout=90):
        calls.append(argv)
        assert set(env) == {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"}
        stdout = ""
        if "--find" in argv:
            stdout = str(compiler)
        elif "--show-sdk-path" in argv:
            stdout = str(sdk)
        elif "-c" in argv:
            Path(argv[-1]).write_bytes(b"AIR")
        return dict(argv=argv, exit_code=0, error=None, stdout=stdout, stderr="")

    monkeypatch.setattr(mc, "_run", run)

    def write():
        manifest.write_text(json.dumps(value))
        return mc.sha(manifest.read_bytes())

    return value, manifest, out, write, calls, primary, header, sdk


def test_success_and_old_cli_dispatch(setup):
    v, p, o, write, calls, *_ = setup
    h = write()
    result = mc.compile_closure(p, h, o)
    assert result["compiled"] and not result["execution_verified"]
    assert result["units"][0]["include_edges"] == {"main.metal": ["inc/a.h"], "inc/a.h": []}
    assert "-O3" in calls[-1] and "-fno-fast-math" in calls[-1]
    assert result["units"][0]["revision_label"] == "a" * 40
    with pytest.raises(FileExistsError):
        mc.compile_closure(p, h, o)
    assert (
        metal_compile.main(
            [
                "closure",
                "--manifest",
                str(p),
                "--manifest-sha256",
                h,
                "--output",
                str(o.parent / "cli"),
            ]
        )
        == 0
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(sdk_mpp=1),
        lambda v: v.update(units=[]),
        lambda v: v["units"][0].update(revision="main"),
        lambda v: v["units"].append(v["units"][0]),
        lambda v: v["units"][0].update(files=[]),
        lambda v: v["units"][0]["files"].update({"../bad.h": {"path": "x", "sha256": "a" * 64}}),
        lambda v: v["units"][0]["files"].update(
            {"MAIN.metal": v["units"][0]["files"]["main.metal"]}
        ),
        lambda v: v.update(sdk_mpp=True, standard="metal3.1"),
    ],
)
def test_metadata_no_subprocess(setup, change):
    v, p, o, write, calls, *_ = setup
    change(v)
    with pytest.raises(ValueError):
        mc.compile_closure(p, write(), o)
    assert not calls and not o.exists()


@pytest.mark.parametrize(
    "line",
    [
        "#include BAD",
        '#include "/etc/passwd"',
        '#include "../escape.h"',
        "#include_next <metal_stdlib>",
        "#import <metal_stdlib>",
        '#embed "inc/a.h"',
        "#include <stdio.h>",
        "#include <metal_fiction>",
        '#include "missing.h"',
        '%:include "inc/a.h"',
        '??=include "inc/a.h"',
        '#if __has_include("/etc/passwd")\n#endif',
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>",
        "#inc\\\nlude BAD",
        "#include /*comment*/ BAD",
        "\f#include BAD",
        "\r#include BAD",
    ],
)
def test_bad_includes_no_compiler(setup, line):
    v, p, o, write, calls, primary, *_ = setup
    primary.write_text(line + "\n")
    v["units"][0]["files"]["main.metal"]["sha256"] = mc.sha(primary.read_bytes())
    with pytest.raises(ValueError):
        mc.compile_closure(p, write(), o)
    assert not calls and not o.exists()


def test_relative_parent_resolution(setup):
    v, p, o, write, _, primary, header, _ = setup
    v["units"][0]["primary"] = "dir/main.metal"
    v["units"][0]["files"]["dir/main.metal"] = v["units"][0]["files"].pop("main.metal")
    primary.write_text('#include "../inc/a.h"\n')
    v["units"][0]["files"]["dir/main.metal"]["sha256"] = mc.sha(primary.read_bytes())
    assert mc.compile_closure(p, write(), o)["compiled"]


@pytest.mark.parametrize("bad", [b"\xff", b"#include <metal_stdlib>\n\0"])
def test_bad_encoding(setup, bad):
    v, p, o, write, calls, _, header, _ = setup
    header.write_bytes(bad)
    v["units"][0]["files"]["inc/a.h"]["sha256"] = mc.sha(bad)
    with pytest.raises(ValueError):
        mc.compile_closure(p, write(), o)
    assert not calls


def test_duplicate_json_and_wrong_digest(setup):
    _, p, o, write, calls, *_ = setup
    write()
    with pytest.raises(ValueError):
        mc.compile_closure(p, "0" * 64, o)
    p.write_text('{"format":1,"format":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        mc.compile_closure(p, mc.sha(p.read_bytes()), o)
    p.write_text('{"format":NaN}')
    with pytest.raises(ValueError, match="Nonfinite"):
        mc.compile_closure(p, mc.sha(p.read_bytes()), o)
    assert not calls


def test_original_symlink_and_drift_rejected(setup):
    v, p, o, write, calls, primary, header, _ = setup
    header.unlink()
    header.symlink_to(primary)
    with pytest.raises(ValueError):
        mc.compile_closure(p, write(), o)
    assert not calls


def test_caps_before_read(setup, monkeypatch):
    _, p, o, write, calls, *_ = setup
    monkeypatch.setattr(mc, "MAX_TOTAL", 1)
    with pytest.raises(ValueError, match="before read"):
        mc.compile_closure(p, write(), o)
    assert not calls


@pytest.mark.parametrize(
    "failure",
    ["empty", "missing", "symlink", "source", "extra", "exit", "timeout", "compiler", "oldair"],
)
def test_failure_receipt(setup, monkeypatch, failure):
    v, p, o, write, calls, *_ = setup
    if failure == "oldair":
        u = json.loads(json.dumps(v["units"][0]))
        u["id"] = "after"
        v["units"].append(u)
    old = mc._run

    def run(argv, *args, **kw):
        result = old(argv, *args, **kw)
        if "-c" in argv:
            air = Path(argv[-1])
            if failure == "empty":
                air.write_bytes(b"")
            elif failure == "missing":
                air.unlink()
            elif failure == "symlink":
                air.unlink()
                air.symlink_to(p)
            elif failure == "source":
                (air.parent / "sources/inc/a.h").write_bytes(b"drift")
            elif failure == "extra":
                (air.parent / "sources/extra.h").write_bytes(b"new")
            elif failure == "exit":
                result["exit_code"] = 1
            elif failure == "timeout":
                result.update(exit_code=None, error="timeout")
            elif failure == "compiler":
                Path(argv[0]).write_bytes(b"drift")
            elif failure == "oldair" and air.parent.name == "after":
                (o / "before/output.air").write_bytes(b"drift")
        return result

    monkeypatch.setattr(mc, "_run", run)
    report = mc.compile_closure(p, write(), o)
    assert not report["compiled"] and (o / "receipt.json").is_file()


def test_sdk_mpp_headers_bound_and_drift(setup, monkeypatch):
    v, p, o, write, calls, _, header, sdk = setup
    v["sdk_mpp"] = True
    header.write_text("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n")
    v["units"][0]["files"]["inc/a.h"]["sha256"] = mc.sha(header.read_bytes())
    hdr = (
        sdk
        / "System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/MetalPerformancePrimitives.h"
    )
    hdr.parent.mkdir(parents=True)
    hdr.write_bytes(b"installed")
    old = mc._run

    def run(argv, *args, **kw):
        result = old(argv, *args, **kw)
        if "-c" in argv:
            hdr.write_bytes(b"drift")
        return result

    monkeypatch.setattr(mc, "_run", run)
    r = mc.compile_closure(p, write(), o)
    assert not r["compiled"] and r["sdk_headers"] and "-F" in calls[-1]


def test_subprocess_environment_and_timeout(monkeypatch, tmp_path):
    def run(argv, **kwargs):
        assert kwargs["env"] == {"PATH": "/usr/bin"} and "shell" not in kwargs
        raise subprocess.TimeoutExpired(argv, 90, output=b"partial")

    monkeypatch.setattr(subprocess, "run", run)
    assert mc._run(["/tool"], {"PATH": "/usr/bin"}, tmp_path)["error"] == "timeout"


def test_second_unit_rejected_before_any_tools(setup):
    v, p, o, write, calls, *_ = setup
    bad = json.loads(json.dumps(v["units"][0]))
    bad["id"] = "after"
    bad["files"]["main.metal"]["sha256"] = "0" * 64
    v["units"].append(bad)
    with pytest.raises(ValueError, match="SHA-256"):
        mc.compile_closure(p, write(), o)
    assert not calls and not o.exists()


def test_source_count_and_unreachable_file_rejected(setup):
    v, p, o, write, calls, _, header, _ = setup
    v["units"][0]["files"]["extra.h"] = {"path": str(header), "sha256": mc.sha(header.read_bytes())}
    with pytest.raises(ValueError, match="unreachable"):
        mc.compile_closure(p, write(), o)
    for i in range(130):
        v["units"][0]["files"][f"extra{i}.h"] = v["units"][0]["files"]["extra.h"]
    with pytest.raises(ValueError, match="128"):
        mc.compile_closure(p, write(), o)
    assert not calls


def test_sdk_inventory_count_and_bytes(tmp_path, monkeypatch):
    hdr = tmp_path / "System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers"
    hdr.mkdir(parents=True)
    (hdr / "MetalPerformancePrimitives.h").write_bytes(b"abc")
    monkeypatch.setattr(mc, "MAX_UNIT", 2)
    with pytest.raises(ValueError):
        mc._sdk_headers(tmp_path)
    monkeypatch.setattr(mc, "MAX_UNIT", 1000)
    for i in range(64):
        (hdr / f"{i}.h").write_bytes(b"x")
    with pytest.raises(ValueError, match="budget"):
        mc._sdk_headers(tmp_path)


def test_air_size_cap(setup, monkeypatch):
    _, p, o, write, *_ = setup
    old = mc._run

    def run(argv, *a, **kw):
        r = old(argv, *a, **kw)
        if "-c" in argv:
            with Path(argv[-1]).open("wb") as f:
                f.truncate((128 << 20) + 1)
        return r

    monkeypatch.setattr(mc, "_run", run)
    assert not mc.compile_closure(p, write(), o)["compiled"]


def test_case_aliased_unit_rejected(setup):
    v, p, o, write, calls, *_ = setup
    other = json.loads(json.dumps(v["units"][0]))
    other["id"] = "BEFORE"
    v["units"].append(other)
    with pytest.raises(ValueError, match="unit ID"):
        mc.compile_closure(p, write(), o)
    assert not calls


@pytest.mark.parametrize("space", [" ", "\t", "\f", "\v", " \t"])
def test_whitespace_line_splice_rejected_before_tools(setup, space):
    v, p, o, write, calls, primary, *_ = setup
    primary.write_text("#inc\\" + space + '\nlude "/unbound.h"\n#include "inc/a.h"\n')
    v["units"][0]["files"]["main.metal"]["sha256"] = mc.sha(primary.read_bytes())
    with pytest.raises(ValueError, match="line splice"):
        mc.compile_closure(p, write(), o)
    assert not calls and not o.exists()


@pytest.mark.parametrize("key", [mc.MPP, mc.MPP.lower()])
def test_local_mpp_shadow_rejected(setup, key):
    v, p, o, write, calls, _, header, _ = setup
    v["sdk_mpp"] = True
    v["units"][0]["files"][key] = {"path": str(header), "sha256": mc.sha(header.read_bytes())}
    with pytest.raises(ValueError, match="shadows"):
        mc.compile_closure(p, write(), o)
    assert not calls


def test_sdk_total_cap_prevents_second_read(tmp_path, monkeypatch):
    hdr = tmp_path / "System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers"
    hdr.mkdir(parents=True)
    for name in ["MetalPerformancePrimitives.h", "z.h"]:
        (hdr / name).write_bytes(b"1234")
    monkeypatch.setattr(mc, "MAX_UNIT", 5)
    original = mc._read
    reads = []

    def read(p, limit):
        reads.append(str(p))
        return original(p, limit)

    monkeypatch.setattr(mc, "_read", read)
    with pytest.raises(ValueError, match="before read"):
        mc._sdk_headers(tmp_path)
    assert len(reads) == 1


def test_sdk_count_cap_prevents_65th_read(tmp_path, monkeypatch):
    hdr = tmp_path / "System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers"
    hdr.mkdir(parents=True)
    for name in ["MetalPerformancePrimitives.h"] + [f"z{i:02}.h" for i in range(64)]:
        (hdr / name).write_bytes(b"x")
    original = mc._read
    reads = []

    def read(p, limit):
        reads.append(str(p))
        return original(p, limit)

    monkeypatch.setattr(mc, "_read", read)
    with pytest.raises(ValueError, match="before read"):
        mc._sdk_headers(tmp_path)
    assert len(reads) == 64
