import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from exl3_mlx import kernel_execution as e
from exl3_mlx import kernel_replay as r
from exl3_mlx.kernel_eval_suite import Reader, build_report


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def put(path, value):
    path.write_text(json.dumps(value))
    return pin(path)


def predicate(path, value):
    return {"path": path, "equals": value}


def outcome(status, code=0, category="unspecified"):
    return {
        "when": [predicate(["status"], status)],
        "exit_codes": [code],
        "status": status,
        "outcome_class": category,
    }


SCRIPT = """import hashlib,json,pathlib,sys,time
source,out=map(pathlib.Path,sys.argv[1:])
data=json.loads(source.read_text());out.mkdir()
if data.get('sleep'):time.sleep(data['sleep'])
if data.get('missing'):sys.exit(0)
result={'format':'test-only-backend-v1','input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'status':data.get('status','passed')}
(out/'result.json').write_text(json.dumps(result))
sys.exit(data.get('exit',0))
"""


def fixture(tmp_path, stage="compile", data=None):
    capture = tmp_path / "captured.json"
    capture_ref = put(capture, {"synthetic": True, "completed": True})
    stages = {
        name: {"status": "passed", "evidence": [capture_ref]}
        for name in e.STAGES[: e.STAGES.index(stage)]
    }
    initial = {
        "format": "kernel-eval-result-v1",
        "task_id": "task",
        "adapter_id": "adapter",
        "task_sha256": "1" * 64,
        "adapter_sha256": "2" * 64,
        "prompt_sha256": "3" * 64,
        "protocol_sha256": "4" * 64,
        "budget": {"max_output_tokens": 4096, "attempts": 1},
        "attempt": 1,
        "tracks": {"strict": stages},
    }
    initial_ref = put(tmp_path / "initial.json", initial)
    script = tmp_path / "trusted_backend.py"
    script.write_text(SCRIPT)
    inp = tmp_path / "input.json"
    input_ref = put(inp, data or {})
    backend = {
        "format": "kernel-eval-backend-v1",
        "kind": "cpu",
        "executable": pin(Path(sys.executable).resolve()),
        "code": [pin(script)],
        "argv": [str(script), "{input}", "{output}"],
        "cwd": str(tmp_path),
        "env": {},
        "timeout_seconds": 10,
        "result": {
            "path": "{output}/result.json",
            "bindings": [
                predicate(["format"], "test-only-backend-v1"),
                predicate(["input_sha256"], "{input_sha256}"),
            ],
            "stages": {stage: [outcome("passed"), outcome("failed", 1, "model_output")]},
        },
    }
    backend_ref = put(tmp_path / "backend.json", backend)
    plan = {
        "format": "kernel-eval-execution-v1",
        "initial_result": initial_ref,
        "track": "strict",
        "steps": [{"backend": backend_ref, "input": input_ref}],
    }
    plan_ref = put(tmp_path / "plan.json", plan)
    return plan_ref, plan, backend, initial


def refresh(tmp_path, plan, backend):
    plan["steps"][0]["backend"] = put(tmp_path / "backend.json", backend)
    return put(tmp_path / "plan.json", plan)


def run(tmp_path, ref, **kwargs):
    return e.execute(ref["path"], ref["sha256"], tmp_path / "run", **kwargs)


def test_default_does_not_spawn_or_create_output(tmp_path):
    ref, _, _, _ = fixture(tmp_path)
    with patch.object(e.subprocess, "Popen", side_effect=AssertionError("no spawn")):
        result = run(tmp_path, ref)
    assert result["executed"] is False and not (tmp_path / "run").exists()


def test_real_trusted_subprocess_to_suite_and_no_overwrite(tmp_path):
    ref, _, _, _ = fixture(tmp_path)
    result = run(tmp_path, ref, run=True)
    assert result["tracks"]["strict"]["compile"]["status"] == "passed"
    output_ref = pin(tmp_path / "run/result.json")
    suite = {
        "format": "kernel-eval-suite-v1",
        "id": "synthetic-test",
        "tasks": [
            {
                "id": "task",
                "sha256": "1" * 64,
                "prompt_sha256": "3" * 64,
                "protocol_sha256": "4" * 64,
            }
        ],
        "adapters": [{"id": "adapter", "sha256": "2" * 64}],
        "budget": result["budget"],
        "results": [output_ref],
    }
    assert build_report(suite, Reader(tmp_path))["results"][0] == result
    with pytest.raises(FileExistsError):
        run(tmp_path, ref, run=True)


def test_exit_zero_without_semantic_result_is_infrastructure(tmp_path):
    ref, _, _, _ = fixture(tmp_path, data={"missing": True})
    result = run(tmp_path, ref, run=True)["tracks"]["strict"]["compile"]
    assert result["status"] == "refused" and result["outcome_class"] == "infrastructure"
    receipt = json.loads((tmp_path / "run/00-compile/execution.json").read_text())
    assert receipt["exit_code"] == 0 and receipt["timed_out"] is False


def test_completed_numeric_failure_preserved_and_timing_not_started(tmp_path):
    ref, plan, _, _ = fixture(tmp_path, stage="numerical", data={"status": "failed", "exit": 1})
    # Even a deliberately invalid second executable isn't needed: stage one
    # failure stops before a later trusted backend can spawn.
    plan["steps"].append(plan["steps"][0].copy())
    ref = put(tmp_path / "plan.json", plan)
    result = run(tmp_path, ref, run=True)
    assert result["tracks"]["strict"]["numerical"]["outcome_class"] == "model_output"
    assert result["tracks"]["strict"]["numerical"]["status"] == "failed"
    assert len(result["execution"]["runs"]) == 1


def test_timing_requires_prior_numerical_pass(tmp_path):
    ref, plan, backend, initial = fixture(tmp_path, stage="timing")
    backend["result"]["stages"]["timing"] = [
        dict(outcome("passed"), measurements_path=["measurements"])
    ]
    initial["tracks"]["strict"]["numerical"] = {
        "status": "failed",
        "outcome_class": "model_output",
        "evidence": [pin(tmp_path / "captured.json")],
    }
    plan["initial_result"] = put(tmp_path / "initial.json", initial)
    ref = refresh(tmp_path, plan, backend)
    with patch.object(e.subprocess, "Popen", side_effect=AssertionError("timing forbidden")):
        result = run(tmp_path, ref, run=True)
    assert result["execution"]["runs"] == []


def test_cpu_timeout_is_infrastructure_not_candidate_failure(tmp_path):
    ref, plan, backend, _ = fixture(tmp_path, data={"sleep": 5})
    backend["timeout_seconds"] = 0.05
    ref = refresh(tmp_path, plan, backend)
    result = run(tmp_path, ref, run=True)
    assert result["tracks"]["strict"]["compile"]["outcome_class"] == "infrastructure"
    receipt = json.loads((tmp_path / "run/00-compile/execution.json").read_text())
    assert receipt["timed_out"] is True


def test_device_controller_requires_explicit_flag_and_owns_timeout(tmp_path):
    ref, plan, backend, _ = fixture(tmp_path, stage="numerical")
    backend["kind"] = "owned_device_controller"
    backend["timeout_seconds"] = None
    ref = refresh(tmp_path, plan, backend)
    with patch.object(e.subprocess, "Popen", side_effect=AssertionError("device forbidden")):
        with pytest.raises(ValueError, match="explicit"):
            run(tmp_path, ref, run=True)
    backend["timeout_seconds"] = 1
    ref = refresh(tmp_path, plan, backend)
    with pytest.raises(ValueError, match="own timeout"):
        run(tmp_path, ref)


def test_hash_predicate_type_and_ambiguous_outcome_rejections(tmp_path):
    ref, plan, backend, _ = fixture(tmp_path)
    backend["result"]["bindings"][0] = predicate(["status"], True)
    ref = refresh(tmp_path, plan, backend)
    assert run(tmp_path, ref, run=True)["tracks"]["strict"]["compile"]["status"] == "refused"
    assert not e.equal(True, 1) and not e.equal(1, 1.0)
    backend["result"]["bindings"][0] = predicate(["format"], "test-only-backend-v1")
    backend["result"]["stages"]["compile"].append(outcome("passed"))
    ref = refresh(tmp_path, plan, backend)
    result = e.execute(ref["path"], ref["sha256"], tmp_path / "ambiguous", run=True)
    assert "ambiguous" in result["tracks"]["strict"]["compile"]["reason"]
    (tmp_path / "trusted_backend.py").write_text('raise Exception("changed")')
    with pytest.raises(ValueError, match="hash"):
        e.execute(ref["path"], ref["sha256"], tmp_path / "changed")


def test_real_product_replay_backend_good_and_malformed(tmp_path):
    ref, plan, backend, _ = fixture(tmp_path, stage="raw_format")
    baseline = tmp_path / "kernel.metal"
    baseline.write_text("float x = 1;\n")
    generation = tmp_path / "generation.raw"
    diff = "diff --git a/kernel.metal b/kernel.metal\n--- a/kernel.metal\n+++ b/kernel.metal\n@@ -1 +1 @@\n-float x = 1;\n+float x = 2;\n"
    generation.write_bytes((r.OPEN + "git apply <<'PATCH'\n" + diff + "PATCH" + r.CLOSE).encode())
    inp = {
        "generation": pin(generation),
        "baseline": {"kernel.metal": pin(baseline)},
        "required_paths": ["kernel.metal"],
    }
    plan["steps"][0]["input"] = put(tmp_path / "input.json", inp)
    backend["argv"] = [
        "-m",
        "exl3_mlx.kernel_replay",
        "--input",
        "{input}",
        "--sha256",
        "{input_sha256}",
        "--output",
        "{output}",
    ]
    backend["code"] = [pin(Path(r.__file__))]
    backend["env"] = {
        "PYTHONPATH": str(Path(r.__file__).parents[1]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    backend["result"]["path"] = "{output}/replay.json"
    backend["result"]["bindings"] = [
        predicate(["format"], "kernel-eval-replay-v1"),
        predicate(["input_sha256"], "{input_sha256}"),
    ]
    backend["result"]["stages"] = {
        stage: [
            {
                "when": [predicate(["stages", stage, "status"], status)],
                "exit_codes": [0, 1],
                "status": status,
                "outcome_class": "model_output" if status == "failed" else "unspecified",
            }
            for status in ("passed", "failed", "not_run")
        ]
        for stage in ("raw_format", "replay")
    }
    ref = refresh(tmp_path, plan, backend)
    result = run(tmp_path, ref, run=True)
    assert result["tracks"]["strict"]["replay"]["status"] == "passed"
    assert (
        tmp_path / "run/00-raw_format/work/candidate/kernel.metal"
    ).read_text() == "float x = 2;\n"
    generation.write_bytes(generation.read_bytes().replace(b"@@ -1 +1 @@", b"@@ -1 +1,9 @@"))
    inp["generation"] = pin(generation)
    plan["steps"][0]["input"] = put(tmp_path / "input.json", inp)
    ref = refresh(tmp_path, plan, backend)
    result = e.execute(ref["path"], ref["sha256"], tmp_path / "malformed", run=True)
    assert result["tracks"]["strict"]["raw_format"]["status"] == "passed"
    assert result["tracks"]["strict"]["replay"]["status"] == "failed"
    assert result["tracks"]["strict"]["replay"]["outcome_class"] == "model_output"


def test_import_replay_convenience_produces_suite_result(tmp_path):
    _, plan, _, _ = fixture(tmp_path, stage="raw_format")
    source = tmp_path / "kernel.metal"
    source.write_text("float x = 1;\n")
    raw = tmp_path / "generation.raw"
    raw.write_bytes(
        (
            r.OPEN
            + "git apply <<'PATCH'\n--- a/kernel.metal\n+++ b/kernel.metal\n@@ -1 +1 @@\n-float x = 1;\n+float x = 2;\nPATCH"
            + r.CLOSE
        ).encode()
    )
    request = {
        "format": "kernel-eval-replay-request-v1",
        "initial_result": plan["initial_result"],
        "replay": {
            "generation": pin(raw),
            "baseline": {"kernel.metal": pin(source)},
            "required_paths": ["kernel.metal"],
        },
    }
    ref = put(tmp_path / "import.json", request)
    result = e.import_replay(ref["path"], ref["sha256"], tmp_path / "imported", run=True)
    assert result["format"] == "kernel-eval-result-v1"
    assert result["tracks"]["strict"]["replay"]["status"] == "passed"
    assert (tmp_path / "imported/execution/result.json").is_file()


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_full_files_import_exact_bytes_and_no_repair(tmp_path, changed, wrapped):
    _, plan, _, _ = fixture(tmp_path, stage="raw_format")
    baseline = {"kernel.metal": "// baseline\n", "dispatch.json": '{"threads_per_row":1}\n'}
    refs = {}
    for name, content in baseline.items():
        path = tmp_path / name
        path.write_text(content)
        refs[name] = pin(path)
    files = dict(baseline)
    if changed:
        files["kernel.metal"] = "// exact candidate bytes\n"
    content = json.dumps(
        {"format": "kernel-full-files-v1", "rationale": "A decision.", "files": files}
    ).encode()
    if wrapped:
        content = b"```json\n" + content + b"\n```"
    raw = tmp_path / "content.json"
    raw.write_bytes(content)
    request = {
        "format": "kernel-eval-replay-request-v1",
        "initial_result": plan["initial_result"],
        "replay": {
            "protocol": "kernel-full-files-v1",
            "generation": pin(raw),
            "baseline": refs,
            "required_paths": ["kernel.metal", "dispatch.json"],
        },
    }
    ref = put(tmp_path / "import.json", request)
    out = tmp_path / "imported"
    result = e.import_replay(ref["path"], ref["sha256"], out, run=True)
    stages = result["tracks"]["strict"]
    assert stages["raw_format"]["status"] == ("failed" if wrapped else "passed")
    work = out / "execution/00-raw_format/work"
    if wrapped:
        assert not (work / "candidate").exists()
    else:
        assert stages["replay"]["status"] == "passed"
        report = json.loads((work / "replay.json").read_text())
        assert report["unchanged_baseline"] is (not changed)
        for name, expected in files.items():
            assert (work / "candidate" / name).read_bytes() == expected.encode()
    backend = json.loads((out / "replay-backend.json").read_text())
    assert any(Path(x["path"]).name == "kernel_full_files.py" for x in backend["code"])


def test_documented_example_accepts_output_parent_alias(tmp_path):
    alias = tmp_path / "parent-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    example = Path(__file__).resolve().parents[1] / "examples/kernel_eval_replay_example.py"
    env = {"PYTHONPATH": str(Path(e.__file__).parents[1]), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, str(example), "--output", str(alias / "example")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    row = json.loads((tmp_path / "example/run/execution/result.json").read_text())
    assert row["tracks"]["strict"]["replay"]["status"] == "passed"


@pytest.mark.skipif(
    sys.platform != "darwin" or not shutil.which("xcrun"),
    reason="Requires local Apple CPU compiler",
)
def test_actual_metal_compiler_negative_no_gpu(tmp_path):
    ref, plan, backend, _ = fixture(tmp_path)
    script = tmp_path / "trusted_backend.py"
    script.write_text("""import hashlib,json,pathlib,subprocess,sys
source,out=map(pathlib.Path,sys.argv[1:]);out.mkdir()
metal=out/'candidate.metal';metal.write_text('#include <metal_stdlib>\\nkernel void broken( {\\n')
p=subprocess.run(['/usr/bin/xcrun','-sdk','macosx','metal','-std=metal3.1','-c',str(metal),'-o',str(out/'candidate.air')],capture_output=True)
(out/'compiler.stdout').write_bytes(p.stdout);(out/'compiler.stderr').write_bytes(p.stderr)
# This fixture deliberately tests a known syntax error from an available tool.
assert p.returncode != 0 and b'error:' in p.stderr, p.stderr
report={'format':'test-only-backend-v1','input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'status':'failed'}
(out/'result.json').write_text(json.dumps(report));sys.exit(1)
""")
    backend["code"] = [pin(script)]
    ref = refresh(tmp_path, plan, backend)
    result = run(tmp_path, ref, run=True)
    assert result["tracks"]["strict"]["compile"]["status"] == "failed"
    assert result["tracks"]["strict"]["compile"]["outcome_class"] == "model_output"
    assert not (tmp_path / "run/00-compile/work/candidate.air").exists()


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_strided_full_files_import_exact_bytes_and_no_repair(tmp_path, changed, wrapped):
    _, plan, _, _ = fixture(tmp_path, stage="raw_format")
    baseline = {"kernel.metal": "// baseline\n", "dispatch.json": '{"schema":"strided-copy-dispatch-v1","layout":"xy","threads":[16,16]}\n'}
    refs = {}
    for name, content in baseline.items():
        path = tmp_path / name
        path.write_text(content)
        refs[name] = pin(path)
    files = dict(baseline)
    if changed:
        files["kernel.metal"] = "// exact candidate bytes\n"
    content = json.dumps(
        {"format": "kernel-strided-copy-full-files-v1", "rationale": "A decision.", "files": files}
    ).encode()
    if wrapped:
        content = b"```json\n" + content + b"\n```"
    raw = tmp_path / "content.json"
    raw.write_bytes(content)
    request = {
        "format": "kernel-eval-replay-request-v1",
        "initial_result": plan["initial_result"],
        "replay": {
            "protocol": "kernel-strided-copy-full-files-v1",
            "generation": pin(raw),
            "baseline": refs,
            "required_paths": ["kernel.metal", "dispatch.json"],
        },
    }
    ref = put(tmp_path / "import.json", request)
    out = tmp_path / "imported"
    result = e.import_replay(ref["path"], ref["sha256"], out, run=True)
    stages = result["tracks"]["strict"]
    assert stages["raw_format"]["status"] == ("failed" if wrapped else "passed")
    work = out / "execution/00-raw_format/work"
    if wrapped:
        assert not (work / "candidate").exists()
    else:
        assert stages["replay"]["status"] == "passed"
        report = json.loads((work / "replay.json").read_text())
        assert report["unchanged_baseline"] is (not changed)
        for name, expected in files.items():
            assert (work / "candidate" / name).read_bytes() == expected.encode()
    backend = json.loads((out / "replay-backend.json").read_text())
    assert any(Path(x["path"]).name == "kernel_strided_response.py" for x in backend["code"])
