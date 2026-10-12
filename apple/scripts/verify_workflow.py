"""Sequential real-model train/resume/eval/generate checks under normal memory guards.

Requires several model loads and a new output directory. Downloads nothing.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

MODEL_METADATA = (
    "config.json",
    "expert-manifest-v2.json",
    "engram/engram-manifest.json",
    "engram-token-map.u32.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
CHECKPOINT_FILES = {
    "adapter.safetensors",
    "adapter_config.json",
    "optimizer.safetensors",
    "random.safetensors",
}
PHASES = {"train", "resume", "base-eval", "adapter-eval", "generate"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_identity(project):
    # Resolving a top-level package spec does not import it or initialize Metal.
    spec = importlib.util.find_spec("exl3_mlx")
    expected = (project / "src/exl3_mlx").resolve()
    if spec is None or spec.origin is None or Path(spec.origin).resolve().parent != expected:
        raise ValueError("Run with the environment installed from this source checkout")
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": sorted(
            [d.metadata["Name"], d.version]
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        ),
    }


def provenance(project, model, fixture, *, runtime=None):
    sources = sorted((project / "src/exl3_mlx").glob("*.py"))
    if not sources:
        raise ValueError("No runtime source files found")
    sources += [
        project / name for name in ("scripts/verify_workflow.py", "pyproject.toml", "uv.lock")
    ]
    return {
        "format": "exl3-workflow-verification-provenance-v1",
        "source": {str(p.relative_to(project)): sha256(p) for p in sources},
        "runtime": runtime if runtime is not None else runtime_identity(project),
        "model_metadata": {name: sha256(model / name) for name in MODEL_METADATA},
        "model_identity_scope": "Metadata/tokenizer/template hashes; NOT base weight payload hashes",
        "fixture_sha256": sha256(fixture),
    }


def phase_evidence(output, name):
    if name not in PHASES:
        raise ValueError(f"Unknown verification phase: {name}")
    files = {
        f"{name}.{suffix}": sha256(output / f"{name}.{suffix}")
        for suffix in ("stdout.json", "stderr.log")
    }
    if name in {"train", "resume"}:
        manifests = sorted((output / name).glob("step-*/trainer_state.json"))
        if not manifests:
            raise ValueError(f"No checkpoint manifests for {name}")
        for manifest in manifests:
            state = json.loads(manifest.read_text())
            if (
                state.get("format") != "exl3-mlx-training-state-v1"
                or set(state.get("files", {})) != CHECKPOINT_FILES
            ):
                raise ValueError(f"Invalid checkpoint manifest: {manifest}")
            files[str(manifest.relative_to(output))] = sha256(manifest)
            for filename, expected in state["files"].items():
                path = manifest.parent / filename
                actual = sha256(path)
                if actual != expected:
                    raise ValueError(f"Checkpoint checksum mismatch: {path}")
                files[str(path.relative_to(output))] = actual
    return files


def validate_continuation(old, current, output):
    if old.get("model") != current["model"] or old.get("provenance") != current["provenance"]:
        raise ValueError("Cannot continue: verification provenance is missing or differs")
    if old.get("fixture_sha256") != current["fixture_sha256"]:
        raise ValueError("Cannot continue: fixture identity differs")
    for name, phase in old["phases"].items():
        if phase.get("exit_code") == 0 and phase.get("evidence") != phase_evidence(output, name):
            raise ValueError(f"Cannot reuse {name}: saved phase evidence is missing or differs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--memory-gib", type=float, default=12)
    p.add_argument(
        "--continue-run",
        action="store_true",
        help="Reuse completed phases in this output directory",
    )
    p.add_argument(
        "--wait-for-memory",
        type=float,
        default=0,
        help="Seconds to wait for normal startup headroom per pending phase",
    )
    args = p.parse_args()
    root = Path(args.model).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if args.wait_for_memory < 0:
        p.error("wait-for-memory must be nonnegative")
    project = Path(__file__).resolve().parents[1]
    data = project / "examples/smoke-tokens.jsonl"
    prefix = [sys.executable, "-m", "exl3_mlx.workflow"]
    common = ["--model", str(root), "--memory-gib", str(args.memory_gib)]
    report = {
        "scope": "4-token operational validation; not quality or native full-model parity",
        "model": str(root),
        "phases": {},
        "fixture_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "provenance": provenance(project, root, data),
    }
    if args.continue_run:
        old = json.loads((output / "verification.json").read_text())
        try:
            validate_continuation(old, report, output)
        except (ValueError, OSError) as error:
            p.error(str(error))
        report["phases"] = old["phases"]
    else:
        output.mkdir(parents=True, exist_ok=False)

    def run(name, command):
        previous = report["phases"].get(name, {})
        if previous.get("exit_code") == 0:
            if previous.get("command") != prefix + command:
                p.error(f"Cannot reuse {name}: invocation changed")
            print(f"Reusing completed {name}", flush=True)
            return json.loads((output / f"{name}.stdout.json").read_text())
        if name in {"train", "resume"} and (output / name).exists():
            raise SystemExit(
                "An incomplete training phase exists. Use the trainer's --resume on its last complete step, or choose a new verification output directory."
            )
        if args.wait_for_memory:
            from exl3_mlx.memory import MemoryBudget

            deadline = time.monotonic() + args.wait_for_memory
            announced = False
            while True:
                try:
                    MemoryBudget(args.memory_gib).check_startup_headroom()
                    break
                except MemoryError:
                    if time.monotonic() >= deadline:
                        raise
                    if not announced:
                        print(f"Waiting for normal memory headroom before {name}", flush=True)
                        announced = True
                    time.sleep(5)
        attempt = previous.get("attempt", 1) + 1 if previous else 1
        if previous:
            for suffix in ("stdout.json", "stderr.log"):
                old_log = output / f"{name}.{suffix}"
                if old_log.exists():
                    old_log.rename(output / f"{name}.attempt-{attempt - 1}.{suffix}")
        print(f"Running {name}; logs in {output}", flush=True)
        with (
            (output / f"{name}.stdout.json").open("w") as stdout,
            (output / f"{name}.stderr.log").open("w") as stderr,
        ):
            result = subprocess.run(prefix + command, stdout=stdout, stderr=stderr)
        phase = {"exit_code": result.returncode, "attempt": attempt, "command": prefix + command}
        if not result.returncode:
            phase["evidence"] = phase_evidence(output, name)
        report["phases"][name] = phase
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        if result.returncode:
            raise SystemExit(
                f"{name} stopped ({result.returncode}); inspect {name}.stderr.log. Completed checkpoints remain recoverable."
            )
        return json.loads((output / f"{name}.stdout.json").read_text())

    training = ["--data", str(data), "--seq-length", "4", "--save-every", "1"]
    full = run(
        "train",
        [
            "train",
            *common,
            *training,
            "--output",
            str(output / "train"),
            "--steps",
            "2",
            "--rank",
            "2",
            "--alpha",
            "4",
            "--expert-adapter-layers",
            "0",
            "--verify-reload",
        ],
    )
    resumed = run(
        "resume",
        [
            "train",
            *common,
            *training,
            "--output",
            str(output / "resume"),
            "--steps",
            "1",
            "--resume",
            str(output / "train/step-000001"),
        ],
    )
    import mlx.core as mx

    from exl3_mlx.memory import MemoryBudget

    with MemoryBudget(2, cache_mib=64, reserve_gib=16):
        for name in ("adapter.safetensors", "optimizer.safetensors"):
            a = mx.load(str(Path(full["checkpoint"]) / name))
            b = mx.load(str(Path(resumed["checkpoint"]) / name))
            if set(a) != set(b) or not all(mx.array_equal(a[k], b[k]).item() for k in a):
                raise AssertionError(f"Full-model resume mismatch: {name}")
        del a, b
        mx.clear_cache()
    base = run("base-eval", ["evaluate", *common, "--data", str(data), "--seq-length", "4"])
    adapted = run(
        "adapter-eval",
        [
            "evaluate",
            *common,
            "--data",
            str(data),
            "--seq-length",
            "4",
            "--adapter",
            full["checkpoint"],
        ],
    )
    generated = run(
        "generate",
        [
            "generate",
            *common,
            "--adapter",
            full["checkpoint"],
            "--raw-prompt",
            "--prompt",
            "The capital of France",
            "--max-tokens",
            "1",
        ],
    )
    report.update(
        completed=True,
        full_model_optimizer_resume_exact=True,
        adapter_prediction_reload_exact=full["adapter_prediction_reload_exact"],
        base_loss=base["loss"],
        adapter_loss=adapted["loss"],
        generated_ids=generated["generated_ids"],
        peak_mlx_gib=max(full["peak_mlx_gib"], resumed["peak_mlx_gib"]),
    )
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
