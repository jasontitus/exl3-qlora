"""Sequential real-model train/resume/eval/generate checks under normal memory guards.

Requires several model loads and a new output directory. Downloads nothing.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


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
    data = Path(__file__).resolve().parents[1] / "examples/smoke-tokens.jsonl"
    prefix = [sys.executable, "-m", "exl3_mlx.workflow"]
    common = ["--model", str(root), "--memory-gib", str(args.memory_gib)]
    report = {
        "scope": "4-token operational validation; not quality or native full-model parity",
        "model": str(root),
        "phases": {},
        "fixture_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }
    if args.continue_run:
        old = json.loads((output / "verification.json").read_text())
        if old["model"] != str(root):
            p.error("Cannot continue against a different model")
        fixture_hash = old.get("fixture_sha256")
        if fixture_hash is None and old["phases"].get("train", {}).get("exit_code") == 0:
            fixture_hash = json.loads((output / "train.stdout.json").read_text())["contract"][
                "data_sha256"
            ]
        if fixture_hash != report["fixture_sha256"]:
            p.error("Cannot continue with a different or unverified fixture")
        report["phases"] = old["phases"]
    else:
        output.mkdir(parents=True, exist_ok=False)

    def run(name, command):
        previous = report["phases"].get(name, {})
        if previous.get("exit_code") == 0:
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
        report["phases"][name] = {"exit_code": result.returncode, "attempt": attempt}
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
