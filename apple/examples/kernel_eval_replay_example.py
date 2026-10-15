"""Runnable CPU replay-to-comparison example. No model, compiler or GPU.

Run from the repository root with PYTHONPATH=apple/src. Add --malformed to
demonstrate a retained hunk-count failure using the same pipeline.
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path

from exl3_mlx import kernel_execution, kernel_replay
from exl3_mlx.kernel_eval_suite import Reader, build_report, render_markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--malformed", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir()

    def ref(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def write(name, value):
        path = root / name
        path.write_text(json.dumps(value, indent=2) + "\n")
        return ref(path)

    old = "#include <metal_stdlib>\nconstant float scale = 1.0f;\n"
    new = old.replace("1.0f", "2.0f")
    baseline = root / "kernel.metal"
    baseline.write_text(old)
    patch = "".join(
        difflib.unified_diff(
            old.splitlines(True),
            new.splitlines(True),
            fromfile="a/kernel.metal",
            tofile="b/kernel.metal",
        )
    )
    if args.malformed:
        patch = patch.replace("+1,2", "+1,20")
    raw = root / "generation.raw"
    raw.write_text(
        kernel_replay.OPEN + "git apply <<'PATCH'\n" + patch + "PATCH" + kernel_replay.CLOSE
    )
    capture = write(
        "capture.json",
        {
            "format": "example-synthetic-capture-v1",
            "synthetic": True,
            "raw": ref(raw),
            "note": "Handwritten fixture, no model execution.",
        },
    )
    protocol = hashlib.sha256(Path(kernel_replay.__file__).read_bytes()).hexdigest()
    result = {
        "format": "kernel-eval-result-v1",
        "task_id": "example",
        "adapter_id": "synthetic",
        "task_sha256": ref(baseline)["sha256"],
        "adapter_sha256": hashlib.sha256(b"synthetic").hexdigest(),
        "prompt_sha256": hashlib.sha256(b"fixture prompt").hexdigest(),
        "protocol_sha256": protocol,
        "budget": {"max_output_tokens": 4096, "attempts": 1},
        "attempt": 1,
        "tracks": {"strict": {"generation": {"status": "passed", "evidence": [capture]}}},
    }
    initial = write("initial-result.json", result)
    request = write(
        "replay-request.json",
        {
            "format": "kernel-eval-replay-request-v1",
            "initial_result": initial,
            "replay": {
                "generation": ref(raw),
                "baseline": {"kernel.metal": ref(baseline)},
                "required_paths": ["kernel.metal"],
            },
        },
    )
    kernel_execution.import_replay(request["path"], request["sha256"], root / "run", run=True)
    suite = {
        "format": "kernel-eval-suite-v1",
        "id": "synthetic-replay-example",
        "tasks": [
            {
                "id": result["task_id"],
                "sha256": result["task_sha256"],
                "prompt_sha256": result["prompt_sha256"],
                "protocol_sha256": protocol,
            }
        ],
        "adapters": [{"id": result["adapter_id"], "sha256": result["adapter_sha256"]}],
        "budget": result["budget"],
        "results": [ref(root / "run/execution/result.json")],
    }
    write("suite.json", suite)
    report = build_report(suite, Reader(root))
    write("comparison.json", report)
    (root / "comparison.md").write_text(render_markdown(report))
    print(
        json.dumps(
            {
                "result": str(root / "run/execution/result.json"),
                "comparison": str(root / "comparison.json"),
                "scope": "Synthetic CPU replay only; source/compile/numerical/timing remain unrun.",
            }
        )
    )


if __name__ == "__main__":
    main()
