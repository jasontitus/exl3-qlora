#!/usr/bin/env python3
"""Adversarial CPU-only validation of the compiled expression probe gate."""

import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original_raw = args.job.read_bytes()
    job = json.loads(original_raw)
    binary = job["harness_binary"]["path"]
    results = []
    with tempfile.TemporaryDirectory(prefix="expression-probe-cpu-") as name:
        root = Path(name)

        def bound(path):
            return {"path": str(path), "sha256": sha(path.read_bytes())}

        def save(path, value):
            path.write_text(json.dumps(value, indent=2) + "\n")

        def run(label, current, expected=1, contains=None):
            path, receipt = root / (label + ".json"), root / (label + "-receipt.json")
            save(path, current)
            proc = subprocess.run(
                [binary, str(path), str(receipt), "--validate-only"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            report = json.loads(receipt.read_text()) if receipt.exists() else None
            assert proc.returncode == expected, (label, proc.stdout, proc.stderr)
            if report is not None:
                assert report["gpu_executed"] is False
                assert report["passed"] is False  # CPU validation is not numeric success.
            if contains:
                assert contains in proc.stderr, (label, proc.stderr)
            results.append(
                {
                    "test": label,
                    "exit_code": proc.returncode,
                    "gpu_executed": False,
                    "passed": True,
                    "error": proc.stderr.strip(),
                }
            )
            return receipt

        valid_receipt = run("valid_bindings", copy.deepcopy(job), 0)
        current = copy.deepcopy(job)
        plan = json.loads(Path(current["plan"]["path"]).read_text())
        plan["generation"]["max_tokens"] = 7
        path = root / "changed-plan.json"
        save(path, plan)
        current["plan"] = bound(path)
        run("coherent_changed_plan", current, contains="Wrong predeclared plan")
        current = copy.deepcopy(job)
        current["probes"].reverse()
        run("swapped_probes", current, contains="Probe plan changed")
        current = copy.deepcopy(job)
        current["probes"][0]["cases"][0]["n"] = 1
        run("changed_case", current, contains="Probe plan changed")
        current = copy.deepcopy(job)
        current["probes"][0]["candidate"]["library"]["sha256"] = "0" * 64
        run("library_digest", current, contains="Changed file")
        current = copy.deepcopy(job)
        current["probes"][0]["expression"] = "sumf"
        run("expression_not_response", current, contains="Candidate differs from supplied response")
        current = copy.deepcopy(job)
        source = Path(current["probes"][0]["candidate"]["source"]["path"]).read_text()
        path = root / "altered-source.metal"
        path.write_text(source.replace("total = simd_sum(v);", "total = 0;"))
        assert path.read_text() != source
        current["probes"][0]["candidate"]["source"] = bound(path)
        run("coherent_wrong_source", current, contains="Generated expression/source mismatch")
        # Even repairing response fields cannot permit an unsafe index. Use a
        # declared negative-control expression slot to reach range validation.
        current = copy.deepcopy(job)
        current["origin"] = "untrusted_test_only"
        responses = json.loads(Path(current["responses"]["path"]).read_text())
        responses["origin"] = "untrusted_test_only"
        responses["responses"]["broadcast_mul_index"]["text"] = "tpig"
        path = root / "changed-responses.json"
        save(path, responses)
        current["responses"] = bound(path)
        current["probes"][1]["raw_text"] = "tpig"
        current["probes"][1]["expression"] = "tpig"
        run("unrecognized_control_origin", current, contains="Control differs from plan")
        # Existing receipts are never overwritten, even for valid jobs.
        saved = valid_receipt.read_bytes()
        proc = subprocess.run(
            [binary, str(args.job.resolve()), str(valid_receipt), "--validate-only"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert proc.returncode == 2 and valid_receipt.read_bytes() == saved
        results.append(
            {"test": "receipt_nonoverwrite", "exit_code": 2, "gpu_executed": False, "passed": True}
        )
    report = {
        "format": "expression-probe-cpu-bindings-v1",
        "job_sha256": sha(original_raw),
        "binary": job["harness_binary"],
        "test_count": len(results),
        "tests": results,
        "gpu_executed": False,
    }
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"{len(results)} CPU binding/negative checks passed; no GPU")


if __name__ == "__main__":
    main()
