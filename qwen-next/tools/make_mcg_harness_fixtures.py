"""Bounded synthetic inner-MCG fixtures; no GPU imports or checkpoint reads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tools/trellis_reference.py"
KERNEL = ROOT / "kernels/mcg_decode.metal"
UPSTREAM = ROOT / "sources/src/exl3/expert_exl3.zig"
REFERENCE_SHA = "36827164c88dd6c25dfde67077abe3ad5fb45067f85940a64035fe79dbf29493"
UPSTREAM_SHA = "33d37858ec6647c60fb975e5c011b1354aa4453e9258a0e79a1f781d22a5e0a5"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def word_hash(words):
    return digest(struct.pack(f"<{len(words)}H", *words))


def reference():
    if digest(REFERENCE.read_bytes()) != REFERENCE_SHA:
        raise ValueError("Reviewed CPU reference identity changed")
    if digest(UPSTREAM.read_bytes()) != UPSTREAM_SHA:
        raise ValueError("Pinned upstream source identity changed")
    spec = importlib.util.spec_from_file_location("mcg_fixture_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_case(ref, case_id, k, n, start, width, halfwords, window, seed):
    # Explicit reproducible uint32 LCG, not a language RNG/version dependency.
    state = seed
    words = []
    for _ in range((k // 16) * (n // 16) * halfwords):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        words.append((state >> 8) & 65535)
    if seed == 0:
        words = [0] * len(words)
    full = ref.reconstruct_inner(words, k, n, halfwords, window=window)
    expected = [
        b for row in range(k) for b in full[row * n + start : row * n + start + width]
    ]
    return {
        "case_id": case_id,
        "k": k,
        "n": n,
        "start": start,
        "width": width,
        "halfwords": halfwords,
        "window": window,
        "packed": words,
        "expected_fp16_bits": expected,
        "packed_sha256": word_hash(words),
        "expected_sha256": word_hash(expected),
    }


def build_suite():
    ref = reference()
    cases = [make_case(ref, "known-zero", 16, 16, 0, 16, 16, 14, 0)]
    cases.extend(
        make_case(ref, f"rate-{rate}-w14", 16, 16, 0, 16, rate, 14, 8231 + rate)
        for rate in range(16, 129, 2)
    )
    cases.extend(
        make_case(
            ref,
            f"crop-rate-{rate}-w{window}",
            32,
            48,
            16,
            32,
            rate,
            window,
            rate * 1921 + window,
        )
        for rate in (16, 40, 48, 128)
        for window in range(8, 17)
    )
    return {
        "schema": "qwen-mcg-metal-synthetic-v1",
        "scope": "synthetic-inner-mcg-only",
        "comparison": "exact-fp16-bits",
        "kernel_sha256": digest(KERNEL.read_bytes()),
        "cpu_reference_sha256": REFERENCE_SHA,
        "upstream_sha256": UPSTREAM_SHA,
        "generator_sha256": digest(Path(__file__).read_bytes()),
        "limitations": [
            "No real Qwen payload or window14 converter known-answer qualification",
            "No Hadamard/scales/public matrix, serving arithmetic, matmul, VJP or training",
            "Synthetic rate/window coverage does not establish checkpoint format support",
        ],
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = build_suite()
    raw = (json.dumps(suite, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with args.output.open("xb") as handle:
        handle.write(raw)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": digest(raw),
                "cases": len(suite["cases"]),
                "gpu_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
