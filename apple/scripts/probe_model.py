"""Real 40-layer text forward/backward validation, with a process memory watchdog.

No optimizer update. Requires an existing local mtplx streaming repack; downloads
nothing. Run from apple/: uv run python scripts/probe_model.py --model PATH
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from tokenizers import Tokenizer

from exl3_mlx.deepseek import DeepSeekTextModel
from exl3_mlx.memory import MemoryBudget


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--memory-gib", type=float, default=16.0)
    p.add_argument(
        "--report",
        default=str(Path(__file__).resolve().parents[1] / "reports/full-model-probe.json"),
    )
    args = p.parse_args()
    root = Path(args.model).expanduser()
    start = time.monotonic()

    def progress(message):
        print(f"{time.monotonic() - start:.1f}s {message}", flush=True)

    with MemoryBudget(args.memory_gib):
        tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        ids = [0] + tok.encode("The capital of France is Paris.", add_special_tokens=False).ids
        model = DeepSeekTextModel(root, rank=2, alpha=4.0, progress=progress)
        x = mx.array([ids[:4]])
        model.eval()
        y = model(x)
        mx.eval(y)
        if not mx.all(mx.isfinite(y)).item():
            raise AssertionError("Nonfinite logits")
        model.train()

        def loss(m):
            return nn.losses.cross_entropy(m(x), mx.array([ids[1:5]]), reduction="mean")

        value, grads = nn.value_and_grad(model, loss)(model)
        mx.eval(value, grads)
        flat = tree_flatten(grads)
        if not all(mx.all(mx.isfinite(v)).item() for _, v in flat):
            raise AssertionError("Nonfinite gradients")
        report = model.report()
        report.update(
            loss=value.item(),
            elapsed_seconds=time.monotonic() - start,
            gradient_tensors=len(flat),
            all_grads_finite=True,
            optimizer_update_tested=False,
            input_ids=ids[:4],
            target_ids=ids[1:5],
            memory_budget_gib=args.memory_gib,
        )
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
