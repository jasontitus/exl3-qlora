"""Small, reproducible primitives and checkpoint inspection commands."""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from .checkpoint import Checkpoint
from .linear import EXL3Linear, EXL3QLoRALinear
from .remote import fetch_linear


def train_smoke(base, *, steps=60, output=None, seed=8):
    """Fit a synthetic rank-4 residual on a real or synthetic frozen base.

    This tests adapter optimization only; it is not language-model fine-tuning.
    Artifacts use this primitive's A/B orientation, not an asserted PEFT mapping.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if output is not None and Path(output).exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    mx.random.seed(seed)
    layer = EXL3QLoRALinear(base, rank=4, alpha=4)
    frozen = {k: np.array(v) for k, v in tree_flatten(base.parameters())}
    x = mx.random.normal((8, base.in_features)) / (base.in_features / 128) ** 0.5
    target = mx.stop_gradient(
        base(x)
        + (x @ mx.random.normal((base.in_features, 4)))
        @ mx.random.normal((4, base.out_features))
        * 0.01
    )
    mx.eval(target)

    def loss(m):
        return mx.mean((m(x) - target) ** 2)

    vg = nn.value_and_grad(layer, loss)
    opt = optim.Adam(learning_rate=0.01)
    history = []
    for _ in range(steps):
        value, grads = vg(layer)
        if set(dict(tree_flatten(grads))) != {"lora_a", "lora_b"}:
            raise AssertionError("Frozen base leaked into optimizer parameters")
        opt.update(layer, grads)
        mx.eval(value, layer.trainable_parameters(), opt.state)
        history.append(value.item())
    final_loss = loss(layer).item()
    if not np.isfinite(final_loss) or final_loss >= history[0]:
        raise AssertionError("Smoke training failed to decrease loss")
    for key, value in tree_flatten(base.parameters()):
        np.testing.assert_array_equal(np.array(value), frozen[key])
    report = {
        "task": "synthetic residual fit (not full-model training)",
        "steps": steps,
        "initial_loss": history[0],
        "final_loss": final_loss,
        "loss_history": history,
        "base_unchanged": True,
        "trainable_parameters": layer.lora_a.size + layer.lora_b.size,
        "device": mx.device_info()["device_name"],
        "mlx": mx.__version__,
    }
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True)
        mx.save_safetensors(
            str(output / "adapter.safetensors"), dict(tree_flatten(layer.trainable_parameters()))
        )
        config = {
            "format": "exl3-mlx-linear-v1",
            "rank": layer.rank,
            "alpha": layer.alpha,
            "in_features": base.in_features,
            "out_features": base.out_features,
            "base_bpw": base.bpw,
            "codebook": base.codebook,
        }
        (output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
        reloaded = EXL3QLoRALinear(base, rank=layer.rank, alpha=layer.alpha)
        reloaded.load_weights(str(output / "adapter.safetensors"), strict=False)
        np.testing.assert_array_equal(np.array(layer(x)), np.array(reloaded(x)))
        report["adapter_reload_exact"] = True
        (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    if len(sys.argv) > 1 and sys.argv[1] in {"prepare", "train", "evaluate", "generate", "doctor"}:
        from .workflow import main as workflow_main

        return workflow_main()
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog="Full workflow: prepare, train, evaluate, generate, doctor (COMMAND --help)",
    )
    sub = p.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Read local checkpoint headers; does not load a model")
    inspect.add_argument("path")
    check = sub.add_parser(
        "validate-linear", help="Compare one matrix's decode/forward/dX to NumPy"
    )
    check.add_argument("path")
    check.add_argument("--key", required=True)
    check.add_argument("--chunk-size", type=int, default=512)
    check.add_argument("--report")
    fetch = sub.add_parser("fetch-linear", help="Download one matrix with bounded HTTP ranges")
    fetch.add_argument("--repo", required=True)
    fetch.add_argument("--revision", required=True)
    fetch.add_argument("--shard", required=True)
    fetch.add_argument("--key", required=True)
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--max-mib", type=int, default=32)
    smoke = sub.add_parser("train-smoke", help="Fit a synthetic residual; not full-model SFT")
    smoke.add_argument("--checkpoint")
    smoke.add_argument("--key")
    smoke.add_argument("--steps", type=int, default=60)
    smoke.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "inspect":
        report = Checkpoint(args.path).summary()
    elif args.command == "fetch-linear":
        report = fetch_linear(
            args.repo,
            args.revision,
            args.shard,
            args.key,
            args.output,
            max_bytes=args.max_mib << 20,
        )
    elif args.command == "validate-linear":
        from .validation import validate_linear

        base = Checkpoint(args.path).load_linear(args.key, chunk_size=args.chunk_size)
        report = validate_linear(base)
        report.update(checkpoint=args.path, key=args.key)
        if args.report:
            out = Path(args.report)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2) + "\n")
    else:
        if bool(args.checkpoint) != bool(args.key):
            p.error("--checkpoint and --key must be provided together")
        if args.checkpoint:
            base = Checkpoint(args.checkpoint).load_linear(args.key)
        else:
            rng = np.random.default_rng(17)
            packed = rng.integers(0, 65536, (8, 8, 48), dtype=np.uint16).view(np.int16)
            base = EXL3Linear(mx.array(packed), mx.ones(128) * 0.05, mx.ones(128))
        report = train_smoke(base, steps=args.steps, output=args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
