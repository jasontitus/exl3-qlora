"""Measure attention/expert adapter influence on one fixed prompt.

Family ablation is a causal diagnostic for this prompt, not isolated training or
an additive attribution of quality. Every condition starts from the saved adapter.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.tuner.trainer import default_loss

from exl3_mlx.data import load_tokenizer, tokenize_row, validate_example
from exl3_mlx.deepseek import DeepSeekTextModel
from exl3_mlx.memory import MemoryBudget
from exl3_mlx.train import load_adapter, load_adapter_config
from exl3_mlx.training import batch_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args()
    root, adapter, output = (
        Path(x).expanduser().resolve() for x in (args.model, args.adapter, args.report)
    )
    if output.exists():
        p.error("Choose a new report path")
    config = load_adapter_config(adapter)
    adapter_hashes = {
        name: hashlib.sha256((adapter / name).read_bytes()).hexdigest()
        for name in ("adapter.safetensors", "adapter_config.json")
    }
    start = time.monotonic()
    report = {
        "scope": __doc__,
        "model": str(root),
        "adapter": str(adapter),
        "adapter_sha256": hashlib.sha256(
            (adapter / "adapter.safetensors").read_bytes()
        ).hexdigest(),
        "adapter_files_sha256": adapter_hashes,
        "prompt": "Code?",
        "answer": "violet",
        "conditions": {},
        "status": "running",
    }

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    with MemoryBudget(16, cache_mib=64, reserve_gib=32) as budget:
        t = load_tokenizer(root)
        row = validate_example(
            tokenize_row({"prompt": "Code?", "completion": "violet"}, t),
            seq_length=16,
            vocab_size=json.loads((root / "config.json").read_text())["text_config"]["vocab_size"],
        )
        report["tokenized"] = row
        model = DeepSeekTextModel(
            root,
            rank=config["rank"],
            alpha=config["alpha"],
            adapter_layers=config["adapter_layers"],
            expert_adapter_layers=config["expert_adapter_layers"],
            cache_mib=128,
            chunk_size=512,
        )
        model.eval()
        original = None
        for condition in ("all_on", "attention_only", "expert_only", "all_off"):
            if any(
                hashlib.sha256((adapter / name).read_bytes()).hexdigest() != digest
                for name, digest in adapter_hashes.items()
            ):
                raise ValueError("Adapter changed between ablation conditions")
            load_adapter(model, adapter)
            updates = []
            groups = {
                "attention": {"b_tensors": 0, "nonzero_b_tensors": 0, "b_squared_norm": 0.0},
                "expert": {"b_tensors": 0, "nonzero_b_tensors": 0, "b_squared_norm": 0.0},
            }
            for name, v in tree_flatten(model.trainable_parameters()):
                if not name.endswith("lora_b"):
                    continue
                group = "expert" if ".ffn.experts." in name else "attention"
                if condition == "all_on":
                    arr = np.array(v)
                    groups[group]["b_tensors"] += 1
                    groups[group]["nonzero_b_tensors"] += int(np.any(arr != 0))
                    groups[group]["b_squared_norm"] += float(np.sum(arr.astype(np.float64) ** 2))
                if (
                    condition == "all_off"
                    or (condition == "attention_only" and group == "expert")
                    or (condition == "expert_only" and group == "attention")
                ):
                    updates.append((name, mx.zeros_like(v)))
            if condition == "all_on":
                report["saved_adapter_families"] = groups
            model.load_weights(updates, strict=False)
            mx.eval(model.trainable_parameters())
            budget.check()
            tokens, lengths = batch_for(row)
            logits = model(tokens[:, :-1])
            loss, count = default_loss(lambda _, value=logits: value, tokens, lengths)
            values = np.array(logits).astype(np.float32)
            if not np.isfinite(values).all() or not np.isfinite(loss.item()):
                raise FloatingPointError("Nonfinite ablation logits or loss")
            positions = values[0, row["loss_start"] - 1 :]
            first = positions[0].astype(np.float64)
            prob = np.exp(first - first.max())
            prob /= prob.sum()
            if original is None:
                original = values.copy()
            report["conditions"][condition] = {
                "loss": loss.item(),
                "supervised_tokens": count.item(),
                "logits_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "target_top1_ids": positions.argmax(-1).tolist(),
                "first_target_probability": float(prob[row["input_ids"][row["loss_start"]]]),
                "max_abs_logit_delta_from_all_on": float(np.max(np.abs(values - original))),
                "elapsed_seconds": time.monotonic() - start,
            }
            print(condition, json.dumps(report["conditions"][condition]), flush=True)
            del logits, values, positions, loss, count, tokens
            budget.check()
            save()
        report["expert_family_affects_this_prompt"] = (
            report["conditions"]["attention_only"]["max_abs_logit_delta_from_all_on"] > 0
        )
        report["attention_family_affects_this_prompt"] = (
            report["conditions"]["expert_only"]["max_abs_logit_delta_from_all_on"] > 0
        )
        report["status"] = "complete"
        report["model_report"] = model.report()
        report["observed_memory_gib"] = budget.usage() / 2**30
        save()


if __name__ == "__main__":
    main()
