"""Controlled DeepSeek EXL3 behavior-change experiment using the installed trainer.

Not a general quality benchmark: fit two codeword prompts and test one unseen
wording. All measurements use the same checkpoint, template and greedy decoding.
The baseline output is capped at four tokens. A passing trained answer must end
with EOS, and must survive adapter reload. Requires the local streaming repack;
downloads nothing. Use a new output directory for every experiment.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.tuner.trainer import default_loss

from exl3_mlx.data import (
    load_tokenizer,
    tokenize_row,
    training_data_identity,
    validate_example,
)
from exl3_mlx.deepseek import DeepSeekTextModel
from exl3_mlx.memory import MemoryBudget
from exl3_mlx.train import load_adapter
from exl3_mlx.training import batch_for, optimizer_step, sample_indices, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--memory-gib", type=float, default=16)
    parser.add_argument(
        "--resume-adapter", help="Warm start adapter weights with a fresh Adam optimizer"
    )
    parser.add_argument(
        "--baseline-report",
        help="Reuse this script's completed base measurements; ablation rechecks the logits",
    )
    args = parser.parse_args()
    if args.max_steps < 4 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Use at least four steps and a positive finite learning rate")
    root, out = (
        Path(args.model).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
    )
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {
        "status": "running",
        "scope": "Controlled codeword fitting; not a general quality claim",
        "model": str(root),
        "target": "violet",
        "train_prompts": ["Code?", "Code:"],
        "held_out_prompt": "Code please?",
        "history": [],
        "settings": vars(args),
    }

    def record():
        temporary = out / "report.tmp.json"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(out / "report.json")

    def log(message):
        print(f"[{time.monotonic() - start:.1f}s] {message}", flush=True)

    with MemoryBudget(args.memory_gib, cache_mib=64, reserve_gib=32) as budget:
        budget.abort_path = out / "memory-abort.json"
        t = load_tokenizer(root)
        vocab = json.loads((root / "config.json").read_text())["text_config"]["vocab_size"]
        prompts = report["train_prompts"] + [report["held_out_prompt"]]
        rows = [
            validate_example(
                tokenize_row({"prompt": p, "completion": "violet"}, t),
                seq_length=16,
                vocab_size=vocab,
            )
            for p in prompts
        ]
        (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:2]))
        (out / "valid.jsonl").write_text(json.dumps(rows[2]) + "\n")
        (out / "source.jsonl").write_text(
            "".join(json.dumps({"prompt": p, "completion": "violet"}) + "\n" for p in prompts)
        )
        report["tokenization"] = [{"prompt": p, **r} for p, r in zip(prompts, rows)]
        contract = {
            "learning_rate": args.learning_rate,
            "grad_accumulation": 1,
            "clip_grad": 1.0,
            "seed": 8,
            "seq_length": 16,
            "truncate": False,
            "chunk_size": 512,
            "no_checkpoint": False,
            **training_data_identity(out / "train.jsonl", root),
            "mlx": mx.__version__,
            "mlx_lm": importlib.metadata.version("mlx-lm"),
        }
        report["contract"] = contract
        report["phase"] = "loading"
        record()
        mx.random.seed(8)
        log(
            "Loading the 3-bpw checkpoint; rank 4 attention adapters and EXL3 expert adapters in layer 0"
        )
        model = DeepSeekTextModel(
            root,
            rank=4,
            alpha=8,
            cache_mib=128,
            chunk_size=512,
            expert_adapter_layers=[0],
        )
        frozen_ids = {
            k: id(v)
            for k, v in tree_flatten(model.parameters())
            if not k.endswith(("lora_a", "lora_b"))
        }
        report["initial_model"] = model.report()
        log(f"Loaded {report['initial_model']['trainable_parameters']:,} adapter parameters")

        def measure(row):
            model.eval()
            budget.check()
            tokens, lengths = batch_for(row)
            logits = model(tokens[:, :-1])
            loss, n = default_loss(lambda _, value=logits: value, tokens, lengths)
            positions = logits[0, row["loss_start"] - 1 :]
            predictions = mx.argmax(positions, axis=-1)
            expected = row["input_ids"][row["loss_start"] :]
            first_probs = mx.softmax(positions[0])
            prob = first_probs[expected[0]]
            mx.eval(loss, n, predictions, prob, logits)
            result = {
                "loss": loss.item(),
                "supervised_tokens": n.item(),
                "first_target_probability": prob.item(),
                "predicted_target_ids": predictions.tolist(),
                "expected_target_ids": expected,
                "all_target_top1_correct": predictions.tolist() == expected,
                "logits_sha256": hashlib.sha256(np.array(logits).tobytes()).hexdigest(),
            }
            del logits, tokens, positions, first_probs, predictions
            budget.check()
            return result

        def greedy(row, max_new_tokens=4):
            model.eval()
            prefix = row["input_ids"][: row["loss_start"]]
            ids, generated = list(prefix), []
            for _ in range(max_new_tokens):
                budget.check()
                logits = model(mx.array([ids], dtype=mx.int32))
                if not mx.all(mx.isfinite(logits[0, -1])).item():
                    raise FloatingPointError("Nonfinite generation")
                nxt = mx.argmax(logits[0, -1]).item()
                del logits
                generated.append(nxt)
                ids.append(nxt)
                if nxt == t.eos_token_id:
                    break
            return {
                "text": t.decode(generated, skip_special_tokens=True),
                "generated_ids": generated,
                "prompt_ids": prefix,
                "max_new_tokens": max_new_tokens,
                "ended_with_eos": bool(generated and generated[-1] == t.eos_token_id),
                "decoding": "greedy; no sampling",
                "template": "checkpoint local chat template",
            }

        report["phase"] = "baseline"
        record()
        report["before"] = {}
        if args.baseline_report:
            baseline_path = Path(args.baseline_report).expanduser().resolve()
            baseline_bytes = baseline_path.read_bytes()
            baseline = json.loads(baseline_bytes)
            if baseline["phase"] != "complete":
                raise ValueError("Baseline experiment must have completed")
            if baseline["model"] != str(root) or baseline["tokenization"] != report["tokenization"]:
                raise ValueError("Baseline model or tokenized examples differ")
            if baseline["contract"]["mlx"] != mx.__version__:
                raise ValueError("Baseline MLX version differs")
            if not baseline["adapter_off_matches_base_logits"]:
                raise ValueError("Baseline experiment did not pass its adapter-off check")
            report["before"] = baseline["before"]
            report["baseline_source"] = {
                "path": str(baseline_path),
                "sha256": hashlib.sha256(baseline_bytes).hexdigest(),
            }
            log(f"Reusing baseline from {baseline_path}; final ablation will verify it again")
            record()
        else:
            for i in (0, 2):
                key = "trained_prompt" if i == 0 else "held_out_prompt"
                report["before"][key] = {
                    "metrics": measure(rows[i]),
                    "generation": greedy(rows[i]),
                }
                log(f"Baseline {prompts[i]!r}: {report['before'][key]}")
                record()
        if report["before"]["trained_prompt"]["generation"]["text"].strip() == "violet":
            raise AssertionError(
                "Base already returns target; this fixture cannot demonstrate learning"
            )
        if args.resume_adapter:
            checkpoint = Path(args.resume_adapter).expanduser().resolve()
            load_adapter(model, checkpoint)
            report["warm_start"] = {
                "path": str(checkpoint),
                "adapter_sha256": hashlib.sha256(
                    (checkpoint / "adapter.safetensors").read_bytes()
                ).hexdigest(),
                "optimizer": "fresh Adam; not an exact optimizer continuation",
            }
            log(f"Warm started {checkpoint}; new Adam at learning rate {args.learning_rate}")
        optimizer = optim.Adam(learning_rate=args.learning_rate)
        report["phase"] = "training"
        record()
        final_checkpoint = None
        for step in range(1, args.max_steps + 1):
            # Use the public trainer sampler; the held-out wording never enters optimization.
            row_index = sample_indices(1, step - 1, 2, 8)[0]
            metrics = optimizer_step(model, optimizer, [rows[row_index]], check=budget.check)
            metrics.update(
                step=step,
                prompt=prompts[row_index],
                elapsed_seconds=time.monotonic() - start,
                peak_mlx_gib=mx.get_peak_memory() / 2**30,
            )
            report["history"].append(metrics)
            log("Update " + json.dumps(metrics))
            if step == 1 or step % 2 == 0 or step == args.max_steps:
                final_checkpoint = out / f"step-{step:06d}"
                save_checkpoint(
                    model,
                    optimizer,
                    final_checkpoint,
                    {
                        "step": step,
                        "cursor": step,
                        "contract": contract,
                        "metrics": metrics,
                    },
                )
                report["checkpoint"] = str(final_checkpoint)
            record()
            if step >= 4 and step % 2 == 0:
                probe = measure(rows[0])
                metrics["probe_after_update"] = probe
                log("Learning check " + json.dumps(probe))
                record()
                if probe["all_target_top1_correct"]:
                    candidate = greedy(rows[0])
                    metrics["generation_after_update"] = candidate
                    record()
                    if candidate["text"].strip() == "violet" and candidate["ended_with_eos"]:
                        break
        report["phase"] = "after-training"
        report["after"] = {}
        for i in (0, 2):
            key = "trained_prompt" if i == 0 else "held_out_prompt"
            report["after"][key] = {
                "metrics": measure(rows[i]),
                "generation": greedy(rows[i]),
            }
            log(f"After training {prompts[i]!r}: {report['after'][key]}")
            record()
        report["frozen_parameter_objects_unchanged"] = frozen_ids == {
            k: id(v)
            for k, v in tree_flatten(model.parameters())
            if not k.endswith(("lora_a", "lora_b"))
        }
        report["phase"] = "adapter-ablation-and-reload"
        record()
        for name, value in tree_flatten(model.trainable_parameters()):
            model.load_weights([(name, mx.zeros_like(value))], strict=False)
        mx.eval(model.trainable_parameters())
        off = measure(rows[0])
        report["adapter_off_matches_base_logits"] = (
            off["logits_sha256"] == report["before"]["trained_prompt"]["metrics"]["logits_sha256"]
        )
        load_adapter(model, final_checkpoint)
        reloaded = measure(rows[0])
        generation = greedy(rows[0])
        report["reloaded"] = {"metrics": reloaded, "generation": generation}
        report["adapter_reload_exact"] = report["reloaded"] == report["after"]["trained_prompt"]
        report["learned_target_on_trained_prompt"] = (
            generation["text"].strip() == "violet" and generation["ended_with_eos"]
        )
        report["generalized_to_held_out_wording"] = (
            report["after"]["held_out_prompt"]["generation"]["text"].strip() == "violet"
            and report["after"]["held_out_prompt"]["generation"]["ended_with_eos"]
        )
        report["output_changed"] = (
            generation["generated_ids"]
            != report["before"]["trained_prompt"]["generation"]["generated_ids"]
        )
        report["answer_loss_decreased"] = (
            report["after"]["trained_prompt"]["metrics"]["loss"]
            < report["before"]["trained_prompt"]["metrics"]["loss"]
        )
        report["elapsed_seconds"] = time.monotonic() - start
        report["final_model"] = model.report()
        report["observed_memory_gib"] = budget.usage() / 2**30
        report["status"] = (
            "passed"
            if all(
                report[k]
                for k in (
                    "learned_target_on_trained_prompt",
                    "output_changed",
                    "adapter_reload_exact",
                    "adapter_off_matches_base_logits",
                    "frozen_parameter_objects_unchanged",
                    "answer_loss_decreased",
                )
            )
            else "needs-investigation"
        )
        report["phase"] = "complete"
        record()
        log(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "status",
                        "output_changed",
                        "learned_target_on_trained_prompt",
                        "generalized_to_held_out_wording",
                        "adapter_reload_exact",
                        "adapter_off_matches_base_logits",
                        "observed_memory_gib",
                        "elapsed_seconds",
                    )
                },
                indent=2,
            )
        )
        if report["status"] != "passed":
            raise SystemExit(f"Behavior checks did not pass; inspect {out / 'report.json'}")


if __name__ == "__main__":
    main()
