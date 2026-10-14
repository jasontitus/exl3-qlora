import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear as UpstreamLoRA
from test_block_training import Dense

from exl3_mlx.data import load_dataset, prepare, tokenize_row, validate_example
from exl3_mlx.native import LoRALinear
from exl3_mlx.training import (
    batch_for,
    evaluate,
    load_checkpoint,
    masked_loss,
    optimizer_step,
    sample_indices,
    save_checkpoint,
)
from exl3_mlx.workflow import parse_args


class Toy(nn.Module):
    def __init__(self, root):
        super().__init__()
        mx.random.seed(42)
        self.linear = LoRALinear(Dense(8, 12), rank=2, alpha=4.0)
        self.embedding = mx.random.normal((12, 8)) * 0.1
        self.freeze(keys=["embedding"], recurse=False)
        self._root = root
        self.adapter_config = dict(rank=2, alpha=4.0, adapter_layers=[0], expert_adapter_layers=[])

    def __call__(self, ids):
        return self.linear(self.embedding[ids])


@pytest.fixture
def base(tmp_path):
    root = tmp_path / "base"
    (root / "engram").mkdir(parents=True)
    for name in (
        "expert-manifest-v2.json",
        "engram/engram-manifest.json",
        "engram-token-map.u32.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        (root / name).write_text("{}")
    (root / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 12}}))
    return root


def examples():
    return [{"input_ids": [0, 2, 3, 4], "loss_start": 2}, {"input_ids": [0, 7, 8], "loss_start": 1}]


def test_actual_upstream_lora_forward_used():
    assert LoRALinear.__call__ is UpstreamLoRA.__call__
    layer = LoRALinear(Dense(8, 12), rank=2)
    with pytest.raises(NotImplementedError, match="requantization"):
        layer.fuse()


def test_masked_loss_and_evaluation_are_token_weighted(base):
    model = Toy(base)
    rows = examples()
    tokens, lengths = batch_for(rows[0])
    value = masked_loss(model, tokens, lengths)
    ce = nn.losses.cross_entropy(model(tokens[:, :-1]), tokens[:, 1:])
    assert np.isclose(value.item(), ce[:, 1:].mean().item())
    results = [evaluate(model, [row]) for row in rows]
    combined = evaluate(model, rows)
    assert model.training
    assert combined["supervised_tokens"] == 4
    assert combined["loss"] == sum(r["loss"] * r["supervised_tokens"] for r in results) / 4


def test_evaluation_progress_preserves_scores_and_reports_only_finished_rows(base):
    model = Toy(base)
    rows = examples()
    expected = evaluate(model, rows)
    messages = []
    assert evaluate(model, rows, log=messages.append) == expected
    assert len(messages) == 2 * len(rows)
    assert messages[0].startswith("Evaluation example 1/2 started")
    assert messages[-1] == "Evaluation example 2/2 completed; cumulative supervised tokens 4"
    checks = 0

    def fail_after_scoring():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("memory check failed")

    messages.clear()
    with pytest.raises(RuntimeError, match="memory check failed"):
        evaluate(model, rows, check=fail_after_scoring, log=messages.append)
    assert len(messages) == 1 and "started" in messages[0]
    assert model.training


def test_accumulation_matches_single_token_weighted_objective(base):
    rows = examples()
    rows[1]["loss_start"] = 2  # Unequal token counts, so averaging microbatch means is wrong.
    a, b = Toy(base), Toy(base)
    oa, ob = optim.Adam(1e-3), optim.Adam(1e-3)
    optimizer_step(a, oa, rows)

    def objective(m):
        return (masked_loss(m, *batch_for(rows[0])) * 2 + masked_loss(m, *batch_for(rows[1]))) / 3

    _, grads = nn.value_and_grad(b, objective)(b)
    grads, _ = optim.clip_grad_norm(grads, 1.0)
    ob.update(b, grads)
    for (ka, va), (kb, vb) in zip(
        tree_flatten(a.trainable_parameters()), tree_flatten(b.trainable_parameters())
    ):
        assert ka == kb
        np.testing.assert_allclose(np.array(va), np.array(vb), rtol=1e-6, atol=1e-8)


def test_atomic_checkpoint_exact_optimizer_and_rng_resume(base, tmp_path):
    a, b = Toy(base), Toy(base)
    oa, ob = optim.Adam(1e-3), optim.Adam(1e-3)
    rows, contract = examples(), {"data_sha256": "abc", "lr": 1e-3}
    optimizer_step(a, oa, rows)
    path = tmp_path / "step-000001"
    save_checkpoint(a, oa, path, {"step": 1, "cursor": 2, "contract": contract})
    expected_random = mx.random.uniform(shape=(5,))
    mx.eval(expected_random)
    optimizer_step(a, oa, rows)
    load_checkpoint(b, ob, path, contract)
    assert mx.array_equal(expected_random, mx.random.uniform(shape=(5,))).item()
    optimizer_step(b, ob, rows)
    for ta, tb in ((a.trainable_parameters(), b.trainable_parameters()), (oa.state, ob.state)):
        for (ka, va), (kb, vb) in zip(sorted(tree_flatten(ta)), sorted(tree_flatten(tb))):
            assert ka == kb
            assert mx.array_equal(va, vb).item()
    assert evaluate(a, rows) == evaluate(b, rows)
    assert not list(tmp_path.glob(".checkpoint-*"))
    with pytest.raises(ValueError, match="settings/data differ"):
        load_checkpoint(b, ob, path, {"different": True})
    raw = path / "random.safetensors"
    raw.write_bytes(raw.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_checkpoint(b, ob, path, contract)


def test_sampler_resume_crosses_epoch_boundary():
    whole = sample_indices(15, 0, 4, 8)
    resumed = sample_indices(7, 0, 4, 8) + sample_indices(8, 7, 4, 8)
    assert whole == resumed
    assert sorted(whole[:4]) == list(range(4))
    assert whole[:4] != whole[4:8]


def test_explicit_truncation_never_drops_all_targets():
    row = {"input_ids": [0, 2, 3, 4, 5], "loss_start": 4}
    with pytest.raises(ValueError, match="exceed"):
        validate_example(row, seq_length=2, vocab_size=12)
    with pytest.raises(ValueError, match="all supervised targets"):
        validate_example(row, seq_length=2, vocab_size=12, truncate=True)
    with pytest.raises(ValueError, match="Invalid text token"):
        validate_example({"input_ids": [0, True]}, seq_length=2, vocab_size=12)


class Tokenizer:
    bos_token_id, eos_token_id = 0, 1

    def encode(self, text):
        return [2, 3]

    def apply_chat_template(self, messages, *, add_generation_prompt=False, **kwargs):
        prefix = [0, 2, 3]
        return prefix if add_generation_prompt else prefix + [4, 1]


def test_upstream_data_formats_and_answer_masks():
    t = Tokenizer()
    assert tokenize_row({"text": "hello"}, t) == {"input_ids": [0, 2, 3, 1], "loss_start": 1}
    row = {"prompt": "Q", "completion": "A"}
    assert tokenize_row(row, t) == {"input_ids": [0, 2, 3, 4, 1], "loss_start": 3}
    assert tokenize_row(row, t, all_tokens=True)["loss_start"] == 1
    t.apply_chat_template = lambda m, add_generation_prompt=False, **kw: (
        [9] if add_generation_prompt else [0, 4]
    )
    with pytest.raises(ValueError, match="prefix is inconsistent"):
        tokenize_row(row, t)


def test_preparation_deduplicates_before_split_and_checks_provenance(base, tmp_path, monkeypatch):
    monkeypatch.setattr("exl3_mlx.data.load_tokenizer", lambda _: Tokenizer())
    source = tmp_path / "data.jsonl"
    source.write_text("\n".join(json.dumps({"input_ids": [0, 2, n]}) for n in [3, 4, 5, 6, 6]))
    out = tmp_path / "prepared"
    report = prepare(
        SimpleNamespace(
            model=str(base),
            output=str(out),
            data=str(source),
            seq_length=4,
            seed=8,
            valid_fraction=0.1,
            test_fraction=0.1,
            all_tokens=False,
            truncate=False,
        )
    )
    assert report["counts"] == dict(train=2, valid=1, test=1)
    assert report["duplicates_removed"] == 1
    assert len(load_dataset(out / "train.jsonl", base, seq_length=4)) == 2
    (out / "train.jsonl").write_text('{"input_ids":[0,2,8]}\n')
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_dataset(out / "train.jsonl", base, seq_length=4)


def test_config_cli_override_and_strict_types(tmp_path):
    file = tmp_path / "run.json"
    file.write_text(json.dumps(dict(model="M", data="D", output="O", steps=20, seq_length=4)))
    args = parse_args(["train", "--config", str(file), "--steps", "3"])
    assert (args.model, args.steps, args.seq_length) == ("M", 3, 4)
    file.write_text('{"steps": "three"}')
    with pytest.raises(SystemExit):
        parse_args(["train", "--config", str(file)])


def test_run_checkpoint_retention_resume_and_reload(base, tmp_path, monkeypatch):
    from exl3_mlx.workflow import train_run

    class TinyModel(Toy):
        def report(self):
            return {
                "trainable_parameters": sum(
                    v.size for _, v in tree_flatten(self.trainable_parameters())
                )
            }

    monkeypatch.setattr("exl3_mlx.workflow.make_model", lambda args, log: TinyModel(base))
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in examples()))
    valid = tmp_path / "valid.jsonl"
    valid.write_text('{"input_ids":[0,9,2,1],"loss_start":2}\n')
    budget = SimpleNamespace(check=lambda: None, limit=2**30, usage=lambda: 0)

    def run(output, *extra):
        args = parse_args(
            [
                "train",
                "--model",
                str(base),
                "--data",
                str(data),
                "--output",
                str(output),
                "--valid-data",
                str(valid),
                "--steps",
                "3",
                "--seq-length",
                "4",
                "--save-every",
                "1",
                "--keep-last",
                "2",
                "--verify-reload",
                *extra,
            ]
        )
        return train_run(args, budget, lambda _: None)

    full = run(tmp_path / "full")
    assert full["adapter_prediction_reload_exact"]
    assert len(list((tmp_path / "full").glob("step-*"))) == 2
    assert full["history"][-1]["validation"]["supervised_tokens"] == 2
    resumed = run(
        tmp_path / "resumed", "--resume", str(tmp_path / "full/step-000002"), "--steps", "1"
    )
    assert resumed["completed_step"] == 3
    for filename in ("adapter.safetensors", "optimizer.safetensors"):
        a = mx.load(str(Path(full["checkpoint"]) / filename))
        b = mx.load(str(Path(resumed["checkpoint"]) / filename))
        assert a.keys() == b.keys()
        assert all(mx.array_equal(a[k], b[k]).item() for k in a)
    with pytest.raises(FileExistsError):
        run(tmp_path / "full")


def test_resume_sparse_list_optimizer_tree(base, tmp_path):
    class Nested(Toy):
        def __init__(self, root):
            super().__init__(root)
            self.layers = [Dense(8, 8), LoRALinear(Dense(8, 12), rank=2)]
            del self.linear

        def __call__(self, ids):
            return self.layers[1](self.layers[0](self.embedding[ids]))

    mx.random.seed(14)
    a = Nested(base)
    mx.random.seed(14)
    b = Nested(base)
    oa, ob = optim.Adam(1e-3), optim.Adam(1e-3)
    optimizer_step(a, oa, examples())
    path = tmp_path / "nested"
    save_checkpoint(a, oa, path, {"step": 1, "cursor": 2, "contract": {}})
    optimizer_step(a, oa, examples())
    load_checkpoint(b, ob, path, {})
    optimizer_step(b, ob, examples())
    assert evaluate(a, examples()) == evaluate(b, examples())


def test_best_checkpoint_pinning_does_not_change_exact_optimizer_resume(
    base, tmp_path, monkeypatch
):
    from exl3_mlx import workflow

    monkeypatch.setattr(Toy, "report", lambda self: {"trainable_parameters": 40}, raising=False)
    monkeypatch.setattr(workflow, "make_model", lambda args, log: Toy(base))
    losses = iter(range(1, 9))
    monkeypatch.setattr(workflow, "evaluate", lambda *args, **kwargs: {"loss": next(losses)})
    data, valid = tmp_path / "train.jsonl", tmp_path / "valid.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in examples()))
    valid.write_text('{"input_ids":[0,9,2,1],"loss_start":2}\n')
    budget = SimpleNamespace(check=lambda: None, limit=2**30, usage=lambda: 0)

    def run(name, *flags):
        return workflow.train_run(
            parse_args(
                [
                    "train",
                    "--model",
                    str(base),
                    "--data",
                    str(data),
                    "--valid-data",
                    str(valid),
                    "--output",
                    str(tmp_path / name),
                    "--steps",
                    "3",
                    "--eval-every",
                    "1",
                    "--save-every",
                    "1",
                    "--keep-last",
                    "1",
                    *flags,
                ]
            ),
            budget,
            lambda message: None,
        )

    first = run("first")
    assert sorted(p.name for p in (tmp_path / "first").glob("step-*")) == [
        "step-000001",
        "step-000003",
    ]
    resumed = run("resumed", "--resume", first["checkpoint"], "--steps", "1")
    uninterrupted = run("whole", "--steps", "4")
    assert resumed["best_checkpoint"] == first["best_checkpoint"]
    assert resumed["best_validation_loss"] == 1
    assert resumed["contract"] == first["contract"]
    for filename in ("adapter.safetensors", "optimizer.safetensors", "random.safetensors"):
        a = mx.load(str(Path(resumed["checkpoint"]) / filename))
        b = mx.load(str(Path(uninterrupted["checkpoint"]) / filename))
        assert a.keys() == b.keys()
        assert all(mx.array_equal(a[key], b[key]).item() for key in a)


def test_interrupted_save_never_publishes_partial_checkpoint(base, tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("simulated disk full")

    model = Toy(base)
    opt = optim.Adam(1e-3)
    optimizer_step(model, opt, examples())
    monkeypatch.setattr("exl3_mlx.train.save_adapter", fail)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(model, opt, tmp_path / "saved", {})
    assert not (tmp_path / "saved").exists()
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_raw_chat_resume_identity_binds_template(base, tmp_path):
    from exl3_mlx.data import training_data_identity

    path = tmp_path / "chat.jsonl"
    path.write_text('{"prompt":"Q","completion":"A"}\n')
    old = training_data_identity(path, base)
    (base / "chat_template.jinja").write_text("changed template")
    new = training_data_identity(path, base)
    assert old["data_sha256"] == new["data_sha256"]
    assert old != new
    path.write_text('{"input_ids":[0,2,1]}\n')
    assert set(training_data_identity(path, base)) == {"data_sha256"}
