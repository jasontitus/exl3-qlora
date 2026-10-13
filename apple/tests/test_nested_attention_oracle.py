"""Reuse the pre-existing independent NumPy oracle and its original tolerances."""

import test_composed_gradient as oracle


def test_nested_attention_matches_existing_anchored_numpy_oracle(monkeypatch):
    original = oracle.make_model

    def nested(data):
        model = original(data)
        for block in model.blocks:
            block.checkpoint_attention = True
        model.train()
        return model

    monkeypatch.setattr(oracle, "make_model", nested)
    # The existing oracle independently replays every quantizer's local STE,
    # actual route choices, both shared-KV blocks and all trainable arrays in
    # NumPy float64. Its finite-difference tolerances are not changed here.
    oracle.test_composed_recompute_gradient_matches_independent_anchored_replay(monkeypatch)
