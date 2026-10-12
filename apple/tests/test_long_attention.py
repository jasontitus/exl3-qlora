"""Independent oracles for long-context selection and bounded attention VJPs."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from exl3_mlx.attention import (
    FrozenIndexer,
    candidate_mask,
    empty_attention_state,
    pow2_exact,
    quantize_activation,
    select_index_scores,
)
from exl3_mlx.recompute import recompute
from exl3_mlx.sparse_attention import sparse_latent_attention


def attention_oracle(query, window, compressed, sink, selected, width):
    result = np.empty(query.shape, dtype=np.float64)
    for batch in range(query.shape[0]):
        for position in range(query.shape[1]):
            rows = [window[batch, i] for i in range(max(0, position - width + 1), position + 1)]
            rows += [compressed[batch, i] for i in selected[batch, position] if i >= 0]
            keys = np.asarray(rows, dtype=np.float64)
            for head in range(query.shape[2]):
                scores = np.append(
                    keys @ query[batch, position, head] / math.sqrt(query.shape[-1]), sink[head]
                )
                probability = np.exp(scores - scores.max())
                probability /= probability.sum()
                result[batch, position, head] = probability[:-1] @ keys
    return result


@pytest.mark.parametrize("chunk", [1, 3, 11])
def test_sparse_attention_forward_and_all_vjp_inputs_against_float64_oracle(chunk):
    rng = np.random.default_rng(71)
    values = [rng.normal(size=s) for s in ((2, 7, 3, 8), (2, 7, 8), (2, 7, 8), (3,))]
    selected = np.empty((2, 7, 3), np.int32)
    for b in range(2):
        for s in range(7):
            selected[b, s] = [-1, 0, 0] if s < 2 else [s, 0, s - 1]
    dy = rng.normal(size=values[0].shape)
    arrays = [mx.array(v, dtype=mx.float32) for v in values]

    def fn(q, w, c, sink):
        return sparse_latent_attention(
            q, w, c, mx.array(selected), sink, window_size=3, query_chunk=chunk
        )

    (out,), grads = mx.vjp(fn, arrays, [mx.array(dy, dtype=mx.float32)])
    expected = attention_oracle(*values, selected, 3)
    np.testing.assert_allclose(np.array(out), expected, rtol=2e-6, atol=5e-7)
    # Each role gets a numerical derivative, including repeated reads into the
    # same compressed position, both batches, and the denominator-only sink.
    for which, coordinates in enumerate(
        (
            [(0, 0, 0, 2), (1, 5, 2, 4)],
            [(0, 0, 3), (1, 3, 5)],
            [(0, 0, 1), (1, 5, 6)],
            [(0,), (2,)],
        )
    ):
        for coord in coordinates:
            plus, minus = [v.copy() for v in values], [v.copy() for v in values]
            plus[which][coord] += 1e-5
            minus[which][coord] -= 1e-5
            expected_grad = (
                np.sum(
                    (attention_oracle(*plus, selected, 3) - attention_oracle(*minus, selected, 3))
                    * dy
                )
                / 2e-5
            )
            assert np.array(grads[which])[coord] == pytest.approx(expected_grad, rel=4e-5, abs=3e-6)


def selection_oracle(scores, visible, topk, *, block_size, keep_blocks=0, candidates=None):
    scores = scores.copy()
    result = np.full((*scores.shape[:-1], min(topk, scores.shape[-1])), -1, np.int32)
    candidate = np.zeros(scores.shape, bool) if keep_blocks else candidates
    for b in range(scores.shape[0]):
        for s in range(scores.shape[1]):
            row = scores[b, s]
            row[visible[s] :] = -np.inf
            if keep_blocks:
                blocks = [max(row[i : i + block_size]) for i in range(0, len(row), block_size)]
                if visible[s]:
                    blocks[(visible[s] - 1) // block_size] = np.inf
                chosen = sorted(range(len(blocks)), key=lambda i: (-blocks[i], i))[:keep_blocks]
                for i in chosen:
                    if blocks[i] > -np.inf:
                        candidate[b, s, i * block_size : (i + 1) * block_size] = True
            elif candidates is not None:
                row[~candidates[b, s]] = -np.inf
            ranked = sorted(range(len(row)), key=lambda i: (-row[i], i))[:topk]
            chosen = sorted(i for i in ranked if row[i] > -np.inf)
            result[b, s, : len(chosen)] = chosen
    return result, candidate


def test_index_selection_candidates_causality_partial_blocks_and_ties():
    rng = np.random.default_rng(27)
    scores = rng.normal(size=(2, 9, 13)).astype(np.float32)
    scores[:, 6, :4] = 2.0  # Explicit deterministic low-position ties.
    visible = np.array([0, 1, 2, 3, 4, 7, 8, 12, 13], np.int32)
    want, candidates = selection_oracle(scores, visible, 6, block_size=4, keep_blocks=1)
    got, packed = select_index_scores(
        mx.array(scores), mx.array(visible), topk=6, candidate_blocks=1, block_size=4
    )
    np.testing.assert_array_equal(np.array(got), want)
    np.testing.assert_array_equal(np.array(candidate_mask(packed, 13, 4)), candidates)
    # Consumers use their own scoring inside the source's candidate blocks.
    other = -scores
    want, _ = selection_oracle(other, visible, 6, block_size=4, candidates=candidates)
    got, unused = select_index_scores(
        mx.array(other), mx.array(visible), topk=6, block_size=4, candidates=packed
    )
    assert unused is None
    np.testing.assert_array_equal(np.array(got), want)
    assert np.all(np.array(got)[0, 1, 1:] == -1)


def test_real_topk_boundary_513_and_candidate_boundary_16385():
    # Small query count lets the actual thresholds be checked without dense
    # S-by-S attention or a full-model GPU workload.
    visible = mx.array([511, 512, 513])
    scores = mx.broadcast_to(mx.arange(513, dtype=mx.float32), (1, 3, 513))
    got, _ = select_index_scores(scores, visible, topk=512)
    assert np.array(got)[0, 0, -1] == -1
    np.testing.assert_array_equal(np.array(got)[0, 1], np.arange(512))
    np.testing.assert_array_equal(np.array(got)[0, 2], np.arange(1, 513))
    scores = -mx.arange(16385, dtype=mx.float32)[None, None]
    _, packed = select_index_scores(
        scores, mx.array([16385]), topk=512, candidate_blocks=2048, block_size=8
    )
    keep = np.array(candidate_mask(packed, 16385, 8))[0, 0]
    assert keep[-1]  # Always pin the newest partial block, despite its low score.
    assert not keep[16376:16384].any()


def fp4_e8m0_oracle(values):
    blocks = values.astype(np.float64).reshape(-1, 32)
    scales = 2.0 ** np.ceil(
        np.log2(np.maximum(np.max(np.abs(blocks), axis=1, keepdims=True), 6 * 2.0**-126) / 6)
    )
    values = blocks / scales
    levels = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float64)
    distances = np.abs(np.abs(values)[..., None] - levels)
    # Prefer even code at a midpoint, then the unique nearest level.
    codes = np.argmin(distances + np.arange(8)[None, None] % 2 * 1e-12, axis=-1)
    return (np.sign(values) * levels[codes] * scales).reshape(-1, 32)


@pytest.mark.parametrize("dtype", [mx.float32, mx.int32])
def test_exact_power_of_two_preserves_negative_exponents(dtype):
    exponents = np.array([-126, -23, -9, -7, -1, 0, 1, 8], dtype=np.int32)
    np.testing.assert_array_equal(
        np.array(pow2_exact(mx.array(exponents, dtype=dtype))),
        np.ldexp(np.ones(exponents.shape, np.float32), exponents),
    )


def test_fp8_forward_against_independent_e4m3_codebook():
    # Ascending positive E4M3 encodings include subnormals and finite exponent
    # 15 codes; their low bit supplies the nearest-even midpoint tie rule.
    levels = np.array(
        [m * 2.0**-9 for m in range(8)]
        + [2.0**e * (1 + m / 8) for e in range(-6, 9) for m in range(8)],
        np.float64,
    )
    levels = levels[levels <= 448]
    values = np.random.default_rng(5).normal(size=(5, 32)).astype(np.float32)
    values *= np.array([0, 0.01, 1, 20, 10000], np.float32)[:, None]
    scale = 2.0 ** np.ceil(
        np.log2(np.maximum(np.max(np.abs(values), axis=-1, keepdims=True), 1e-4) / 448)
    )
    normalized = values.astype(np.float64) / scale.astype(np.float64)
    distances = np.abs(np.abs(normalized)[..., None] - levels)
    codes = np.argmin(distances + np.arange(len(levels)) % 2 * 1e-12, axis=-1)
    expected = np.sign(normalized) * levels[codes] * scale
    got = quantize_activation(mx.array(values), "fp8")
    np.testing.assert_array_equal(np.array(got), expected.astype(np.float32))


def test_indexer_fp4_uses_e8m0_32_element_scales_and_identity_surrogate():
    values = np.random.default_rng(6).normal(size=(3, 32)).astype(np.float32)
    values[0] = 0
    values[1, :16] = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, -6]
    values[1, 16:] = 0
    got = quantize_activation(mx.array(values), "fp4_e8m0")
    np.testing.assert_array_equal(np.array(got), fp4_e8m0_oracle(values))
    dy = mx.array(np.random.default_rng(9).normal(size=values.shape).astype(np.float32))
    (_,), (dx,) = mx.vjp(lambda x: quantize_activation(x, "fp4_e8m0"), [mx.array(values)], [dy])
    np.testing.assert_array_equal(np.array(dx), np.array(dy))


@pytest.mark.parametrize("length,topk,blocks,block_size", [(11, 3, 2, 2), (513, 512, 2048, 8)])
def test_frozen_indexer_matches_installed_reference_selection(length, topk, blocks, block_size):
    from mlx_lm.models import deepseek_v41 as upstream
    from mlx_lm.models.rope_utils import YarnRoPE
    from test_block_training import Dense

    args = upstream.ModelArgs(
        hidden_size=32,
        q_lora_rank=32,
        head_dim=32,
        qk_rope_head_dim=8,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=topk,
        compress_ratios=(1,),
        kv_source_layer_ids=(0,),
        index_source_layer_ids=(0,),
        candidate_source_layer_id=0,
        candidate_topk_blocks=blocks,
        candidate_block_size=block_size,
    )
    config = vars(args) | {
        "rope_scaling": {
            "factor": 2,
            "original_max_position_embeddings": 16,
            "beta_fast": 32,
            "beta_slow": 1,
        }
    }
    mx.random.seed(88)
    projections = {"wq_b": Dense(32, 64), "weights_proj": Dense(32, 2), "wk": Dense(32, 32)}
    projections["wq_b"].weight = mx.array(
        np.concatenate([np.eye(32), np.eye(32) * 0.7]).astype(np.float32)
    )
    projections["wk"].weight = mx.eye(32)
    projections["weights_proj"].weight = mx.abs(projections["weights_proj"].weight)
    indexer = FrozenIndexer(0, config, projections, mx.ones(32), query_chunk=64)
    ref = upstream.Indexer(args, 0)
    for our_name, ref_name in (
        ("wq_b", "q_b_proj"),
        ("weights_proj", "weights_proj"),
        ("wk", "k_proj"),
    ):
        getattr(ref, ref_name).weight = projections[our_name].weight
    ref.k_norm.weight = mx.ones(32)
    rotary = YarnRoPE(
        dims=8,
        traditional=True,
        base=args.compress_rope_theta,
        scaling_factor=2,
        original_max_position_embeddings=16,
        beta_fast=32,
        beta_slow=1,
        mscale=0,
        mscale_all_dim=0,
    )
    # Positive non-RoPE components dominate the small rotating tail, avoiding
    # all-zero ReLU score ties where upstream argpartition has no tie contract.
    rng = np.random.default_rng(13)
    values = [rng.uniform(1, 3, (2, length, 32)).astype(np.float32) for _ in range(3)]
    for value in values:
        value[..., -8:] *= 0.05
    x, qr, latent = map(mx.array, values)
    keys = indexer.keys(latent)
    ref_keys = upstream._apply_rope(ref.k_norm(ref.k_proj(latent)), rotary, rope_dim=8)
    ref_keys = upstream._fake_quant_fp4(ref_keys, block_size=32)
    np.testing.assert_allclose(np.array(keys), np.array(ref_keys), rtol=1e-6, atol=1e-6)
    got, packed = indexer(x, qr, keys, mx.zeros((2, length, 0), dtype=mx.uint8))
    want, candidates = ref(x, qr, index_k=ref_keys, candidates=None, rope=rotary, start_pos=0)
    want = np.array(want)
    want = np.sort(np.where(want >= 0, want, length), axis=-1)
    want[want == length] = -1
    np.testing.assert_array_equal(np.array(got), want)
    np.testing.assert_array_equal(
        np.array(candidate_mask(packed, length, block_size)), np.array(candidates)
    )
    assert tree_flatten(indexer.trainable_parameters()) == []


def test_recompute_with_shared_index_state_preserves_adapter_and_kv_gradients(tmp_path):
    from test_block_training import Dense, make_block
    from test_streaming_moe import make_checkpoint

    from exl3_mlx.hyperconnections import expand_streams
    from exl3_mlx.streaming import MatrixStore

    mx.random.seed(13)
    store = MatrixStore(
        make_checkpoint(tmp_path / "expert.safetensors"), cache_bytes=0, chunk_size=128
    )
    blocks = nn.Sequential(make_block(store, 0), make_block(store, 1))
    source = blocks.layers[0].attn
    for block in blocks.layers:
        block.attn.config.update(
            index_topk=1,
            index_n_heads=2,
            index_head_dim=32,
            candidate_source_layer_id=0,
            candidate_topk_blocks=1,
            candidate_block_size=2,
        )
        for name in ("wq_a", "wo_b"):
            block.attn.linears[name].lora_b = mx.random.normal((2, 128)) * 0.01
    source.indexer = FrozenIndexer(
        0,
        source.config,
        {
            "wq_b": Dense(128, 64),
            "weights_proj": Dense(128, 2),
            "wk": Dense(128, 32),
        },
        mx.ones(32),
        query_chunk=2,
    )
    h, pre = expand_streams(mx.random.normal((1, 7, 128)))

    def loss(model, checkpoint):
        state = empty_attention_state(1, 7, 128, 32, h.dtype)
        streams, mixing = h, pre
        for block in model.layers:
            args = (streams, mixing, state, mx.zeros((0,)))
            streams, mixing, state = recompute(block, *args) if checkpoint else block(*args)
        return mx.mean(streams * streams) + mx.mean(state[0] * state[0])

    plain, pg = nn.value_and_grad(blocks, lambda m: loss(m, False))(blocks)
    checked, cg = nn.value_and_grad(blocks, lambda m: loss(m, True))(blocks)
    np.testing.assert_allclose(plain.item(), checked.item(), rtol=1e-6)
    for (pk, pv), (ck, cv) in zip(tree_flatten(pg), tree_flatten(cg), strict=True):
        assert pk == ck and pk.endswith(("lora_a", "lora_b"))
        assert mx.linalg.norm(pv).item() > 0
        np.testing.assert_allclose(np.array(pv), np.array(cv), rtol=5e-5, atol=3e-7)
