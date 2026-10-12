import hashlib
import json

import mlx.core as mx
import numpy as np
import pytest
from exl3_mlx.engram import hash_ids, hash_layout
from exl3_mlx.storage import ExpertBank, TensorArchive
from exl3_mlx.streaming import MatrixStore


def test_bank_ranges_hash_and_shape_validation(tmp_path):
    rng = np.random.default_rng(2)
    segments = []
    data = bytearray()
    for component in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("code", "rout", "rin"):
            array = (
                rng.integers(0, 65536, (8, 8, 48), dtype=np.uint16).view(np.int16)
                if suffix == "code"
                else np.ones(128, dtype=np.float16) * 0.2
            )
            segments.append(
                {
                    "component": component + "." + suffix,
                    "dtype": "I16" if suffix == "code" else "F16",
                    "shape": list(array.shape),
                    "offset": len(data),
                    "length": array.nbytes,
                }
            )
            data.extend(array.tobytes())
    logical = len(data)
    data.extend(bytes((-logical) % 4096))
    (tmp_path / "experts.bin").write_bytes(data)
    record = {
        "layer": 0,
        "expert": 0,
        "sidecar_offset": 0,
        "record_bytes": len(data),
        "logical_bytes": logical,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    manifest = {
        "format": "mtplx-expert-manifest-v2",
        "quantization": {
            "mode": "exl3",
            "codebook": "mul1",
            "codebook_multiplier": 0x83DCD12D,
            "tile": {"size": 16, "layout": "exl3-tensor-core", "bitstream": "msb-first-swap16"},
            "hadamard": {"size": 128, "order": "sylvester-natural", "scale": "1/sqrt(128)"},
            "scales": {"rin": "exl3.suh", "rout": "exl3.svh"},
            "source": {"repo": "test"},
        },
        "dims": {"hidden": 128, "inter": 128, "n_experts": 1, "n_layers": 1},
        "sidecar": {"file": "experts.bin", "size": len(data)},
        "records": [record],
        "layers": [
            {
                "layer": 0,
                "K": 3,
                "base_offset": 0,
                "record_bytes": len(data),
                "logical_bytes": logical,
                "segments": segments,
            }
        ],
    }
    file = tmp_path / "expert-manifest-v2.json"
    file.write_text(json.dumps(manifest))
    bank = ExpertBank(tmp_path)
    assert bank.verify_record(0, 0) == record["sha256"]
    store = MatrixStore(bank, cache_bytes=0, chunk_size=128)
    key = "layers.0.ffn.experts.0.w1"
    x = mx.random.normal((2, 128))
    np.testing.assert_array_equal(
        np.array(store.linear(key)(x)), np.array(bank.load_linear(key, chunk_size=128)(x))
    )
    manifest["layers"][0]["segments"][0]["offset"] = len(data)
    file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="bounds"):
        ExpertBank(tmp_path)


def test_tensor_archive_bf16_and_row_ranges(tmp_path):
    source = {"bf": mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.bfloat16), "f": mx.ones(3)}
    mx.save_safetensors(str(tmp_path / "test.safetensors"), source)
    archive = TensorArchive(tmp_path)
    assert archive.array("bf", (1, 2)).dtype == mx.bfloat16
    np.testing.assert_array_equal(
        np.array(archive.array("bf", (1, 2)).astype(mx.float32)), [[3, 4]]
    )
    with pytest.raises(ValueError, match="range"):
        archive.array("bf", (1, 3))


def test_hashing_matches_scalar_reference_and_sequence_boundaries():
    c = {
        "engram_layer_ids": [1, 14],
        "engram_max_ngram_size": 3,
        "engram_vocab_size": 5,
        "engram_n_heads": 2,
        "engram_compressed_vocab_size": 8,
        "engram_num_embeddings": [36, 84],
        "engram_pad_token_id": 2,
    }
    # Distinct primes: layer1 [5,7,11,13]; layer14 [17,19,23,29].
    c["engram_num_embeddings"] = [36, 88]
    layout = hash_layout(c)
    primes, offsets, mults = layout
    ids = np.array([[0, 3, 1], [5, 2, 4]])
    tokenmap = np.arange(8, dtype=np.uint32)
    actual = hash_ids(ids, tokenmap, c, layout)
    expected = np.zeros((2, 3, 2, 4), dtype=np.int64)
    for b in range(2):
        for t in range(3):
            for layer in range(2):
                acc = int(ids[b, t]) * int(mults[layer, 0])
                for n in range(1, 3):
                    acc ^= int(ids[b, t - n] if t >= n else 2) * int(mults[layer, n])
                    for h in range(2):
                        col = (n - 1) * 2 + h
                        expected[b, t, layer, col] = acc % int(primes[layer, col]) + int(
                            offsets[layer, col]
                        )
    np.testing.assert_array_equal(actual, expected)


def test_native_mxfp8_linear_forward_and_dx_match_dequantized_reference(tmp_path):
    from exl3_mlx.attention import quantize_activation
    from exl3_mlx.native import NativeLinear

    mx.random.seed(11)
    w, scale = mx.quantize(mx.random.normal((128, 128)), group_size=32, bits=8, mode="mxfp8")
    mx.save_safetensors(str(tmp_path / "native.safetensors"), {"p.weight": w, "p.scales": scale})
    layer = NativeLinear(TensorArchive(tmp_path), "p")
    dense = mx.dequantize(w, scale, group_size=32, bits=8, mode="mxfp8")
    x, dy = mx.random.normal((2, 128)), mx.random.normal((2, 128))
    (got,), (dx,) = mx.vjp(layer, [x], [dy])
    np.testing.assert_allclose(np.array(got), np.array(quantize_activation(x) @ dense.T), atol=2e-5)
    np.testing.assert_allclose(np.array(dx), np.array(dy @ dense), atol=2e-5)
    assert layer.trainable_parameters() == {}
