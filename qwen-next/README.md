# Qwen architecture references

This directory contains isolated CPU architecture/storage references and historical qualification evidence. It does not provide a supported Qwen model loader, trainer or serving engine. Production model dispatch is unchanged.

## QSA selection and gated attention

Two NumPy components provide reusable mathematical checks for Stage A development:

- `tools/qwen_qsa_selector_reference.py`: scores caller-preprocessed index queries and pooled keys, selects completed four-token blocks, and appends the causal incomplete tail. It returns compact sequence-local routes with `-1` padding. Captured routes can be replayed only against the same input identity; checksums provide consistency, not authenticity.
- `tools/qwen_qsa_attention_reference.py`: consumes normalized/rotated Q/K, V, raw per-channel gate logits and selected routes. It computes grouped-query softmax attention, applies the elementwise sigmoid gate and returns immutable flattened context for the `o_proj` boundary. It does not implement that projection or the remaining block.

The configured model has 24 query heads, 2 KV heads and dimension 256. Each contiguous group of 12 query heads shares one KV head. Main attention scores use `dot(q,k)/sqrt(D)`; the indexer's ReLU score must not be substituted here.

The selector uses FP32 mathematical scores. Attention uses FP64 on supplied FP16/32/64 values. Neither establishes native bitwise parity: the captured native attention casts softmax probabilities to V's dtype before multiplication, and its split/reduction order differs. Compact selector slots and tie behavior also differ from some native references. The input boundary assumes the caller already applied the correct projection, normalization, partial RoPE and pooled-key preparation. No model weights or device libraries are imported.

Run the CPU reference tests from the repository root, using an environment with NumPy already available:

```sh
python -m unittest discover -s qwen-next/tests -p 'test_qwen_qsa_*_reference.py' -v
python -m unittest discover -s qwen-next/tests -p 'test_qsa_source_contract.py' -v
```

The 22 methods cover independent scalar attention, the 24:2 head mapping, nonuniform channel gates, reduced-budget sparse composition, the actual 2,051/2,052-token selector boundary, causal/unique routes, padding, noncontiguous inputs and malformed/overflow refusal. They do not establish native numerical tolerances or execute upstream model code.

A small direct use, with eight already-preprocessed keys and four query heads:

```python
import sys
import numpy as np
sys.path.insert(0, "qwen-next/tools")
from qwen_qsa_attention_reference import attend

q = np.zeros((1, 1, 4, 2), dtype=np.float32)
k = np.zeros((1, 8, 2, 2), dtype=np.float32)
v = np.full_like(k, 2.0)
gate = np.zeros_like(q)
routes = np.full((1, 1, 32), -1, dtype=np.int32)
routes[0, 0, :8] = np.arange(8)
result = attend(q, k, v, gate, routes,
                np.array([[7]], dtype=np.int64),
                np.array([8], dtype=np.int64))
assert np.array_equal(result.context(), np.ones((1, 1, 8)))
```

`attend` requires plain `numpy.ndarray` inputs in all seven roles and rejects subclasses, including masked arrays with empty masks, before arithmetic. Noncontiguous plain arrays are supported. `attend` rejects duplicate/future/out-of-sequence routes and empty attention sets for valid queries. Explicit right-padded queries produce zero context; that is a CPU extension, not native padding qualification. It gathers one query/head's selected K/V at a time. Fixture caps include 4,096 history tokens, 64 query rows, 2 batches, 16 MiB aggregate input and 8 MiB returned context. Individual array counters are not peak RSS measurements. No 16K capacity claim follows from these tests.

The portable source identities and semantic seams are recorded in [reference-contracts/qsa-cpu-v2.json](reference-contracts/qsa-cpu-v2.json). The next gates remain upstream preprocessing, a complete A0 context/residual/post-state boundary and downstream differentiable suffix, real-checkpoint/native parity, and separately measured capacity. Sparse backward, adapter training and useful model quality are not supplied by these references.
