"""Conservative implementation limit, independent of sparse index_topk.

Whole examples retain causal/Engram history. Resource guards may reject a run
below this length; the model's larger native window is not a memory guarantee.
"""

MAX_CONTEXT_TOKENS = 32768

# Changes to arithmetic that invalidate exact optimizer continuation.
ENGINE_SEMANTICS = "deepseek-v41-sparse-prefill-v4-grouped-vjp-nested-attention"
