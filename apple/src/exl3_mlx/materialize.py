"""Opaque first-order autograd boundaries for the eager streaming prototype.

Under an outer MLX gradient trace, eval/stop_gradient alone does not release the
internal graph of a Python custom_function. Round-trip ONLY its small activation
result through a host buffer to remove that graph. Packed/dense weights never
leave Metal for computation. This adds synchronization/copy overhead; replacing
it with a C++ primitive is a performance follow-up, not a prerequisite for correct
first-order adapter training. BF16 bytes are preserved exactly.
"""

import mlx.core as mx
import numpy as np


def materialize(value):
    mx.eval(value)
    if value.dtype == mx.bfloat16:
        return mx.array(np.array(value.view(mx.uint16))).view(mx.bfloat16)
    return mx.array(np.array(value))
