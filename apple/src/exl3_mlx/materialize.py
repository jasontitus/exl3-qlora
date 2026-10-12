"""Opaque first-order autograd boundaries for the eager streaming prototype.

Under an outer MLX gradient trace, eval/stop_gradient alone does not release the
internal graph of a Python custom_function. Round-trip its result through a host
buffer to remove that graph. This is used for activation/gradient boundaries and
for frozen decoded FP16 tiles when populating the optional tile cache. Tile
population therefore adds a host copy; decoding and all numerical arithmetic
remain on Metal. Replacing these synchronization/copy boundaries with a C++
primitive is a performance follow-up, not a prerequisite for correct first-order
adapter training. BF16 bytes are preserved exactly.
"""

import mlx.core as mx
import numpy as np


def materialize(value):
    mx.eval(value)
    if value.dtype == mx.bfloat16:
        return mx.array(np.array(value.view(mx.uint16))).view(mx.bfloat16)
    return mx.array(np.array(value))
