"""Eager block recomputation compatible with data-dependent disk-backed experts.

MLX's general checkpoint wrapper retains a large lazy graph around the repeated
explicit evaluations in this path. Materialize the custom boundary instead, then
rebuild only that block during its first-order VJP. Parameters are explicit primals
so their adapter gradients are preserved; frozen arrays remain module constants.
"""

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from .materialize import materialize


def recompute(module, *args):
    def inner(parameters, *inputs):
        module.update(parameters)
        return module(*inputs)

    @mx.custom_function
    def forward(parameters, *inputs):
        detached = tree_map(mx.stop_gradient, (parameters, *inputs))
        try:
            outputs = inner(*detached)
            mx.eval(outputs)
        finally:
            module.update(parameters)
        return tree_map(materialize, outputs)

    @forward.vjp
    def backward(primals, cotangents, outputs):
        # mx.vjp expects one cotangent per positional output (tuple here).
        def rebuild(values):
            iterator = iter(values)
            return tree_map(lambda _: next(iterator), primals)

        def flattened(*values):
            return inner(*rebuild(values))

        values = [mx.stop_gradient(v) for _, v in tree_flatten(primals)]
        _, gradients = mx.vjp(flattened, values, list(cotangents))
        mx.eval(gradients)
        module.update(primals[0])
        return rebuild([materialize(g) for g in gradients])

    return forward(module.trainable_parameters(), *args)
