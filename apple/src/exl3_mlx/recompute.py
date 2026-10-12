"""Eager block recomputation compatible with data-dependent disk-backed experts.

MLX's general checkpoint wrapper retains a large lazy graph around the repeated
explicit evaluations in this path. Materialize the custom boundary instead, then
rebuild only that block during its first-order VJP. Parameters are explicit primals
so their adapter gradients are preserved; frozen arrays remain module constants.
"""

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from .materialize import materialize


def recompute(module, *args, progress=None):
    original_parameters = module.trainable_parameters()

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
            module.update(original_parameters)
        return tree_map(materialize, outputs)

    @forward.vjp
    def backward(primals, cotangents, outputs):
        if progress:
            progress("recomputing backward")
        # Route indices/candidate bitmaps are state, not differentiable inputs.
        # Flatten both sides so nested shared-attention state remains functional
        # across the boundary without asking MLX to differentiate integer leaves.
        leaves = [mx.stop_gradient(v) for _, v in tree_flatten(primals)]
        differentiable = [mx.issubdtype(v.dtype, mx.floating) for v in leaves]
        output_leaves = [v for _, v in tree_flatten(outputs)]
        output_differentiable = [mx.issubdtype(v.dtype, mx.floating) for v in output_leaves]

        def rebuild(values):
            iterator = iter(values)
            return tree_map(lambda _: next(iterator), primals)

        def flattened(*values):
            iterator = iter(values)
            full = [
                next(iterator) if flag else value for flag, value in zip(differentiable, leaves)
            ]
            result = inner(*rebuild(full))
            return tuple(
                value
                for (_, value), flag in zip(tree_flatten(result), output_differentiable)
                if flag
            )

        values = [v for v, flag in zip(leaves, differentiable) if flag]
        tangents = [
            value
            for (_, value), flag in zip(tree_flatten(cotangents), output_differentiable)
            if flag
        ]
        entry_parameters = module.trainable_parameters()
        try:
            _, gradients = mx.vjp(flattened, values, tangents)
            mx.eval(gradients)
        finally:
            module.update(entry_parameters)
        iterator = iter(gradients)
        result = rebuild(
            [
                materialize(next(iterator)) if flag else mx.zeros_like(value)
                for value, flag in zip(leaves, differentiable)
            ]
        )
        if progress:
            progress(f"backward complete; active MLX {mx.get_active_memory() / 2**30:.2f} GiB")
        return result

    return forward(original_parameters, *args)
