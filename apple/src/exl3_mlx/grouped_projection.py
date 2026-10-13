"""Frozen independent group projections with a group-local first-order VJP.

The reference slice/concatenate expression asks autodiff to scatter each input
slice gradient into a complete [..., groups, features] array. These groups do
not overlap, so stack their local VJPs directly instead. Frozen projection math,
activation quantizers and input/output dtypes are unchanged. This eager boundary
supports first-order differentiation, like the surrounding streaming engine.
"""

import mlx.core as mx
from mlx.utils import tree_flatten

from .materialize import materialize


def grouped_projection_reference(value, projections):
    return mx.concatenate(
        [projection(value[..., index, :]) for index, projection in enumerate(projections)],
        axis=-1,
    )


def grouped_projection(value, projections):
    """Project [..., G, D] groups and concatenate their possibly unequal widths.

    Only frozen modules use the bounded VJP. If any projection has trainable
    parameters, retain the ordinary expression so those gradients stay visible.
    Callers must not mutate frozen weights/configuration during differentiation.
    """
    projections = tuple(projections)
    if value.ndim < 2 or not projections or value.shape[-2] != len(projections):
        raise ValueError("Expected one projection for each nonempty input group")

    def frozen():
        return all(
            not tree_flatten(projection.trainable_parameters()) for projection in projections
        )

    if not frozen():
        return grouped_projection_reference(value, projections)
    widths = tuple(projection.out_features for projection in projections)
    if any(type(width) is not int or width < 1 for width in widths):
        raise ValueError("Frozen group projections require positive output widths")

    @mx.custom_function
    def forward(inp):
        inp = mx.stop_gradient(inp)
        pieces = [
            materialize(projection(inp[..., index, :]))
            for index, projection in enumerate(projections)
        ]
        return materialize(mx.concatenate(pieces, axis=-1))

    @forward.vjp
    def backward(primals, cotangent, output):
        if not frozen():
            raise ValueError("Frozen group projection trainability changed during differentiation")
        inp, cotangent = mx.stop_gradient(primals), mx.stop_gradient(cotangent)
        pieces, offset = [], 0
        for index, (projection, width) in enumerate(zip(projections, widths, strict=True)):
            # The primal of this VJP is the local slice, not the full grouped
            # tensor. No group-axis scatter is needed for this derivative.
            _, (gradient,) = mx.vjp(
                projection,
                [inp[..., index, :]],
                [cotangent[..., offset : offset + width]],
            )
            pieces.append(materialize(gradient))
            offset += width
        return materialize(mx.stack(pieces, axis=-2))

    return forward(value)
