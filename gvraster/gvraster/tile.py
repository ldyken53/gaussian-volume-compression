"""Sampling on a jittered lattice with a tile rasterizer (grids)."""

import torch

from . import _tile as _C


class _Sample(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, means, raw_scales, raw_rotations, raw_values, raw_weights, jitter, setup
    ):
        pairs, values, influence, *buffers = _C.sample_forward(
            means, raw_scales, raw_rotations, raw_values, raw_weights, jitter, *setup
        )
        ctx.setup, ctx.pairs = setup, pairs
        ctx.save_for_backward(
            raw_scales,
            raw_rotations,
            raw_values,
            raw_weights,
            jitter,
            values,
            influence,
            *buffers,
        )
        return values, influence

    @staticmethod
    def backward(ctx, grad_values, grad_influence):
        (
            raw_scales,
            raw_rotations,
            raw_values,
            raw_weights,
            jitter,
            values,
            influence,
            *buffers,
        ) = ctx.saved_tensors
        grads = _C.sample_backward(
            raw_scales,
            raw_rotations,
            raw_values,
            raw_weights,
            jitter,
            *ctx.setup,
            values,
            influence,
            grad_values,
            grad_influence,
            ctx.pairs,
            *buffers,
        )
        return (*grads, None, None)


def sample(
    means,
    raw_scales,
    raw_rotations,
    raw_values,
    raw_weights,
    jitter,
    bounds_min,
    bounds_max,
    cells,
    max_scale,
    rule,
):
    """(values, influence) of the Gaussians with raw parameters at the cells^3 lattice points
    spanning [bounds_min, bounds_max], each moved by its jitter (3 floats per point, in [z, y, x]
    order), under the rule of gvraster.bvh.sample."""
    setup = (
        *(float(x) for x in bounds_min),
        *(float(x) for x in bounds_max),
        int(cells),
        float(max_scale),
        float(rule[0]),
        float(rule[1]),
    )
    return _Sample.apply(
        means, raw_scales, raw_rotations, raw_values, raw_weights, jitter, setup
    )
