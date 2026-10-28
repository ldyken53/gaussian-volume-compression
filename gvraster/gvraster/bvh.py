"""Sampling at arbitrary points through a BVH over them (meshes), and the activations.

set_points makes points the positions every later sample() evaluates at.
"""

import torch

from . import _bvh as _C

# (points)
set_points = _C.set_points
# (points, key): set_points, reusing the BVH built for key
set_points_cached = _C.set_points_cached
# () -> the points' indices in BVH leaf order
leaf_order = _C.leaf_order
# (flags): one float per current point; > 0 flags it for sample()'s reach
set_point_flags = _C.set_point_flags
# (points, key): register exterior point set key for margin_fp
margin_set_exterior = _C.margin_set_exterior
# (means, conics, weights, T, key) -> (value, d means, d conics, d weights)
margin_fp = _C.margin_fp


class _Sample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, scales, rotations, values, weights, rule, reach):
        threshold, floor = float(rule[0]), float(rule[1])
        out_values, influence, conics = _C.sample_forward(
            means, scales, rotations, values, weights, threshold, floor, reach
        )
        ctx.rule = (threshold, floor)
        ctx.save_for_backward(
            means, scales, rotations, values, weights, out_values, influence, conics
        )
        return out_values, influence

    @staticmethod
    def backward(ctx, grad_values, grad_influence):
        means, scales, rotations, values, weights, out_values, influence, conics = (
            ctx.saved_tensors
        )
        grads = _C.sample_backward(
            means,
            scales,
            rotations,
            conics,
            values,
            weights,
            out_values,
            influence,
            grad_values,
            grad_influence,
            *ctx.rule,
        )
        return (*grads, None, None)


def sample(means, scales, rotations, values, weights, rule, reach=None):
    """(values, influence) of the activated Gaussians at the points. With rule (T, floor), a
    point is undefined (-1) where its influence I <= T, and is otherwise V / max(I, floor).
    reach, if given (P floats): set to each Gaussian's largest w G at a flagged point.
    """
    return _Sample.apply(means, scales, rotations, values, weights, rule, reach)


class _Activate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_scales, raw_rotations, raw_values, raw_weights, max_scale):
        scales, rotations, values, weights = _C.activate_forward(
            raw_scales, raw_rotations, raw_values, raw_weights, float(max_scale)
        )
        ctx.max_scale = float(max_scale)
        ctx.save_for_backward(raw_scales, raw_rotations, values, weights)
        return scales, rotations, values, weights

    @staticmethod
    def backward(ctx, d_scales, d_rotations, d_values, d_weights):
        raw_scales, raw_rotations, values, weights = ctx.saved_tensors
        grads = _C.activate_backward(
            raw_scales,
            raw_rotations,
            values,
            weights,
            ctx.max_scale,
            d_scales.contiguous(),
            d_rotations.contiguous(),
            d_values.contiguous(),
            d_weights.contiguous(),
        )
        return (*grads, None)


def activate(raw_scales, raw_rotations, raw_values, raw_weights, max_scale):
    """(scales, rotations, values, weights): exp softly capped at norm max_scale, normalize,
    sigmoid, sigmoid."""
    return _Activate.apply(
        raw_scales, raw_rotations, raw_values, raw_weights, max_scale
    )
