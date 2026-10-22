"""Autograd wrappers for the fused activation kernels in fused_ops.cu, and the shared fused
volume loss (gvcommon.fused_loss) on this extension's kernel."""

import torch
from gvcommon.fused_loss import volume_loss

from . import _C


class _FusedActivations(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_scaling, raw_rotation, raw_values, raw_weight, max_scale):
        scales, rotations, values, weights = _C.fused_act_forward(
            raw_scaling, raw_rotation, raw_values, raw_weight, float(max_scale)
        )
        ctx.max_scale = float(max_scale)
        ctx.save_for_backward(raw_scaling, raw_rotation, values, weights)
        return scales, rotations, values, weights

    @staticmethod
    def backward(ctx, g_scales, g_rotations, g_values, g_weights):
        raw_scaling, raw_rotation, values, weights = ctx.saved_tensors
        ds, dq, dv, dw = _C.fused_act_backward(
            raw_scaling,
            raw_rotation,
            values,
            weights,
            ctx.max_scale,
            g_scales.contiguous(),
            g_rotations.contiguous(),
            g_values.contiguous(),
            g_weights.contiguous(),
        )
        return ds, dq, dv, dw, None


def fused_activations(raw_scaling, raw_rotation, raw_values, raw_weight, max_scale):
    """(exp + soft norm cap, normalize, sigmoid, sigmoid) of the raw parameters."""
    return _FusedActivations.apply(
        raw_scaling, raw_rotation, raw_values, raw_weight, max_scale
    )


def fused_volume_loss(cells, gt, weights, **terms):
    """[l_v + fn + fp, l_v, fn, fp]; terms as gvcommon.fused_loss.volume_loss. The data term
    skips samples where gt or the render is -1."""
    return volume_loss(_C, cells, gt, weights, **terms)
