"""Autograd wrappers for the fused activation and loss kernels in fused_ops.cu."""
import torch
from . import _C


class _FusedActivations(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_scaling, raw_rotation, raw_values, raw_weight, max_scale):
        scales, rotations, values, weights = _C.fused_act_forward(
            raw_scaling, raw_rotation, raw_values, raw_weight, float(max_scale))
        ctx.max_scale = float(max_scale)
        ctx.save_for_backward(raw_scaling, raw_rotation, values, weights)
        return scales, rotations, values, weights

    @staticmethod
    def backward(ctx, g_scales, g_rotations, g_values, g_weights):
        raw_scaling, raw_rotation, values, weights = ctx.saved_tensors
        ds, dq, dv, dw = _C.fused_act_backward(
            raw_scaling, raw_rotation, values, weights, ctx.max_scale,
            g_scales.contiguous(), g_rotations.contiguous(),
            g_values.contiguous(), g_weights.contiguous())
        return ds, dq, dv, dw, None


def fused_activations(raw_scaling, raw_rotation, raw_values, raw_weight, max_scale):
    """(exp + soft norm cap, normalize, sigmoid, sigmoid) of the raw parameters."""
    return _FusedActivations.apply(raw_scaling, raw_rotation, raw_values, raw_weight, max_scale)


class _FusedVolumeLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cells, gt, weights, fn_reg):
        out, acc = _C.fused_loss_forward(cells, gt, weights, float(fn_reg))
        ctx.fn_reg = float(fn_reg)
        ctx.save_for_backward(cells, gt, weights, acc)
        ctx.mark_non_differentiable(gt)
        return out

    @staticmethod
    def backward(ctx, g_out):
        cells, gt, weights, acc = ctx.saved_tensors
        dc, dw = _C.fused_loss_backward(cells, gt, weights, acc, ctx.fn_reg, g_out.contiguous())
        return dc, None, dw, None


def fused_volume_loss(cells, gt, weights, fn_reg):
    """[mean (c-g)^2 + FN term, the mean, the FN term], as train.py computes them with
    --loss l2 --fn_mode abs."""
    return _FusedVolumeLoss.apply(cells, gt, weights, fn_reg)
