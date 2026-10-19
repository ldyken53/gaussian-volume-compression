"""Autograd wrappers for the fused activation and loss kernels in fused_ops.cu."""

import torch

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


class _FusedVolumeLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cells, gt, weights, cfg):
        out, acc = _C.fused_loss_forward(cells, gt, weights, *cfg)
        ctx.cfg = cfg
        ctx.save_for_backward(cells, gt, weights, acc)
        ctx.mark_non_differentiable(gt)
        return out

    @staticmethod
    def backward(ctx, g_out):
        cells, gt, weights, acc = ctx.saved_tensors
        dc, dw = _C.fused_loss_backward(
            cells, gt, weights, acc, *ctx.cfg, g_out.contiguous()
        )
        return dc, None, dw, None


_FP_KINDS = {"lin": 0, "log": 1, "hinge": 2}


def fused_volume_loss(
    cells,
    gt,
    weights,
    *,
    l1,
    fn_mode,
    fn_thresh,
    fn_reg,
    fp_reg=0.0,
    fp_t=0.0,
    fp_mode="lin",
    norm_all=False,
    rel_pow=0.0,
):
    """[l_v + fn + fp, l_v, fn, fp]; see fused_ops.cu for the terms."""
    cfg = (
        int(bool(l1)),
        int(fn_mode == "log"),
        int(fn_mode == "rel"),
        float(fn_thresh),
        float(fn_reg),
        float(fp_reg),
        float(fp_t),
        _FP_KINDS[fp_mode],
        int(bool(norm_all)),
        float(rel_pow),
    )
    return _FusedVolumeLoss.apply(cells, gt, weights, cfg)
