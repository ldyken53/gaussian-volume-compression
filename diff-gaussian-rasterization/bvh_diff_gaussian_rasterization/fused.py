"""Autograd wrappers for the fused activation and loss kernels in fused_ops.cu
(ported from gaussian-volume-struct)."""
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
    def forward(ctx, cells, gt, weights, cfg):
        out, acc = _C.fused_loss_forward(cells, gt, weights, *cfg)
        ctx.cfg = cfg
        ctx.save_for_backward(cells, gt, weights, acc)
        ctx.mark_non_differentiable(gt)
        return out

    @staticmethod
    def backward(ctx, g_out):
        cells, gt, weights, acc = ctx.saved_tensors
        dc, dw = _C.fused_loss_backward(cells, gt, weights, acc, *ctx.cfg, g_out.contiguous())
        return dc, None, dw, None


_FP_KINDS = {"lin": 0, "log": 1, "hinge": 2}


def fused_volume_loss(cells, gt, weights, *, l1, masked, fn_mode, fn_thresh, fn_reg, fp_reg=0.0,
                      fn_rel=False, fp_t=0.0, fp_mode="lin", norm_all=False, rel_pow=0.0):
    """[l_v + fn + fp, l_v, fn, fp] exactly as train.py's torch path computes them.

    l1: |c-g| instead of (c-g)^2. masked: data mean over cells with gt != -1 and
    cells != -1 (non-surface models) instead of over every cell. fn_mode: "abs" / "rel"
    (linear hinge at fn_thresh, rel scaling fn_reg by the detached data loss) or "log"
    (log barrier at fn_thresh); fn_rel applies rel's data-loss scaling to any mode.
    fp_reg > 0 adds the --encode_surface FP term on exterior cells with w > 0: fp_mode "lin" (w),
    "log" (clamp(log(w / fp_t), 0)) or "hinge" (clamp(w - fp_t, 0), mean over its violators).
    norm_all: FN / FP sums over the number of interior / exterior cells instead.
    rel_pow > 0: both hinge weights times l_v^rel_pow (detached); 0.5 with L2 = RMSE scaling.
    """
    cfg = (int(bool(l1)), int(bool(masked)), int(fn_mode == "log"), int(fn_mode == "rel" or fn_rel),
           float(fn_thresh), float(fn_reg), float(fp_reg), float(fp_t), _FP_KINDS[fp_mode], int(bool(norm_all)), float(rel_pow))
    return _FusedVolumeLoss.apply(cells, gt, weights, cfg)
