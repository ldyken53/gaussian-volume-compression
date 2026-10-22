"""Training kernels both renderers use: the one-launch Adam step (+ EMA) and the fused volume
loss with its autograd wrapper (terms in csrc/ops/volume_loss.cu)."""

import torch

from . import _ops as _C

# (params, grads, exp_avgs, exp_avg_sqs, lrs, step, beta1, beta2, eps, emas, ema_w): one Adam
# step over several tensors in a single launch; emas (may be empty) lerp toward the new params.
multi_adam_ = _C.multi_adam_

_FP_KINDS = {"lin": 0, "log": 1, "hinge": 2}


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


def volume_loss(
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
    mask_data=True,
):
    """[l_v + fn + fp, l_v, fn, fp]; see csrc/ops/volume_loss.cu for the terms. mask_data: the
    data term skips cells where gt or the render is -1 (else it covers every cell)."""
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
        int(bool(mask_data)),
    )
    return _FusedVolumeLoss.apply(cells, gt, weights, cfg)
