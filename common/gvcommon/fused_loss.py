"""Autograd wrapper of the fused volume loss (common/cuda/volume_loss.cuh), which both
extensions compile: `ext` is the extension module whose kernels run."""

import torch

_FP_KINDS = {"lin": 0, "log": 1, "hinge": 2}


class _FusedVolumeLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cells, gt, weights, ext, cfg):
        out, acc = ext.fused_loss_forward(cells, gt, weights, *cfg)
        ctx.ext, ctx.cfg = ext, cfg
        ctx.save_for_backward(cells, gt, weights, acc)
        ctx.mark_non_differentiable(gt)
        return out

    @staticmethod
    def backward(ctx, g_out):
        cells, gt, weights, acc = ctx.saved_tensors
        dc, dw = ctx.ext.fused_loss_backward(
            cells, gt, weights, acc, *ctx.cfg, g_out.contiguous()
        )
        return dc, None, dw, None, None


def volume_loss(
    ext,
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
    """[l_v + fn + fp, l_v, fn, fp]; see volume_loss.cuh for the terms. mask_data: the data
    term skips cells where gt or the render is -1 (else it covers every cell)."""
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
    return _FusedVolumeLoss.apply(cells, gt, weights, ext, cfg)
