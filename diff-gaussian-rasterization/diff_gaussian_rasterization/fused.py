import torch

from . import _C


class _FusedVolumeLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cells, gt, weights, fn_reg, tau):
        out, acc = _C.fused_loss_forward(cells, gt, weights, float(fn_reg), float(tau))
        ctx.fn_reg, ctx.tau = float(fn_reg), float(tau)
        ctx.save_for_backward(cells, gt, weights, acc)
        ctx.mark_non_differentiable(gt)
        return out

    @staticmethod
    def backward(ctx, g_out):
        cells, gt, weights, acc = ctx.saved_tensors
        dc, dw = _C.fused_loss_backward(
            cells, gt, weights, acc, ctx.fn_reg, ctx.tau, g_out.contiguous()
        )
        return dc, None, dw, None, None


def fused_volume_loss(cells, gt, weights, fn_reg, tau):
    """[loss, mean squared error, log-barrier false-negative term at threshold tau]."""
    return _FusedVolumeLoss.apply(cells, gt, weights, fn_reg, tau)
