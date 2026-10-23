"""The training kernels: Adam over several tensors in one launch, updating their parameter
averages too, and the training loss."""

import torch

from . import _ops as _C

# (params, grads, exp_avgs, exp_avg_sqs, lrs, step, beta1, beta2, eps, emas, ema_w): one Adam
# step; emas (may be empty) move to lerp(ema, new param, ema_w).
multi_adam_ = _C.multi_adam_

_FP_MODES = {"lin": 0, "log": 1, "hinge": 2}


class _VolumeLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, gt, influence, cfg):
        out, acc = _C.volume_loss_forward(values, gt, influence, *cfg)
        ctx.cfg = cfg
        ctx.save_for_backward(values, gt, influence, acc)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        values, gt, influence, acc = ctx.saved_tensors
        d_values, d_influence = _C.volume_loss_backward(
            values, gt, influence, acc, *ctx.cfg, grad_out.contiguous()
        )
        return d_values, None, d_influence, None


def volume_loss(
    values,
    gt,
    influence,
    *,
    l1,
    fn_mode,
    fn_tau,
    lambda_fn,
    lambda_fp=0.0,
    fp_tau=0.0,
    fp_mode="lin",
    norm_all=False,
    rel_pow=0.0,
    mask_data=True,
):
    """[loss, data, FN, FP]; csrc/ops/volume_loss.cu defines the terms."""
    cfg = (
        int(bool(l1)),
        int(fn_mode == "log"),
        int(fn_mode == "rel"),
        float(fn_tau),
        float(lambda_fn),
        float(lambda_fp),
        float(fp_tau),
        _FP_MODES[fp_mode],
        int(bool(norm_all)),
        float(rel_pow),
        int(bool(mask_data)),
    )
    return _VolumeLoss.apply(values, gt, influence, cfg)
