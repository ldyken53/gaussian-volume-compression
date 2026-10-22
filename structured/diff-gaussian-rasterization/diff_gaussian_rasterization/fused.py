from gvcommon.fused_loss import volume_loss

from . import _C


def fused_volume_loss(cells, gt, weights, fn_reg, tau):
    """[loss, mean squared error over every cell, log-barrier false-negative term at threshold
    tau, 0]: the shared fused volume loss (gvcommon.fused_loss) on this extension's kernel.
    """
    return volume_loss(
        _C,
        cells,
        gt,
        weights,
        l1=False,
        fn_mode="log",
        fn_thresh=tau,
        fn_reg=fn_reg,
        mask_data=False,
    )
