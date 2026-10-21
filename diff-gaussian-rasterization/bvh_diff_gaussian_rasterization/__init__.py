import torch

from . import _C


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, scales, rotations, values, weights, rule):
        occ_t, floor = float(rule[0]), float(rule[1])
        cells, cell_weights, conics = _C.rasterize_gaussians(
            means3D, scales, rotations, values, weights, occ_t, floor
        )
        ctx.rule = (occ_t, floor)
        ctx.save_for_backward(
            means3D, scales, rotations, values, weights, cells, cell_weights, conics
        )
        return cells, cell_weights

    @staticmethod
    def backward(ctx, grad_cells, grad_cell_weights):
        means3D, scales, rotations, values, weights, cells, cell_weights, conics = (
            ctx.saved_tensors
        )
        grads = _C.rasterize_gaussians_backward(
            means3D,
            scales,
            rotations,
            conics,
            values,
            weights,
            cells,
            cell_weights,
            grad_cells,
            grad_cell_weights,
            *ctx.rule,
        )
        return (*grads, None)


def rasterize_gaussians(means3D, scales, rotations, values, weights, rule):
    """(cells, accumulated weights) at the samples of the last BVH build under the render rule
    (occ_t, floor): a sample is empty (-1) where aw <= occ_t, else it renders S / max(aw, floor)."""
    return _RasterizeGaussians.apply(means3D, scales, rotations, values, weights, rule)
