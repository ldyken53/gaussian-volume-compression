from typing import NamedTuple

import torch
import torch.nn as nn

from . import _C


class GaussianRasterizationSettings(NamedTuple):
    volume_mins: tuple[float, float, float]
    volume_maxes: tuple[float, float, float]
    cell_count: int
    max_scale: float


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, scales, rotations, values, weights, jitter, settings):
        num_rendered, cells, cell_weights, geom, binning, img = _C.rasterize_gaussians(
            means3D,
            scales,
            rotations,
            values,
            weights,
            jitter,
            settings.max_scale,
            *settings.volume_mins,
            *settings.volume_maxes,
            settings.cell_count,
        )
        ctx.settings = settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            scales,
            rotations,
            values,
            weights,
            jitter,
            geom,
            binning,
            img,
            cells,
            cell_weights,
        )
        return cells, cell_weights

    @staticmethod
    def backward(ctx, grad_cells, grad_cell_weights):
        s = ctx.settings
        (
            scales,
            rotations,
            values,
            weights,
            jitter,
            geom,
            binning,
            img,
            cells,
            cell_weights,
        ) = ctx.saved_tensors
        grads = _C.rasterize_gaussians_backward(
            scales,
            rotations,
            values,
            weights,
            jitter,
            cells,
            cell_weights,
            s.max_scale,
            *s.volume_mins,
            *s.volume_maxes,
            s.cell_count,
            grad_cells,
            grad_cell_weights,
            geom,
            ctx.num_rendered,
            binning,
            img,
        )
        return (*grads, None, None)


class GaussianRasterizer(nn.Module):
    """Renders raw (pre-activation) Gaussian parameters onto a jittered cell lattice."""

    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means3D, scales, rotations, values, weights, jitter):
        return _RasterizeGaussians.apply(
            means3D, scales, rotations, values, weights, jitter, self.raster_settings
        )
