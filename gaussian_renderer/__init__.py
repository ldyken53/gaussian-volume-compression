
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from scene.gaussian_model import GaussianModel

_rasterizer: GaussianRasterizer | None = None

def init_rasterizer(
    pc: GaussianModel,
    pipe,
    samples,
    cell_count=100,
    bg=-1.0,
    scaling_modifier=1.0
) -> None:
    """
    Initialize the GaussianRasterizer once with these settings.
    Must be called before any render(...) calls.
    """
    global _rasterizer

    raster_settings = GaussianRasterizationSettings(
        volume_mins=pc.mins,
        volume_maxes=pc.maxes,
        cell_count=cell_count,
        bg=bg,
        scale_modifier=scaling_modifier,
        debug=pipe.debug,
    )

    # store the rasterizer; we'll patch the settings per-call
    _rasterizer = GaussianRasterizer(
        raster_settings=raster_settings,
        samples=samples
    )


def render(
    pc: GaussianModel,
    pipe,
    jitter,
    cell_count=100,
    bg=-1.0,
    scaling_modifier=1.0
):
    """
    Render the scene.

    """

    means3D = pc.get_xyz
    scales = pc.get_scaling
    rotations = pc.get_rotation
    values = pc.get_values
    weights = pc.get_weight

    out_cells, out_weights = _rasterizer(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        values=values,
        weights=weights,
        jitter=jitter
    )

    out = {
        "cells": out_cells,
        "weights": out_weights
    }

    return out
