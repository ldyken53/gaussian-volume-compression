from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from scene.gaussian_model import GaussianModel


def render(pc: GaussianModel, jitter, cell_count=100):
    """Render the model onto a cell_count^3 jittered lattice: {"cells", "weights"}."""
    settings = GaussianRasterizationSettings(
        volume_mins=pc.mins,
        volume_maxes=pc.maxes,
        cell_count=cell_count,
        max_scale=float(pc.max_scale),
        occ_t=float(pc.occ_t),
        floor=float(pc.floor),
    )
    cells, weights = GaussianRasterizer(settings)(
        pc.get_xyz, pc._scaling, pc._rotation, pc._values, pc._weight, jitter
    )
    return {"cells": cells, "weights": weights}
