#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from scene.gaussian_model import GaussianModel

try:
    from diff_gaussian_rasterization.fused import fused_activations
except ImportError:  # a rasterizer build without fused_ops.cu
    fused_activations = None
try:
    from diff_gaussian_rasterization import _C as _dgr_C
    # A FUSE_ACT build applies the activations inside the rasterizer: it takes the raw
    # parameters, with max_scale passed in the scale_modifier slot.
    _RAW_ACT = bool(getattr(_dgr_C, "RAW_ACTIVATIONS", False))
except ImportError:
    _RAW_ACT = False
import os


def render(
    pc: GaussianModel,
    pipe,
    jitter,
    cell_count=100,
    bg=-1.0,
    scaling_modifier=1.0,
    debug=False
):
    """
    Render the scene.

    """

    # A FUSE_ACT build always takes raw parameters: FUSED=0 only switches the torch-side
    # activations of builds that do not apply them (it used to hand a raw build activated
    # values, which it then activated a second time).
    raw = _RAW_ACT
    if raw and scaling_modifier != 1.0:
        raise ValueError("a FUSE_ACT rasterizer build uses scale_modifier for max_scale")
    raster_settings = GaussianRasterizationSettings(
        volume_mins=pc.mins,
        volume_maxes=pc.maxes,
        cell_count=cell_count,
        bg=bg,
        scale_modifier=float(pc.max_scale) if raw else scaling_modifier,
        debug=pipe.debug if not debug else True,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    if raw:
        scales, rotations, values, weights = pc._scaling, pc._rotation, pc._values, pc._weight
    elif fused_activations is not None and os.environ.get("FUSED", "1") != "0":
        # One kernel each way for get_scaling / get_rotation / get_values / get_weight.
        scales, rotations, values, weights = fused_activations(
            pc._scaling, pc._rotation, pc._values, pc._weight, pc.max_scale)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        values = pc.get_values
        weights = pc.get_weight

    # Rasterize visible Gaussians to cells, obtain their radii
    out_cells, out_weights, radii = rasterizer(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        values=values,
        weights=weights,
        jitter=jitter,
        debug=debug
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    out = {
        "cells": out_cells,
        "weights": out_weights,
        # Boolean mask, not .nonzero(): nonzero() must read the result count back to
        # the host to size its output, forcing a device sync every iteration. No caller
        # uses this as indices -- train.py and volume_metrics.py only unpack it.
        "visibility_filter": radii > 0,
        "radii": radii,
    }

    return out
