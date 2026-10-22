"""BVH backend: Gaussians rendered at arbitrary sample points (unstructured meshes).

The samples are those of the last build_bvh / build_bvh_cached call; every render until the
next build evaluates at them.
"""

import torch

from . import _bvh as _C

# (samples): the points every render evaluates at
build_bvh = _C.build_bvh
# (samples, key): build_bvh, reusing the BVH built for batch key
build_bvh_cached = _C.build_bvh_cached
# (): primIDs of the last BVH, in leaf order
sample_bvh_prim_ids = _C.sample_bvh_prim_ids
# (samples, key): register exterior sample set key for margin_fp
margin_set_exterior = _C.margin_set_exterior
# (means, conic, weights, T, key) -> (value, d means, d conic, d weights)
margin_fp = _C.margin_fp


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
    """(cells, aw) at the last BVH build's samples under the render rule (occ_t, floor):
    a sample is empty (-1) where aw <= occ_t, else S / max(aw, floor)."""
    return _RasterizeGaussians.apply(means3D, scales, rotations, values, weights, rule)


class _FusedActivations(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_scaling, raw_rotation, raw_values, raw_weight, max_scale):
        scales, rotations, values, weights = _C.fused_act_forward(
            raw_scaling, raw_rotation, raw_values, raw_weight, float(max_scale)
        )
        ctx.max_scale = float(max_scale)
        ctx.save_for_backward(raw_scaling, raw_rotation, values, weights)
        return scales, rotations, values, weights

    @staticmethod
    def backward(ctx, g_scales, g_rotations, g_values, g_weights):
        raw_scaling, raw_rotation, values, weights = ctx.saved_tensors
        ds, dq, dv, dw = _C.fused_act_backward(
            raw_scaling,
            raw_rotation,
            values,
            weights,
            ctx.max_scale,
            g_scales.contiguous(),
            g_rotations.contiguous(),
            g_values.contiguous(),
            g_weights.contiguous(),
        )
        return ds, dq, dv, dw, None


def fused_activations(raw_scaling, raw_rotation, raw_values, raw_weight, max_scale):
    """(exp + soft norm cap, normalize, sigmoid, sigmoid) of the raw parameters."""
    return _FusedActivations.apply(
        raw_scaling, raw_rotation, raw_values, raw_weight, max_scale
    )
