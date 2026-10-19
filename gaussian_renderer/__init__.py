from bvh_diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from scene.gaussian_model import GaussianModel
import os
import torch

# The Morton reorder is a memory-locality hint, but it gathers five grad-requiring
# tensors every render, so autograd pays a scatter-add back on each one -- 0.75 ms an
# iteration at mito 64x (index_put_ 0.42 + indexing_backward 0.16 + index 0.14), which
# is ~9% of the iteration once the batch is subsampled. GV_MORTON=0 disables it so the
# hint can be priced against what it costs.
_MORTON = os.environ.get("GV_MORTON", "1") != "0"

try:
    from bvh_diff_gaussian_rasterization.fused import fused_activations
except ImportError:  # a rasterizer build without fused_ops.cu
    fused_activations = None
# One kernel each way for get_scaling / get_rotation / get_values / get_weight instead of
# ~20 elementwise launches (struct d0b03fb). FUSED=0 restores the torch getters.
_FUSED = fused_activations is not None and os.environ.get("FUSED", "1") != "0"

_rasterizer: GaussianRasterizer | None = None

_morton_order = None
_morton_state = (0, 0)  # (gaussian count, renders since refresh)

def init_rasterizer(
    pc: GaussianModel,
    pipe,
    cell_count=100,
    bg=-1.0,
    scaling_modifier=1.0,
    use_gaussian_bvh=False
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
        use_gaussian_bvh=use_gaussian_bvh,
        debug=pipe.debug,
    )

    # store the rasterizer; we'll patch the settings per-call
    _rasterizer = GaussianRasterizer(
        raster_settings=raster_settings,
    )


def build_bvh(samples, debug=False, use_gaussian_bvh=False):
    _rasterizer.build_bvh(samples, debug, use_gaussian_bvh)


def build_bvh_cached(samples, key):
    """Select batch `key`'s sample BVH, building it the first time (full-batch runs only:
    the cache holds one BVH per distinct sample set)."""
    from bvh_diff_gaussian_rasterization import _C
    _C.build_bvh_cached(samples, int(key))


def sample_bvh_prim_ids():
    """primIDs of the sample BVH last built (int32, leaf order)."""
    from bvh_diff_gaussian_rasterization import _C
    return _C.sample_bvh_prim_ids()


def morton3d_unit(xyz: torch.Tensor) -> torch.Tensor:
    """Morton codes for points already in [0, 1]^3."""
    norm = (xyz * ((1 << 21) - 1)).long()

    def part1by2(n):
        n = n & 0x1fffff
        n = (n | (n << 32)) & 0x1f00000000ffff
        n = (n | (n << 16)) & 0x1f0000ff0000ff
        n = (n | (n << 8))  & 0x100f00f00f00f00f
        n = (n | (n << 4))  & 0x10c30c30c30c30c3
        n = (n | (n << 2))  & 0x1249249249249249
        return n

    return part1by2(norm[:, 0]) | (part1by2(norm[:, 1]) << 1) | (part1by2(norm[:, 2]) << 2)


def render(
    pc: GaussianModel,
    debug = False
):
    """
    Render the scene.

    """

    global _morton_order, _morton_state
    means3D = pc.get_xyz
    if _FUSED:
        scales, rotations, values, weights = fused_activations(
            pc._scaling, pc._rotation, pc._values, pc._weight, pc.max_scale)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        values = pc.get_values
        weights = pc.get_weight

    # Sort Gaussians by Morton code for spatial coherence. The order is only a
    # memory-locality hint and goes stale slowly, so refresh it every 100 renders
    # and whenever the Gaussian count changes (densification) instead of paying
    # argsort + five gathers every iteration.
    n = means3D.shape[0]
    if not _MORTON:
        out_cells, out_weights = _rasterizer(
            means3D=means3D, scales=scales, rotations=rotations,
            values=values, weights=weights, debug=debug)
        return {"cells": out_cells, "weights": out_weights}
    prev_n, age = _morton_state
    if _morton_order is None or prev_n != n or age >= 100:
        _morton_order = torch.argsort(morton3d_unit(means3D))
        _morton_state = (n, 0)
    else:
        _morton_state = (n, age + 1)
    order = _morton_order
    means3D = means3D[order]
    scales = scales[order]
    rotations = rotations[order]
    values = values[order]
    weights = weights[order]

    out_cells, out_weights = _rasterizer(
        means3D=means3D,
        scales=scales,
        rotations=rotations,
        values=values,
        weights=weights,
        debug=debug
    )

    out = {
        "cells": out_cells,
        "weights": out_weights
    }

    return out

# ---- Support-margin false-positive term (--fp_mode margin, see margin.cu) --------------
from utils.general_utils import build_rotation


def conic6(pc: GaussianModel):
    """The rasterizer's conic (inverse covariance, epsilon included) as a differentiable
    (P, 6) tensor (c00, c01, c02, c11, c12, c22): Sigma = R diag(s^2) R^T exactly as
    computeCov3D builds it, inverted with the same cofactor formula as preprocessCUDA."""
    # float64 with a floored determinant: a near-degenerate needle's float32 conic is inf, and
    # its backward then multiplies a zero margin gradient by inf (NaN positions -> crash).
    s = pc.get_scaling.double()
    R = build_rotation(pc._rotation).double()
    S = (R * (s * s)[:, None, :]) @ R.transpose(1, 2)
    eps = torch.stack([S[:, 0, 0].abs(), S[:, 1, 1].abs(), S[:, 2, 2].abs()], 1).max(1).values * 1e-5
    a, b, c = S[:, 0, 0] + eps, S[:, 0, 1], S[:, 0, 2]
    d, e, f = S[:, 1, 1] + eps, S[:, 1, 2], S[:, 2, 2] + eps
    det = (a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)).clamp_min(1e-200)
    inv = 1.0 / det
    return torch.stack([(d * f - e * e) * inv, (c * e - b * f) * inv, (b * e - c * d) * inv,
                        (a * f - c * c) * inv, (b * c - a * e) * inv, (a * d - b * b) * inv], 1)


class _MarginFP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, conic, weights, T, key):
        from bvh_diff_gaussian_rasterization import _C
        val, gmu, gc, gw = _C.margin_fp(means.detach().float().contiguous(), conic.detach().float().contiguous(),
                                        weights.detach().reshape(-1).float().contiguous(), float(T), int(key))
        # Degenerate Gaussians (float32 conic overflow) contribute nothing rather than NaN.
        gmu, gc, gw = (torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x in (gmu, gc, gw))
        ctx.save_for_backward(gmu, gc.to(conic.dtype), gw)
        val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
        ctx.wshape = weights.shape
        return val.sum()

    @staticmethod
    def backward(ctx, g):
        gmu, gc, gw = ctx.saved_tensors
        return g * gmu, g * gc, (g * gw).reshape(ctx.wshape), None, None


def margin_set_exterior(samples, key):
    """Register (and BVH) exterior sample set `key` for margin_fp."""
    from bvh_diff_gaussian_rasterization import _C
    _C.margin_set_exterior(samples.contiguous(), int(key))


def margin_fp(pc: GaussianModel, key, T):
    """Sum over exterior samples of set `key` and Gaussians whose support reaches them of
    (qmax - q): zero exactly when no support reaches the exterior."""
    return _MarginFP.apply(pc.get_xyz, conic6(pc), pc.get_weight, T, key)
