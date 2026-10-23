"""Sampling a model (Eq. 1 of the paper). With V = sum w G v and I = sum w G (the accumulated
influence) over the Gaussians, a sample is undefined (-1) where I <= threshold and is otherwise
V / max(I, floor). A rule is the pair (threshold, floor); models train with threshold 0.
"""

import torch

from gvraster import bvh, tile
from veggies.utils import build_rotation

RULES = {
    "structured": (0.002, 0.01),
    "interior": (0.002, 0.01),
    "surface": (0.01, 0.01),
}


def legacy_rule(surface):
    """The rule of models saved before they recorded one."""
    return (0.01, 0.01) if surface else (0.0, 0.01)


def set_points(points):
    """Make points the positions sample_points evaluates at (builds a BVH over them)."""
    bvh.build_bvh(points)


def set_points_cached(points, key):
    """set_points, reusing the BVH built for batch key."""
    bvh.build_bvh_cached(points, int(key))


def leaf_order():
    """Indices of the current points in BVH leaf order."""
    return bvh.sample_bvh_prim_ids()


def sample_points(model, rule):
    """(values, influence) at the current points."""
    means = model.means
    scales, rotations, values, weights = bvh.fused_activations(
        model._scales, model._rotations, model._values, model._weights, model.max_scale
    )
    order = model.locality_order()
    return bvh.rasterize_gaussians(
        means[order],
        scales[order],
        rotations[order],
        values[order],
        weights[order],
        rule,
    )


def sample_grid(model, jitter, cells, rule):
    """(values, influence) on the cells^3 lattice over the model's bounds, each position moved by
    its jitter (3 floats per cell, cells in [z, y, x] order)."""
    settings = tile.GaussianRasterizationSettings(
        volume_mins=model.bounds_min,
        volume_maxes=model.bounds_max,
        cell_count=cells,
        max_scale=float(model.max_scale),
        occ_t=float(rule[0]),
        floor=float(rule[1]),
    )
    return tile.GaussianRasterizer(settings)(
        model.means,
        model._scales,
        model._rotations,
        model._values,
        model._weights,
        jitter,
    )


# --fp_mode margin: a false positive term on the Gaussians' supports.


def conic6(model):
    """The inverse covariance (with the rasterizer's epsilon) as a differentiable (P, 6) tensor,
    in float64 with a floored determinant."""
    s = model.scales.double()
    R = build_rotation(model._rotations).double()
    S = (R * (s * s)[:, None, :]) @ R.transpose(1, 2)
    eps = (
        torch.stack([S[:, 0, 0].abs(), S[:, 1, 1].abs(), S[:, 2, 2].abs()], 1)
        .max(1)
        .values
        * 1e-5
    )
    a, b, c = S[:, 0, 0] + eps, S[:, 0, 1], S[:, 0, 2]
    d, e, f = S[:, 1, 1] + eps, S[:, 1, 2], S[:, 2, 2] + eps
    det = (a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)).clamp_min(
        1e-200
    )
    inv = 1.0 / det
    return torch.stack(
        [
            (d * f - e * e) * inv,
            (c * e - b * f) * inv,
            (b * e - c * d) * inv,
            (a * f - c * c) * inv,
            (b * c - a * e) * inv,
            (a * d - b * b) * inv,
        ],
        1,
    )


class _MarginFP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, conic, weights, T, key):
        val, gmu, gc, gw = bvh.margin_fp(
            means.detach().float().contiguous(),
            conic.detach().float().contiguous(),
            weights.detach().reshape(-1).float().contiguous(),
            float(T),
            int(key),
        )
        # A degenerate Gaussian (float32 conic overflow) contributes nothing rather than NaN.
        gmu, gc, gw = (
            torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x in (gmu, gc, gw)
        )
        ctx.save_for_backward(gmu, gc.to(conic.dtype), gw)
        val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
        ctx.wshape = weights.shape
        return val.sum()

    @staticmethod
    def backward(ctx, g):
        gmu, gc, gw = ctx.saved_tensors
        return g * gmu, g * gc, (g * gw).reshape(ctx.wshape), None, None


def margin_set_exterior(points, key):
    bvh.margin_set_exterior(points.contiguous(), int(key))


def margin_fp(model, key, T):
    """Sum over the exterior points of set key reached by a Gaussian's support of qmax - q."""
    return _MarginFP.apply(model.means, conic6(model), model.weights, T, key)
