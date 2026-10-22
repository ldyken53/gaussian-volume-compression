import os

import torch

from bvh_diff_gaussian_rasterization import _C, rasterize_gaussians
from bvh_diff_gaussian_rasterization.fused import fused_activations
from gvcommon.general_utils import build_rotation
from scene.gaussian_model import GaussianModel

# Render rule (occ_t, floor): a sample is empty where aw <= occ_t, else S / max(aw, floor).
# Each model stores its own in <model>/render_rule; training renders with occ_t = 0.
INTERIOR_RULE = (0.002, 0.01)
SURFACE_RULE = (0.01, 0.01)
_rule = (0.0, 0.01)
_morton_order = None
_morton_state = (0, 0)  # (gaussian count, renders since refresh)


def init_rasterizer(occ_t=0.0, floor=0.01):
    """Render rule for the next renders: empty where aw <= occ_t, else S / max(aw, floor)."""
    global _rule
    _rule = (float(occ_t), float(floor))


def model_render_rule(model_dir, surface):
    """A model's stored render rule; models from before render_rule existed used the old
    hard (surface) or soft (interior) cutoff."""
    path = os.path.join(model_dir, "render_rule")
    if os.path.exists(path):
        occ_t, floor = (float(x) for x in open(path).read().split()[:2])
        return occ_t, floor
    return (0.01, 0.01) if surface else (0.0, 0.01)


def build_bvh(samples):
    """Make `samples` the points every render evaluates at."""
    _C.build_bvh(samples)


def build_bvh_cached(samples, key):
    """build_bvh, reusing the BVH built for batch `key` (full-batch runs only)."""
    _C.build_bvh_cached(samples, int(key))


def sample_bvh_prim_ids():
    """primIDs of the sample BVH last built (int32, leaf order)."""
    return _C.sample_bvh_prim_ids()


def morton3d_unit(xyz: torch.Tensor) -> torch.Tensor:
    """Morton codes for points in [0, 1]^3."""
    norm = (xyz * ((1 << 21) - 1)).long()

    def part1by2(n):
        n = n & 0x1FFFFF
        n = (n | (n << 32)) & 0x1F00000000FFFF
        n = (n | (n << 16)) & 0x1F0000FF0000FF
        n = (n | (n << 8)) & 0x100F00F00F00F00F
        n = (n | (n << 4)) & 0x10C30C30C30C30C3
        n = (n | (n << 2)) & 0x1249249249249249
        return n

    return (
        part1by2(norm[:, 0]) | (part1by2(norm[:, 1]) << 1) | (part1by2(norm[:, 2]) << 2)
    )


def render(pc: GaussianModel):
    """{"cells", "weights"} at the current samples."""
    global _morton_order, _morton_state
    means3D = pc.get_xyz
    scales, rotations, values, weights = fused_activations(
        pc._scaling, pc._rotation, pc._values, pc._weight, pc.max_scale
    )
    # Morton order is a memory-locality hint: refresh it every 100 renders or on a count change.
    n = means3D.shape[0]
    prev_n, age = _morton_state
    if _morton_order is None or prev_n != n or age >= 100:
        _morton_order = torch.argsort(morton3d_unit(means3D))
        _morton_state = (n, 0)
    else:
        _morton_state = (n, age + 1)
    order = _morton_order
    cells, cell_weights = rasterize_gaussians(
        means3D[order],
        scales[order],
        rotations[order],
        values[order],
        weights[order],
        _rule,
    )
    return {"cells": cells, "weights": cell_weights}


def conic6(pc: GaussianModel):
    """The rasterizer's conic (inverse covariance with its epsilon) as a differentiable (P, 6)
    tensor (c00, c01, c02, c11, c12, c22), in float64 with a floored determinant."""
    s = pc.get_scaling.double()
    R = build_rotation(pc._rotation).double()
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
        val, gmu, gc, gw = _C.margin_fp(
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


def margin_set_exterior(samples, key):
    """Register (and build a BVH for) exterior sample set `key` for margin_fp."""
    _C.margin_set_exterior(samples.contiguous(), int(key))


def margin_fp(pc: GaussianModel, key, T):
    """Sum over the exterior samples of set `key` reached by a Gaussian's support of qmax - q."""
    return _MarginFP.apply(pc.get_xyz, conic6(pc), pc.get_weight, T, key)
