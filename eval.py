"""Evaluate trained models against the volume they were fit to (any model type).

Mesh models (a .vtu source): every model is scored on the same fresh points inside the mesh
(or --eval_set's), gt -1 outside it ignored:
  psnr_gt    empty (-1) renders count as 0, so false negatives are penalised
  psnr_both  false negatives are excluded
  fn         points inside the mesh that render empty
Surface models (--surface) also report fp_pct: the share of points just outside the surface
that render occupied. Prints one JSON line per model.

Grid models (a .vtk source): every model is rendered over the full native grid (psnr,
psnr_covered, covered = the share of non-empty cells, ...), and with --continuous T also at T
jittered lattices against the trilinear interpolant (psnr_continuous, the structured metric
of record). Prints one RESULT line per model.

--out writes every model's results as a JSON list. Each model renders under its own
max_scale and render_rule files.
"""

import glob
import json
import math
import os
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import numpy as np
import pyvista as pv
import torch

from gaussian_renderer import (
    build_bvh,
    init_rasterizer,
    model_render_rule,
    render_grid,
    render_points,
    render_rule,
)
from scene import GaussianModel
from scene.model_files import read_max_scale
from scene.samples import sample_exterior_points, sample_mesh_points

N_MESH_POINTS = 1_000_000


def model_spec(spec, max_scale=None):
    """(path, max_scale) of a model given as path[:max_scale]; without one, --max_scale, else
    the model's own max_scale file (else 0.02)."""
    mp, _, ms = spec.partition(":")
    if ms:
        return mp, float(ms)
    return mp, max_scale if max_scale is not None else read_max_scale(mp)


def checkpoint(mp, iteration):
    """The model's ply at `iteration` (-1: its highest saved one past 0), or None."""
    if iteration != -1:
        path = os.path.join(
            mp, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"
        )
        return path if os.path.exists(path) else None
    cands = sorted(
        glob.glob(f"{mp}/point_cloud/iteration_*/point_cloud.ply"),
        key=lambda x: int(x.split("iteration_")[1].split("/")[0]),
    )
    cands = [c for c in cands if "iteration_0/" not in c]
    return cands[-1] if cands else None


# ---- Mesh models: held-out points ----


def probe(mesh, pts):
    probed = pv.PolyData(pts).sample(mesh)
    gt = np.asarray(probed[mesh.array_names[0]]).astype(np.float64).copy()
    gt[~probed["vtkValidPointMask"].astype(bool)] = -1
    return gt


def psnr(sq_sum, n):
    return -10 * math.log10(sq_sum / max(n, 1) + 1e-16)


def eval_mesh(args):
    mesh = pv.read(args.source_path)
    mesh = mesh.cell_data_to_point_data()
    vals = mesh.get_array(mesh.array_names[0]).reshape(-1, 1)
    vmin, vmax = vals.min(), vals.max()
    mesh.get_array(mesh.array_names[0])[:] = ((vals - vmin) / (vmax - vmin)).ravel()
    gmin, gmax = mesh.points.min(), mesh.points.max()
    mesh.translate(np.array([-gmin] * 3), inplace=True)
    mesh.scale(1.0 / (gmax - gmin), inplace=True)
    print(f"mesh ready: {mesh.n_points} pts", flush=True)

    N = N_MESH_POINTS
    torch.manual_seed(0)
    if args.eval_set:
        interior = np.load(args.eval_set + "_samples.npy")
        gt_i = np.load(args.eval_set + "_gt.npy").astype(np.float64)
    else:
        interior = sample_mesh_points(mesh, 1, N).reshape(N, 3).cpu().numpy()
        gt_i = probe(mesh, interior)
    surface = {model_spec(x)[0] for x in args.surface}
    if surface:
        exterior = sample_exterior_points(mesh, 1, N).reshape(N, 3).cpu().numpy()
        gt_e = probe(mesh, exterior)
    else:
        exterior = np.zeros((0, 3), dtype=np.float32)
        gt_e = np.zeros((0,), dtype=np.float64)

    gt_i_t = torch.tensor(gt_i, device="cuda")
    gt_e_t = torch.tensor(gt_e, device="cuda")
    int_t = torch.tensor(interior, dtype=torch.float, device="cuda")
    ext_t = torch.tensor(exterior, dtype=torch.float, device="cuda")

    results = []
    for spec in args.models:
        mp, ms = model_spec(spec, args.max_scale)
        ply = checkpoint(mp, args.iteration)
        if ply is None:
            print(f"SKIP {mp}: no checkpoint")
            continue
        g = GaussianModel()
        g.max_scale = ms
        g.load_ply(ply)
        init_rasterizer(*model_render_rule(mp, mp in surface))
        build_bvh(int_t)
        cells = render_points(g)[0].double()
        m_gt = gt_i_t != -1
        c0 = torch.where(cells == -1, torch.zeros_like(cells), cells)
        p_gt = psnr(float(((c0[m_gt] - gt_i_t[m_gt]) ** 2).sum()), int(m_gt.sum()))
        m_both = torch.logical_and(m_gt, cells != -1)
        p_both = psnr(
            float(((cells[m_both] - gt_i_t[m_both]) ** 2).sum()), int(m_both.sum())
        )
        fn = int(torch.count_nonzero(torch.logical_and(cells == -1, m_gt)))
        r = dict(
            model=mp,
            max_scale=ms,
            checkpoint=ply.split("point_cloud/")[1].split("/")[0],
            gaussians=int(g.get_xyz.shape[0]),
            psnr_gt=round(p_gt, 3),
            psnr_both=round(p_both, 3),
            fn=fn,
            fn_pct=round(100 * fn / int(m_gt.sum()), 4),
            eval_points=int(m_gt.sum()),
        )
        if mp in surface:
            build_bvh(ext_t)
            ce = render_points(g)[0]
            out = gt_e_t == -1
            fp = int(torch.count_nonzero(torch.logical_and(ce != -1, out)))
            r.update(
                fp_exterior=fp,
                exterior_points=int(out.sum()),
                fp_pct=round(100 * fp / int(out.sum()), 3),
            )
        print(json.dumps(r), flush=True)
        results.append(r)
    return results


# ---- Grid models: the full native grid, and continuous positions ----


def volume_gt(mesh):
    """The volume as [z, y, x], the renderer's cell order: no resampling needed."""
    nx, ny, nz = mesh.dimensions
    return mesh.point_data[mesh.point_data.keys()[0]].reshape(nz, ny, nx)


def continuous_psnr(gaussians, mesh, cc, trials, seed=0):
    """PSNR at fresh stratified-random positions against the trilinear interpolant."""
    lo = np.array(mesh.bounds[0::2])
    hi = np.array(mesh.bounds[1::2])
    step = (hi - lo) / (cc - 1)
    ax = [np.linspace(lo[i], hi[i], cc) for i in range(3)]
    # Renderer cell order is [z,y,x]; each entry holds that cell's (x,y,z).
    base = np.stack(np.meshgrid(ax[2], ax[1], ax[0], indexing="ij"), -1)[..., ::-1]
    base = np.ascontiguousarray(base).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    gaussians.mins = list(lo)
    gaussians.maxes = list(hi)
    out = []
    for _ in range(trials):
        pos = np.clip(base + rng.uniform(-0.5, 0.5, base.shape) * step[None, :], lo, hi)
        gt = pv.PolyData(pos).sample(mesh).point_data[mesh.point_data.keys()[0]]
        gt = torch.tensor(
            np.ascontiguousarray(gt.reshape(cc, cc, cc)), dtype=torch.float
        ).cuda()
        jit = torch.tensor((pos - base).ravel(), dtype=torch.float, device="cuda")
        cells = render_grid(gaussians, jit, cc)[0]
        out.append(float(((cells - gt) ** 2).double().mean()))
    return -10.0 * np.log10(max(sum(out) / len(out), 1e-20)), out


def evaluate_grid(gaussians, mesh, gt, block):
    nx, ny, nz = mesh.dimensions
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    # Cell spacing the rasterizer will reproduce: it uses (max-min)/(count-1) per axis.
    d = ((xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1), (zmax - zmin) / (nz - 1))
    origin = (xmin, ymin, zmin)

    jitter = torch.zeros(3 * block**3, dtype=torch.float, device="cuda")
    sse = torch.zeros((), dtype=torch.float64, device="cuda")
    sse_cov = torch.zeros((), dtype=torch.float64, device="cuda")
    n_cov = torch.zeros((), dtype=torch.long, device="cuda")
    n_fn = torch.zeros((), dtype=torch.long, device="cuda")
    n_fp = torch.zeros((), dtype=torch.long, device="cuda")
    # Values are min-max normalised, so g == 0 is the exact background.
    sse_bg = torch.zeros((), dtype=torch.float64, device="cuda")
    n_bg = torch.zeros((), dtype=torch.long, device="cuda")

    # Axis a of the render output is z, b is y, c is x.
    for a0 in range(0, nz, block):
        for b0 in range(0, ny, block):
            for c0 in range(0, nx, block):
                # Edge blocks render full-size and are sliced; cells are independent.
                lo = (
                    origin[0] + c0 * d[0],
                    origin[1] + b0 * d[1],
                    origin[2] + a0 * d[2],
                )
                hi = tuple(lo[i] + (block - 1) * d[i] for i in range(3))
                gaussians.mins = list(lo)
                gaussians.maxes = list(hi)
                cells, wts = render_grid(gaussians, jitter, block)

                la, lb, lc = (
                    min(block, nz - a0),
                    min(block, ny - b0),
                    min(block, nx - c0),
                )
                cells = cells[:la, :lb, :lc]
                wts = wts[:la, :lb, :lc]
                g = torch.as_tensor(
                    np.ascontiguousarray(
                        gt[a0 : a0 + la, b0 : b0 + lb, c0 : c0 + lc], dtype=np.float32
                    )
                ).cuda()

                sse += ((cells - g) ** 2).double().sum()
                covered = cells != -1.0  # occupied under the render rule (aw > occ_t)
                sse_cov += ((cells[covered] - g[covered]) ** 2).double().sum()
                n_cov += covered.sum()
                n_fn += ((~covered) & (g != 0.0)).sum()
                n_fp += (covered & (g == 0.0)).sum()
                bg = g == 0.0
                sse_bg += ((cells[bg]) ** 2).double().sum()
                n_bg += bg.sum()

    n = float(nx) * ny * nz
    psnr = lambda mse: float(-10.0 * np.log10(max(mse, 1e-20)))
    return {
        "cells": int(n),
        "psnr": psnr(float(sse) / n),
        "psnr_covered": psnr(float(sse_cov) / max(int(n_cov), 1)),
        "covered_frac": int(n_cov) / n,
        "fn_frac": int(n_fn) / n,
        "fp_frac": int(n_fp) / n,
        "bg_frac": int(n_bg) / n,
        "psnr_bg": psnr(float(sse_bg) / max(int(n_bg), 1)),
        "psnr_fg": psnr((float(sse) - float(sse_bg)) / max(n - int(n_bg), 1)),
        "sse_bg_share": float(sse_bg) / max(float(sse), 1e-30),
    }


def eval_grid(args):
    t0 = time.time()
    mesh = pv.read(args.source_path)
    if not args.is_scaled:
        name = mesh.point_data.keys()[0]
        v = mesh.get_array(name)
        v[:] = ((v - v.min()) / (v.max() - v.min())).ravel()
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        lo, hi = min(xmin, ymin, zmin), max(xmax, ymax, zmax)
        mesh.translate(np.array([-lo, -lo, -lo]), inplace=True)
        mesh.scale(1.0 / (hi - lo), inplace=True)
    gt = volume_gt(mesh)
    print(
        f"volume {mesh.dimensions} bounds {mesh.bounds} loaded in {time.time() - t0:.1f}s",
        flush=True,
    )

    results = []
    for spec in args.models:
        mp, ms = model_spec(spec, args.max_scale)
        ply = checkpoint(mp, args.iteration)
        if ply is None:
            print(f"SKIP {mp}: no checkpoint")
            continue
        it = int(ply.split("iteration_")[1].split("/")[0])
        gaussians = GaussianModel()
        gaussians.max_scale = ms
        # The model's render rule; models without render_rule predate it: (0, 0.01).
        init_rasterizer(*model_render_rule(mp, False))
        with torch.no_grad():
            gaussians.load_ply(ply, mesh)
            t = time.time()
            r = (
                dict(
                    cells=0,
                    psnr=0.0,
                    psnr_covered=0.0,
                    covered_frac=0.0,
                    fn_frac=0.0,
                    fp_frac=0.0,
                    bg_frac=0.0,
                    psnr_bg=0.0,
                    psnr_fg=0.0,
                    sse_bg_share=0.0,
                )
                if args.skip_native
                else evaluate_grid(gaussians, mesh, gt, args.block)
            )
        cont = ""
        if args.continuous:
            with torch.no_grad():
                cp, trials = continuous_psnr(
                    gaussians, mesh, args.sample_grid, args.continuous
                )
            cont = " psnr_continuous=%.3f" % cp
            r["psnr_continuous"] = round(cp, 3)
        ng = gaussians.get_values.shape[0]
        print(
            f"RESULT model={mp} iter={it} gaussians={ng} "
            f"ratio={r['cells'] / (12.0 * ng):.1f} "
            f"psnr={r['psnr']:.3f} psnr_covered={r['psnr_covered']:.3f} "
            f"covered={r['covered_frac']:.4f} fp={r['fp_frac']:.4f} "
            f"bg={r['bg_frac']:.4f} psnr_bg={r['psnr_bg']:.3f} psnr_fg={r['psnr_fg']:.3f} "
            f"bg_err_share={r['sse_bg_share']:.3f} "
            f"eval_s={time.time() - t:.1f}"
            + cont
            + f" max_scale={gaussians.max_scale:.4f} occ_t={render_rule()[0]:g}",
            flush=True,
        )
        results.append(dict(model=mp, iteration=it, gaussians=ng, max_scale=ms, **r))
        del gaussians
        torch.cuda.empty_cache()
    return results


def main():
    p = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    p.add_argument("-s", "--source_path", required=True)
    p.add_argument(
        "-m",
        "--models",
        "--model_path",
        dest="models",
        nargs="+",
        required=True,
        help="model directories, each optionally path:max_scale",
    )
    p.add_argument("--out", default="", help="write every model's results as JSON")
    p.add_argument(
        "--protocol",
        choices=["auto", "mesh", "grid"],
        default="auto",
        help="auto: grid for a .vtk source, mesh for a .vtu",
    )
    p.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="checkpoint to evaluate; -1 picks each model's highest",
    )
    p.add_argument(
        "--max_scale",
        type=float,
        default=None,
        help="override every model's max_scale file (rendering needs the training value)",
    )
    g = p.add_argument_group("Mesh models")
    g.add_argument(
        "--surface", nargs="*", default=[], help="models trained with --encode_surface"
    )
    g.add_argument(
        "--eval_set",
        default="",
        help="precomputed tet-interior set (see precompute_tet.py)",
    )
    g = p.add_argument_group("Grid models")
    g.add_argument(
        "--is_scaled",
        action="store_true",
        help="volume already in the unit cube with values in [0,1]",
    )
    g.add_argument(
        "--block",
        type=int,
        default=256,
        help="render this many cells per axis at a time",
    )
    g.add_argument(
        "--continuous",
        type=int,
        default=0,
        metavar="TRIALS",
        help="also report PSNR at random continuous positions against the trilinear "
        "interpolant, averaged over TRIALS jitters of a --sample_grid lattice",
    )
    g.add_argument(
        "--skip_native",
        action="store_true",
        help="skip the blocked native-grid sweep; report --continuous only",
    )
    g.add_argument(
        "--sample_grid",
        type=int,
        default=128,
        help="lattice resolution for --continuous (128 = train.py's)",
    )
    args = p.parse_args()
    grid = args.protocol == "grid" or (
        args.protocol == "auto" and args.source_path.lower().endswith(".vtk")
    )
    results = eval_grid(args) if grid else eval_mesh(args)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
