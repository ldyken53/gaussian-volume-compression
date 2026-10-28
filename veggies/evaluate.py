"""Evaluating trained models against the volume they were fit to.

Meshes: every model is scored on the same fresh points inside the mesh (or --eval_set's):
  psnr_gt    undefined (-1) samples count as 0, so false negatives are penalised
  psnr_both  false negatives excluded
  fn         points inside the mesh that are undefined
Surface models also report fp_pct: the share of points just outside the surface that are defined.
One JSON line per model.

Grids: every model is sampled at every native grid node (psnr, psnr_covered, covered = the share
of defined nodes, ...), and with --continuous T also at T jittered lattices against the trilinear
interpolant (psnr_continuous, the metric of record for structured models). One RESULT line per
model.

--out writes every model's results as a JSON list.
"""

import json
import math
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import numpy as np
import pyvista as pv
import torch

from veggies import files, sampling, volume
from veggies.model import VeggiesModel
from veggies.samples import sample_exterior_points, sample_mesh_points

N_MESH_POINTS = 1_000_000


def model_spec(spec, max_scale=None):
    """(path, max_scale) of a model given as path[:max_scale]; else --max_scale, else its file."""
    path, _, ms = spec.partition(":")
    if ms:
        return path, float(ms)
    return path, max_scale if max_scale is not None else files.read_max_scale(path)


def probe(mesh, points):
    probed = pv.PolyData(points).sample(mesh)
    gt = np.asarray(probed[mesh.array_names[0]]).astype(np.float64).copy()
    gt[~probed["vtkValidPointMask"].astype(bool)] = -1
    return gt


def psnr(sq_sum, n):
    return -10 * math.log10(sq_sum / max(n, 1) + 1e-16)


def evaluate_mesh(args):
    mesh, _ = volume.normalize_mesh(
        volume.select_array(pv.read(args.source_path), args.array)
    )
    print(f"mesh ready: {mesh.n_points} pts", flush=True)

    N = N_MESH_POINTS
    torch.manual_seed(0)
    if args.eval_set:
        interior = np.load(args.eval_set + "_samples.npy")
        gt_in = np.load(args.eval_set + "_gt.npy").astype(np.float64)
    else:
        interior = sample_mesh_points(mesh, 1, N).reshape(N, 3).cpu().numpy()
        gt_in = probe(mesh, interior)
    paths = [model_spec(x)[0] for x in args.models]
    surface = {model_spec(x)[0] for x in args.surface}
    surface |= {p for p in paths if files.read_info(p).get("kind") == "surface"}
    if surface:
        exterior = sample_exterior_points(mesh, 1, N).reshape(N, 3).cpu().numpy()
        gt_out = probe(mesh, exterior)
    else:
        exterior = np.zeros((0, 3), dtype=np.float32)
        gt_out = np.zeros((0,), dtype=np.float64)

    gt_in = torch.tensor(gt_in, device="cuda")
    gt_out = torch.tensor(gt_out, device="cuda")
    interior = torch.tensor(interior, dtype=torch.float, device="cuda")
    exterior = torch.tensor(exterior, dtype=torch.float, device="cuda")

    results = []
    for spec in args.models:
        path, ms = model_spec(spec, args.max_scale)
        ply = files.checkpoint(path, args.iteration)
        if ply is None:
            print(f"SKIP {path}: no checkpoint")
            continue
        model = VeggiesModel()
        model.max_scale = ms
        model.load_ply(ply)
        rule = files.read_render_rule(path, sampling.legacy_rule(path in surface))
        sampling.set_points(interior)
        values = sampling.sample_points(model, rule)[0].double()
        inside = gt_in != -1
        as_zero = torch.where(values == -1, torch.zeros_like(values), values)
        p_gt = psnr(
            float(((as_zero[inside] - gt_in[inside]) ** 2).sum()), int(inside.sum())
        )
        both = torch.logical_and(inside, values != -1)
        p_both = psnr(float(((values[both] - gt_in[both]) ** 2).sum()), int(both.sum()))
        fn = int(torch.count_nonzero(torch.logical_and(values == -1, inside)))
        r = dict(
            model=path,
            max_scale=ms,
            checkpoint=ply.split("point_cloud/")[1].split("/")[0],
            gaussians=model.count,
            psnr_gt=round(p_gt, 3),
            psnr_both=round(p_both, 3),
            fn=fn,
            fn_pct=round(100 * fn / int(inside.sum()), 4),
            eval_points=int(inside.sum()),
        )
        if path in surface:
            sampling.set_points(exterior)
            out_values = sampling.sample_points(model, rule)[0]
            outside = gt_out == -1
            fp = int(torch.count_nonzero(torch.logical_and(out_values != -1, outside)))
            r.update(
                fp_exterior=fp,
                exterior_points=int(outside.sum()),
                fp_pct=round(100 * fp / int(outside.sum()), 3),
            )
        print(json.dumps(r), flush=True)
        results.append(r)
    return results


def grid_values(grid):
    """The grid's values as [z, y, x], the rasterizer's cell order."""
    nx, ny, nz = grid.dimensions
    return grid.point_data[grid.point_data.keys()[0]].reshape(nz, ny, nx)


def continuous_psnr(model, grid, rule, cells, trials, seed=0):
    """PSNR at fresh stratified-random positions against the trilinear interpolant."""
    lo = np.array(grid.bounds[0::2])
    hi = np.array(grid.bounds[1::2])
    step = (hi - lo) / (cells - 1)
    axes = [np.linspace(lo[i], hi[i], cells) for i in range(3)]
    base = np.stack(np.meshgrid(axes[2], axes[1], axes[0], indexing="ij"), -1)[
        ..., ::-1
    ]  # [z, y, x] of (x, y, z)
    base = np.ascontiguousarray(base).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    model.bounds_min, model.bounds_max = list(lo), list(hi)
    mses = []
    for _ in range(trials):
        pos = np.clip(base + rng.uniform(-0.5, 0.5, base.shape) * step[None, :], lo, hi)
        gt = pv.PolyData(pos).sample(grid).point_data[grid.point_data.keys()[0]]
        gt = torch.tensor(
            np.ascontiguousarray(gt.reshape(cells, cells, cells)), dtype=torch.float
        ).cuda()
        jitter = torch.tensor((pos - base).ravel(), dtype=torch.float, device="cuda")
        values = sampling.sample_grid(model, jitter, cells, rule)[0]
        mses.append(float(((values - gt) ** 2).double().mean()))
    return -10.0 * np.log10(max(sum(mses) / len(mses), 1e-20)), mses


def native_grid_errors(model, grid, gt, rule, block):
    nx, ny, nz = grid.dimensions
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    # The rasterizer reproduces this spacing: (max - min) / (count - 1) per axis.
    d = ((xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1), (zmax - zmin) / (nz - 1))
    origin = (xmin, ymin, zmin)
    jitter = torch.zeros(3 * block**3, dtype=torch.float, device="cuda")
    zero = lambda dtype: torch.zeros((), dtype=dtype, device="cuda")  # noqa: E731
    sse, sse_def, sse_bg = zero(torch.float64), zero(torch.float64), zero(torch.float64)
    n_def, n_fn, n_fp, n_bg = (zero(torch.long) for _ in range(4))
    for a0 in range(0, nz, block):  # render axes: a = z, b = y, c = x
        for b0 in range(0, ny, block):
            for c0 in range(0, nx, block):
                lo = (
                    origin[0] + c0 * d[0],
                    origin[1] + b0 * d[1],
                    origin[2] + a0 * d[2],
                )
                model.bounds_min = list(lo)
                model.bounds_max = [lo[i] + (block - 1) * d[i] for i in range(3)]
                values, _ = sampling.sample_grid(model, jitter, block, rule)
                la, lb, lc = (
                    min(block, nz - a0),
                    min(block, ny - b0),
                    min(block, nx - c0),
                )
                values = values[:la, :lb, :lc]
                g = torch.as_tensor(
                    np.ascontiguousarray(
                        gt[a0 : a0 + la, b0 : b0 + lb, c0 : c0 + lc], dtype=np.float32
                    )
                ).cuda()
                sse += ((values - g) ** 2).double().sum()
                defined = values != -1.0
                sse_def += ((values[defined] - g[defined]) ** 2).double().sum()
                n_def += defined.sum()
                n_fn += ((~defined) & (g != 0.0)).sum()
                n_fp += (defined & (g == 0.0)).sum()
                # Values are min-max normalized, so 0 is the exact background.
                bg = g == 0.0
                sse_bg += (values[bg] ** 2).double().sum()
                n_bg += bg.sum()
    n = float(nx) * ny * nz
    to_psnr = lambda mse: float(-10.0 * np.log10(max(mse, 1e-20)))  # noqa: E731
    return {
        "cells": int(n),
        "psnr": to_psnr(float(sse) / n),
        "psnr_covered": to_psnr(float(sse_def) / max(int(n_def), 1)),
        "covered_frac": int(n_def) / n,
        "fn_frac": int(n_fn) / n,
        "fp_frac": int(n_fp) / n,
        "bg_frac": int(n_bg) / n,
        "psnr_bg": to_psnr(float(sse_bg) / max(int(n_bg), 1)),
        "psnr_fg": to_psnr((float(sse) - float(sse_bg)) / max(n - int(n_bg), 1)),
        "sse_bg_share": float(sse_bg) / max(float(sse), 1e-30),
    }


def evaluate_grid(args):
    t0 = time.time()
    grid = volume.select_array(pv.read(args.source_path), args.array)
    if not args.is_scaled:
        volume.normalize_grid(grid, grid.point_data.keys()[0])
    gt = grid_values(grid)
    print(
        f"volume {grid.dimensions} bounds {grid.bounds} loaded in {time.time() - t0:.1f}s",
        flush=True,
    )

    results = []
    for spec in args.models:
        path, ms = model_spec(spec, args.max_scale)
        ply = files.checkpoint(path, args.iteration)
        if ply is None:
            print(f"SKIP {path}: no checkpoint")
            continue
        it = int(ply.split("iteration_")[1].split("/")[0])
        model = VeggiesModel()
        model.max_scale = ms
        rule = files.read_render_rule(path, sampling.legacy_rule(False))
        with torch.no_grad():
            model.load_ply(ply, grid)
            t = time.time()
            if args.skip_native:
                keys = (
                    "psnr",
                    "psnr_covered",
                    "covered_frac",
                    "fn_frac",
                    "fp_frac",
                    "bg_frac",
                    "psnr_bg",
                    "psnr_fg",
                    "sse_bg_share",
                )
                r = {"cells": 0, **dict.fromkeys(keys, 0.0)}
            else:
                r = native_grid_errors(model, grid, gt, rule, args.block)
        cont = ""
        if args.continuous:
            with torch.no_grad():
                cp, _ = continuous_psnr(
                    model, grid, rule, args.sample_grid, args.continuous
                )
            cont = f" psnr_continuous={cp:.3f}"
            r["psnr_continuous"] = round(cp, 3)
        n = model.count
        print(
            f"RESULT model={path} iter={it} gaussians={n} "
            f"ratio={r['cells'] / (12.0 * n):.1f} "
            f"psnr={r['psnr']:.3f} psnr_covered={r['psnr_covered']:.3f} "
            f"covered={r['covered_frac']:.4f} fp={r['fp_frac']:.4f} "
            f"bg={r['bg_frac']:.4f} psnr_bg={r['psnr_bg']:.3f} psnr_fg={r['psnr_fg']:.3f} "
            f"bg_err_share={r['sse_bg_share']:.3f} "
            f"eval_s={time.time() - t:.1f}"
            + cont
            + f" max_scale={model.max_scale:.4f} occ_t={rule[0]:g}",
            flush=True,
        )
        results.append(dict(model=path, iteration=it, gaussians=n, max_scale=ms, **r))
        del model
        torch.cuda.empty_cache()
    return results


def main(argv=None):
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
    p.add_argument("--array", help="the scalar array (default: the first)")
    p.add_argument("--out", default="", help="write every model's results as JSON")
    p.add_argument(
        "--protocol",
        choices=["auto", "mesh", "grid"],
        default="auto",
        help="auto: grid for a grid volume",
    )
    p.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="checkpoint to evaluate; -1: each model's highest",
    )
    p.add_argument(
        "--max_scale",
        type=float,
        default=None,
        help="override every model's max_scale file",
    )
    g = p.add_argument_group("Meshes")
    g.add_argument(
        "--surface",
        nargs="*",
        default=[],
        help="surface models saved without veggies.json",
    )
    g.add_argument(
        "--eval_set",
        default="",
        help="precomputed evaluation points (tools/precompute_tet.py)",
    )
    g = p.add_argument_group("Grids")
    g.add_argument(
        "--is_scaled",
        action="store_true",
        help="already in the unit cube, values in [0, 1]",
    )
    g.add_argument(
        "--block", type=int, default=256, help="nodes per axis sampled at a time"
    )
    g.add_argument(
        "--continuous",
        type=int,
        default=0,
        metavar="TRIALS",
        help="also report psnr_continuous over TRIALS jittered lattices",
    )
    g.add_argument(
        "--skip_native", action="store_true", help="report --continuous only"
    )
    g.add_argument(
        "--sample_grid",
        type=int,
        default=128,
        help="lattice resolution for --continuous",
    )
    args = p.parse_args(argv)
    grid = args.protocol == "grid" or (
        args.protocol == "auto" and volume.is_grid_file(args.source_path)
    )
    results = evaluate_grid(args) if grid else evaluate_mesh(args)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
