import argparse
import os
import time

import numpy as np
import pyvista as pv
import torch

from gaussian_renderer import render
from scene import GaussianModel


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
        cells = render(gaussians, jit, cc)["cells"]
        out.append(float(((cells - gt) ** 2).double().mean()))
    return -10.0 * np.log10(max(sum(out) / len(out), 1e-20)), out


def evaluate(gaussians, mesh, gt, block):
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
                rp = render(gaussians, jitter, block)
                cells, wts = rp["cells"], rp["weights"]

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


def main():
    p = argparse.ArgumentParser(
        description="PSNR of a trained model against the full volume at native resolution"
    )
    p.add_argument("-s", "--source_path", required=True)
    p.add_argument("-m", "--model_path", nargs="+", required=True)
    p.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="-1 picks the highest saved iteration in each model dir",
    )
    p.add_argument(
        "--max_scale",
        type=float,
        default=None,
        help="must match training: get_scaling caps on it (default: the model's "
        "max_scale file, else 0.02)",
    )
    p.add_argument(
        "--is_scaled",
        action="store_true",
        help="volume already in the unit cube with values in [0,1]",
    )
    p.add_argument(
        "--block",
        type=int,
        default=256,
        help="render this many cells per axis at a time",
    )
    p.add_argument(
        "--continuous",
        type=int,
        default=0,
        metavar="TRIALS",
        help="also report PSNR at random continuous positions against the trilinear "
        "interpolant, averaged over TRIALS jitters of a --sample_grid lattice",
    )
    p.add_argument(
        "--skip_native",
        action="store_true",
        help="skip the blocked native-grid sweep; report --continuous only",
    )
    p.add_argument(
        "--sample_grid",
        type=int,
        default=128,
        help="lattice resolution for --continuous (128 = train.py's)",
    )
    args = p.parse_args()
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

    for mp in args.model_path:
        it = args.iteration
        if it == -1:
            saved = [
                int(f.split("_")[-1])
                for f in os.listdir(os.path.join(mp, "point_cloud"))
            ]
            it = max(saved)
        gaussians = GaussianModel()
        ms_file = os.path.join(mp, "max_scale")
        gaussians.max_scale = (
            args.max_scale
            if args.max_scale is not None
            else float(open(ms_file).read()) if os.path.exists(ms_file) else 0.02
        )
        # The model's render rule; models without render_rule predate it: (0, 0.01).
        rr = os.path.join(mp, "render_rule")
        gaussians.occ_t, gaussians.floor = (
            (float(x) for x in open(rr).read().split()[:2]) if os.path.exists(rr) else (0.0, 0.01)
        )
        with torch.no_grad():
            gaussians.load_ply(
                os.path.join(mp, "point_cloud", f"iteration_{it}", "point_cloud.ply"),
                mesh,
            )
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
                else evaluate(gaussians, mesh, gt, args.block)
            )
        cont = ""
        if args.continuous:
            with torch.no_grad():
                cp, trials = continuous_psnr(
                    gaussians, mesh, args.sample_grid, args.continuous
                )
            cont = " psnr_continuous=%.3f" % cp
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
            + f" max_scale={gaussians.max_scale:.4f} occ_t={gaussians.occ_t:g}",
            flush=True,
        )
        del gaussians
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
