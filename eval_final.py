"""Held-out evaluation of trained models against one mesh.

Every model is scored on the same fresh interior points (gt -1 outside the mesh is ignored):
  psnr_gt    the model's empty (-1) cells count as 0, so false negatives are penalised
  psnr_both  false negatives are excluded
Surface models (--surface) also report FP%: the share of points just outside the surface that
render as occupied.
"""

import glob
import json
import math
from argparse import ArgumentParser

import numpy as np
import pyvista as pv
import torch

from gaussian_renderer import build_bvh, init_rasterizer, model_render_rule, render
from gvcommon.model_files import read_max_scale
from sampling import sample_exterior_points, sample_mesh_points
from scene import GaussianModel

N = 1_000_000


def probe(mesh, pts):
    probed = pv.PolyData(pts).sample(mesh)
    gt = np.asarray(probed[mesh.array_names[0]]).astype(np.float64).copy()
    gt[~probed["vtkValidPointMask"].astype(bool)] = -1
    return gt


def psnr(sq_sum, n):
    return -10 * math.log10(sq_sum / max(n, 1) + 1e-16)


def model_spec(spec):
    """path[:max_scale]; without one, the model's own max_scale file (else 0.02)."""
    mp, _, ms = spec.partition(":")
    return mp, float(ms) if ms else read_max_scale(mp)


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("-s", "--source_path", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument(
        "--surface", nargs="*", default=[], help="models trained with --encode_surface"
    )
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--eval_set",
        default="",
        help="precomputed tet-interior set (see precompute_tet.py)",
    )
    args = ap.parse_args()

    mesh = pv.read(args.source_path)
    mesh = mesh.cell_data_to_point_data()
    vals = mesh.get_array(mesh.array_names[0]).reshape(-1, 1)
    vmin, vmax = vals.min(), vals.max()
    mesh.get_array(mesh.array_names[0])[:] = ((vals - vmin) / (vmax - vmin)).ravel()
    gmin, gmax = mesh.points.min(), mesh.points.max()
    mesh.translate(np.array([-gmin] * 3), inplace=True)
    mesh.scale(1.0 / (gmax - gmin), inplace=True)
    print(f"mesh ready: {mesh.n_points} pts", flush=True)

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
        mp, ms = model_spec(spec)
        cands = sorted(
            glob.glob(f"{mp}/point_cloud/iteration_*/point_cloud.ply"),
            key=lambda x: int(x.split("iteration_")[1].split("/")[0]),
        )
        cands = [c for c in cands if "iteration_0/" not in c]
        if not cands:
            print(f"SKIP {mp}: no checkpoint")
            continue
        ply = cands[-1]
        g = GaussianModel()
        g.max_scale = ms
        g.load_ply(ply)
        init_rasterizer(*model_render_rule(mp, mp in surface))
        build_bvh(int_t)
        cells = render(g)["cells"].double()
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
            ce = render(g)["cells"]
            out = gt_e_t == -1
            fp = int(torch.count_nonzero(torch.logical_and(ce != -1, out)))
            r.update(
                fp_exterior=fp,
                exterior_points=int(out.sum()),
                fp_pct=round(100 * fp / int(out.sum()), 3),
            )
        print(json.dumps(r), flush=True)
        results.append(r)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
