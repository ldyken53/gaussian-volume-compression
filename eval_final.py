"""Final evaluation for unstructured models, per the test-mesh method in
volume_metrics.py. Loads the mesh once and evaluates every model against the same
fresh out-of-sample point sets.

PSNR ignores all points where the GT probe returns -1 (outside the mesh). Two variants
are reported because they treat the model's own -1 predictions differently:
  psnr_gt   : mask gt != -1 only; the model's -1 becomes 0, so false negatives are
              PENALISED. This is "ignoring all points where the GT returns -1".
  psnr_both : mask gt != -1 AND cells != -1; false negatives are excluded entirely.
FP% for surface models uses sample_exterior_points, i.e. the fraction of points drawn
just outside the surface that the model nonetheless renders as occupied.
"""
import sys, json, math, glob, numpy as np, torch, pyvista as pv
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import init_rasterizer, render, build_bvh
from scene import GaussianModel
from utils.loss_utils import create_window
from volume_metrics import sample_mesh_points, sample_exterior_points

N = 1_000_000

def probe(mesh, pts):
    probed = pv.PolyData(pts).sample(mesh)
    gt = np.asarray(probed[mesh.array_names[0]]).astype(np.float64).copy()
    gt[~probed["vtkValidPointMask"].astype(bool)] = -1
    return gt

def psnr(sq_sum, n):
    return -10 * math.log10(sq_sum / max(n, 1) + 1e-16)

def main():
    window = create_window()
    ap = ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--surface", nargs="*", default=[],
                    help="model dirs that were trained with --encode_surface")
    ap.add_argument("--out", default="")
    ap.add_argument("--eval_set", default="",
                    help="Prefix of a precomputed tet-interior eval set "
                         "(<prefix>_samples.npy / _gt.npy).")
    ap.add_argument("--vertex_eval", action="store_true",
                    help="Sample mesh vertices instead of tet interiors (for meshes "
                         "whose triangulation will not fit on the GPU).")
    lp = ModelParams(ap); op = OptimizationParams(ap); pp = PipelineParams(ap)
    args = ap.parse_args(sys.argv[1:])

    mesh = pv.read(args.source_path)
    mesh = mesh.cell_data_to_point_data()
    vals = mesh.get_array(mesh.array_names[0]).reshape(-1, 1)
    vmin, vmax = vals.min(), vals.max()
    mesh.get_array(mesh.array_names[0])[:] = ((vals - vmin) / (vmax - vmin)).ravel()
    gmin, gmax = mesh.points.min(), mesh.points.max()
    mesh.translate(np.array([-gmin]*3), inplace=True)
    mesh.scale(1.0/(gmax-gmin), inplace=True)
    print(f"mesh ready: {mesh.n_points} pts, bounds {[round(b,4) for b in mesh.bounds]}", flush=True)

    torch.manual_seed(0)
    if args.eval_set:
        # Precomputed tet-interior evaluation set (see precompute_tet.py). Same point
        # distribution as sample_mesh_points, but built once on the CPU so meshes whose
        # triangulation will not fit on the GPU can still be scored the standard way.
        interior = np.load(args.eval_set + "_samples.npy")
        gt_i = np.load(args.eval_set + "_gt.npy").astype(np.float64)
        print(f"eval set: {interior.shape[0]} precomputed tet-interior points "
              f"from {args.eval_set}", flush=True)
    elif args.vertex_eval:
        # Fallback for meshes whose triangulation will not fit on the GPU:
        # sample_mesh_points calls mesh.triangulate() and uploads the tet
        # connectivity, which for impactnorm (247M mixed cells) is 8-40 GB.
        # Instead draw random mesh VERTICES and read their values directly -- no
        # triangulation and no probe. Still out-of-sample (the model trained on
        # cell-interior points) but a different point distribution, so PSNR from
        # this path is not strictly comparable to the tet-interior path.
        idx = np.random.default_rng(0).choice(mesh.n_points, size=N, replace=False)
        interior = np.asarray(mesh.points)[idx].astype(np.float32)
        gt_i = np.asarray(mesh.point_data[mesh.array_names[0]])[idx].astype(np.float64)
    else:
        interior = sample_mesh_points(mesh, 1, N).reshape(N, 3).cpu().numpy()
        gt_i = probe(mesh, interior)
    print(f"interior set: {N} pts, {(gt_i==-1).sum()} outside mesh", flush=True)
    if args.surface:
        exterior = sample_exterior_points(mesh, 1, N).reshape(N, 3).cpu().numpy()
        gt_e = probe(mesh, exterior)
        print(f"exterior set: {N} pts, {(gt_e==-1).sum()} outside mesh", flush=True)
    else:
        # No surface models to score, so the exterior/FP set is never read.
        exterior = np.zeros((0, 3), dtype=np.float32)
        gt_e = np.zeros((0,), dtype=np.float64)
        print("exterior set: skipped (no surface models)", flush=True)

    gt_i_t = torch.tensor(gt_i, device="cuda")
    gt_e_t = torch.tensor(gt_e, device="cuda")
    int_t = torch.tensor(interior, dtype=torch.float, device="cuda")
    ext_t = torch.tensor(exterior, dtype=torch.float, device="cuda")

    results = []
    for spec in args.models:
        # "path" or "path:max_scale" -- _apply_cap uses GaussianModel.max_scale, so a
        # model trained with a non-default --max_scale MUST be evaluated with the same
        # value or every Gaussian is silently rescaled (valley surface 1024x was
        # trained at 0.04 and read 34% false negatives when scored at 0.02).
        mp, _, ms = spec.partition(":")
        ms = float(ms) if ms else 0.02
        cands = sorted(glob.glob(f"{mp}/point_cloud/iteration_*/point_cloud.ply"),
                       key=lambda x: int(x.split("iteration_")[1].split("/")[0]))
        cands = [c for c in cands if "iteration_0/" not in c]
        if not cands:
            print(f"SKIP {mp}: no checkpoint"); continue
        ply = cands[-1]
        g = GaussianModel(); g.max_scale = ms; g.load_ply(ply, mesh)
        pp_ = pp.extract(args); pp_.debug = False
        init_rasterizer(g, pp_)
        build_bvh(int_t)
        cells = render(g)["cells"].double()
        m_gt = gt_i_t != -1
        c0 = torch.where(cells == -1, torch.zeros_like(cells), cells)
        p_gt = psnr(float(((c0[m_gt]-gt_i_t[m_gt])**2).sum()), int(m_gt.sum()))
        m_both = torch.logical_and(m_gt, cells != -1)
        p_both = psnr(float(((cells[m_both]-gt_i_t[m_both])**2).sum()), int(m_both.sum()))
        fn = int(torch.count_nonzero(torch.logical_and(cells == -1, m_gt)))
        r = dict(model=mp, max_scale=ms, checkpoint=ply.split("point_cloud/")[1].split("/")[0],
                 gaussians=int(g.get_xyz.shape[0]),
                 psnr_gt=round(p_gt,3), psnr_both=round(p_both,3),
                 fn=fn, fn_pct=round(100*fn/int(m_gt.sum()),4), eval_points=int(m_gt.sum()))
        if mp in [x.partition(":")[0] for x in args.surface]:
            build_bvh(ext_t)
            ce = render(g)["cells"]
            out = gt_e_t == -1
            fp = int(torch.count_nonzero(torch.logical_and(ce != -1, out)))
            r.update(fp_exterior=fp, exterior_points=int(out.sum()),
                     fp_pct=round(100*fp/int(out.sum()),3))
        print(json.dumps(r), flush=True)
        results.append(r)
    if args.out:
        json.dump(results, open(args.out,"w"), indent=2)

main()
