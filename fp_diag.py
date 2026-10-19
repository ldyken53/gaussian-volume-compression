"""Where do a surface model's false positives / negatives sit? Samples points on the mesh
surface, offsets them along the outward normal by a signed distance d in [-D, D], probes the
mesh for inside/outside, renders each model, and reports per distance bin:
  outside points: FP rate (rendered covered) and aw quantiles of the FPs
  inside points:  FN rate (rendered empty) and the median aw of all inside points
usage: python fp_diag.py -s <mesh.vtu> --models dir[:max_scale] ... [--D 0.02]
Use the rasterizer build (hard/soft, OCC_T env) the models were trained with.
"""
import sys, glob, numpy as np, torch, pyvista as pv
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import init_rasterizer, render, build_bvh
from scene import GaussianModel

ap = ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--D", type=float, default=0.02)
ap.add_argument("--n", type=int, default=400000)
ap.add_argument("--bins", type=int, default=8)
lp = ModelParams(ap); op = OptimizationParams(ap); pp = PipelineParams(ap)
args = ap.parse_args(sys.argv[1:])

mesh = pv.read(args.source_path).cell_data_to_point_data()
vals = mesh.get_array(mesh.array_names[0]).reshape(-1, 1)
mesh.get_array(mesh.array_names[0])[:] = ((vals - vals.min()) / (vals.max() - vals.min())).ravel()
gmin, gmax = mesh.points.min(), mesh.points.max()
mesh.translate(np.array([-gmin] * 3), inplace=True)
mesh.scale(1.0 / (gmax - gmin), inplace=True)

surf = mesh.extract_surface().triangulate().compute_normals(cell_normals=True, point_normals=False)
rng = np.random.default_rng(0)
faces = surf.faces.reshape(-1, 4)[:, 1:]
area = surf.compute_cell_sizes(length=False, volume=False)["Area"]
fi = rng.choice(len(faces), size=args.n, p=area / area.sum())   # area-weighted surface points
u = rng.random((args.n, 2)); su = np.sqrt(u[:, 0])
bary = np.stack([1 - su, su * (1 - u[:, 1]), su * u[:, 1]], 1)
P = (bary[:, :, None] * surf.points[faces[fi]]).sum(1)
d = rng.uniform(-args.D, args.D, args.n)
pts = (P + surf.cell_data["Normals"][fi] * d[:, None]).astype(np.float32)
probed = pv.PolyData(pts).sample(mesh)
inside = probed["vtkValidPointMask"].astype(bool)
print(f"{args.n} points within {args.D} of the surface: {inside.mean():.3f} inside", flush=True)

pts_t = torch.tensor(pts, device="cuda")
edges = np.linspace(-args.D, args.D, 2 * args.bins + 1)
b = np.clip(np.digitize(d, edges) - 1, 0, 2 * args.bins - 1)
for spec in args.models:
    mp, _, ms = spec.partition(":")
    g = GaussianModel(); g.max_scale = float(ms) if ms else 0.02
    ply = sorted(glob.glob(f"{mp}/point_cloud/iteration_*/point_cloud.ply"),
                 key=lambda x: int(x.split("iteration_")[1].split("/")[0]))[-1]
    g.load_ply(ply, mesh)
    pp_ = pp.extract(args); pp_.debug = False
    init_rasterizer(g, pp_)
    build_bvh(pts_t)
    out = render(g)
    cov = (out["cells"] != -1).cpu().numpy()
    aw = out["weights"].detach().cpu().numpy()
    print(f"== {mp}  (FP among outside {cov[~inside].mean()*100:.2f}%, FN among inside {(~cov[inside]).mean()*100:.3f}%)")
    fnm = inside & ~cov
    fa = aw[fnm]
    print(f"   FN cells: {fnm.sum()} | aw==0 (unreached): {(fa == 0).sum()} | 0<aw<0.5*theta: {((fa > 0) & (fa < 0.005)).sum()} "
          f"| near theta (>=0.005): {(fa >= 0.005).sum()} | FN by depth: "
          + " ".join(f"{lo:.3f}:{int((fnm & (d >= -hi) & (d < -lo)).sum())}" for lo, hi in ((0, .0025), (.0025, .005), (.005, .01), (.01, 1))))
    print("   d bin [lo,hi)        n_out  FP%     FP aw p10/p50/p90          n_in   FN%    inside aw p50")
    for k in range(2 * args.bins):
        m = b == k
        mo, mi = m & ~inside, m & inside
        fp = cov[mo]; fn = ~cov[mi]
        q = np.percentile(aw[mo][fp], [10, 50, 90]) if fp.sum() > 5 else [np.nan] * 3
        print(f"   [{edges[k]:+.4f},{edges[k+1]:+.4f})  {mo.sum():6d}  {100*fp.mean() if mo.sum() else 0:5.1f}  "
              f"{q[0]:.2e}/{q[1]:.2e}/{q[2]:.2e}   {mi.sum():6d}  {100*fn.mean() if mi.sum() else 0:5.2f}  "
              f"{np.median(aw[mi]) if mi.sum() else np.nan:.3f}", flush=True)
