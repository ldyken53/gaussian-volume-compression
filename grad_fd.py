#!/usr/bin/env python3
"""Finite-difference check of the struct rasterizer's gradients on a trained model.

  grad_fd.py <volume.vtk> <model_dir> [n_per_param=24]     env FD_LOSS=aw, FD_H_SCALE=<x>

L = sum over covered cells of (cell - t)^2 + sum over cells of aw * r2, in float64, with t and r2
fixed random fields on the 128^3 lattice (any smooth scalar of the outputs checks the adjoint).
Raw parameters (what Adam updates) are perturbed in place; autograd's dL/d(raw) is compared to
central differences for, per parameter type, the 12 largest-gradient Gaussians and the 12 with
the largest gradient among the most anisotropic decile. Runs under whatever SUPPORT_T /
DENOM_FLOOR / OCC_T / OCC_KEEP_AW the environment sets (the model's raster_env if present).
"""
import os, sys
from types import SimpleNamespace
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
vol, mdir = sys.argv[1], sys.argv[2]
n_per = int(sys.argv[3]) if len(sys.argv) > 3 else 24
ef = os.path.join(mdir, "raster_env")
if os.path.exists(ef):
    for line in open(ef).read().split():
        k, v = line.split("=", 1); os.environ.setdefault(k, v)


def vtk_bounds(path):
    """Bounds of a legacy structured-points .vtk from its text header."""
    dims = origin = spacing = None
    with open(path, "rb") as f:
        for _ in range(16):
            tok = f.readline().decode("ascii", "replace").split()
            if not tok: continue
            k = tok[0].upper()
            if k == "DIMENSIONS": dims = [int(v) for v in tok[1:4]]
            elif k == "ORIGIN": origin = [float(v) for v in tok[1:4]]
            elif k in ("SPACING", "ASPECT_RATIO"): spacing = [float(v) for v in tok[1:4]]
    return origin, [origin[i] + (dims[i] - 1) * spacing[i] for i in range(3)]

from gaussian_renderer import render
from scene import GaussianModel
torch.manual_seed(0)
lo, hi = vtk_bounds(vol)
g = GaussianModel(); g.max_scale = float(open(os.path.join(mdir, "max_scale")).read())
it = max(int(d.split("_")[-1]) for d in os.listdir(os.path.join(mdir, "point_cloud")))
g.load_ply(os.path.join(mdir, "point_cloud", f"iteration_{it}", "point_cloud.ply"),
           SimpleNamespace(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])))
names = {"xyz": "_xyz", "scaling": "_scaling", "rotation": "_rotation", "value": "_values", "weight": "_weight"}
for n in names.values(): getattr(g, n).requires_grad_(True)
g.mins, g.maxes = list(lo), list(hi)
cc = 128
gen = torch.Generator(device="cuda").manual_seed(1)
step = torch.tensor([(hi[i] - lo[i]) / (cc - 1) for i in range(3)], device="cuda")
jit = ((torch.rand(cc ** 3, 3, device="cuda", generator=gen) - 0.5) * step).ravel().contiguous()
t = torch.rand(cc, cc, cc, device="cuda", generator=gen).double()
r2 = torch.randn(cc, cc, cc, device="cuda", generator=gen).double()
pipe = SimpleNamespace(debug=False)
FD_LOSS = os.environ.get("FD_LOSS", "full")   # "aw": only sum(aw * r2) -- continuous across OCC_T under OCC_KEEP_AW
def L_of(o):
    c, w = o["cells"].double(), o["weights"].double()
    if FD_LOSS == "aw":
        return (w * r2).sum()
    return (((c - t) ** 2) * (c != -1)).sum() + (w * r2).sum()
with torch.no_grad():
    La = float(L_of(render(g, pipe, jit, cc))); Lb = float(L_of(render(g, pipe, jit, cc)))
noise = abs(La - Lb)
L = L_of(render(g, pipe, jit, cc))
params = {k: getattr(g, v) for k, v in names.items()}
grads = dict(zip(params, torch.autograd.grad(L, list(params.values()))))
with torch.no_grad():
    s = g.get_scaling; aniso = s.max(1).values / s.min(1).values
env = {k: os.environ.get(k) for k in ("SUPPORT_T", "DENOM_FLOOR", "OCC_T", "OCC_KEEP_AW")}
print(f"{os.path.basename(mdir)}: P={g._xyz.shape[0]:,} L={La:.6e} repeat-render noise {noise:.2e} | "
      f"axis ratio median {aniso.median():.2f} p99 {aniso.quantile(0.99):.1f} max {aniso.max():.1f} | env {env}")
_hk = float(os.environ.get("FD_H_SCALE", "1"))
hs = {k: v * _hk for k, v in {"xyz": 1e-4, "scaling": 1e-3, "rotation": 1e-3, "value": 1e-3, "weight": 1e-3}.items()}
for name, p in params.items():
    with torch.no_grad():
        gm = grads[name].abs().sum(1)
        top = gm.argsort(descending=True)[: n_per // 2]
        gan = torch.where(aniso >= aniso.quantile(0.9), gm, torch.zeros_like(gm))
        top_an = gan.argsort(descending=True)[: n_per // 2]
    for grp, idx in (("top", top), ("aniso", top_an)):
        an, fd = [], []
        for i in idx.tolist():
            for c in range(p.shape[1]):
                h = hs[name]
                with torch.no_grad():
                    old = p[i, c].item()
                    p[i, c] = old + h; Lp = float(L_of(render(g, pipe, jit, cc)))
                    p[i, c] = old - h; Lm = float(L_of(render(g, pipe, jit, cc)))
                    p[i, c] = old
                fd.append((Lp - Lm) / (2 * h)); an.append(float(grads[name][i, c]))
        a, f = np.array(an), np.array(fd)
        sig = np.abs(f) > 20 * noise / (2 * hs[name]) + 1e-12
        cos = float(a @ f / (np.linalg.norm(a) * np.linalg.norm(f) + 1e-30))
        ratio = np.median(a[sig] / f[sig]) if sig.any() else float("nan")
        rel = np.median(np.abs(a[sig] - f[sig]) / np.abs(f[sig])) if sig.any() else float("nan")
        within = float(np.mean(np.abs(a[sig] - f[sig]) <= 0.05 * np.abs(f[sig]))) if sig.any() else float("nan")
        print(f"  {name:8s} {grp:5s}: cos {cos:+.4f}  median autograd/FD {ratio:+.3f}  median rel err {rel:.3f}  within5% {within:.2f}  "
              f"({int(sig.sum())}/{len(a)} above noise; axis ratio of these: max {float(aniso[idx].max()):.1f})", flush=True)
