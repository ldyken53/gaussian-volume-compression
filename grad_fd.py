"""Finite-difference check of the full rasterizer gradient on a real model.

usage: python grad_fd.py <model.ply> <samples.npy> <gt.npy> [max_scale] [n_per_param]
Renders one batch, L = sum over live samples of (cell - gt)^2 in float64, and compares
autograd's dL/d(raw parameter) against central differences for a sample of Gaussians,
including the most anisotropic ones. Raw parameters go through the torch getters.
"""
import os
import sys
import numpy as np
import torch
from plyfile import PlyData

os.environ.setdefault("FUSED", "0")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import init_rasterizer, render, build_bvh


class _Pipe:
    debug = False


def main():
    ply, samp_path, gt_path = sys.argv[1:4]
    max_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
    n_per = int(sys.argv[5]) if len(sys.argv) > 5 else 24
    torch.manual_seed(0)
    v = PlyData.read(ply).elements[0]
    g = GaussianModel()
    g.max_scale = max_scale
    cols = lambda names: torch.tensor(np.stack([np.asarray(v[n]) for n in names], 1), dtype=torch.float, device="cuda")
    g._xyz = torch.nn.Parameter(cols(["x", "y", "z"]))
    g._values = torch.nn.Parameter(cols(["value"]))
    g._weight = torch.nn.Parameter(cols(["weight"]))
    g._scaling = torch.nn.Parameter(cols(["scale_0", "scale_1", "scale_2"]))
    g._rotation = torch.nn.Parameter(cols(["rot_0", "rot_1", "rot_2", "rot_3"]))
    g.mins, g.maxes = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    P = g._xyz.shape[0]

    S = np.load(samp_path, mmap_mode="r")
    G = np.load(gt_path, mmap_mode="r")
    samples = torch.tensor(np.ascontiguousarray(S[0]), dtype=torch.float, device="cuda")
    gt = torch.tensor(np.ascontiguousarray(G[0]), dtype=torch.float64, device="cuda")
    keep = gt != -1
    samples, gt = samples[keep], gt[keep]
    # Morton-sort the samples as train.py does (the BVH build relies on locality).
    q = (samples.clamp(0, 1) * ((1 << 21) - 1)).long()
    def part(n):
        n = n & 0x1fffff
        n = (n | (n << 32)) & 0x1f00000000ffff
        n = (n | (n << 16)) & 0x1f0000ff0000ff
        n = (n | (n << 8)) & 0x100f00f00f00f00f
        n = (n | (n << 4)) & 0x10c30c30c30c30c3
        n = (n | (n << 2)) & 0x1249249249249249
        return n
    o = (part(q[:, 0]) | (part(q[:, 1]) << 1) | (part(q[:, 2]) << 2)).argsort()
    samples, gt = samples[o].contiguous(), gt[o]
    init_rasterizer(g, _Pipe(), 128)
    build_bvh(samples)

    def loss_of(cells):
        live = cells != -1
        return (((cells.double() - gt) ** 2) * live).sum()

    with torch.no_grad():
        L0a = float(loss_of(render(g)["cells"]))
        L0b = float(loss_of(render(g)["cells"]))
    print(f"P={P} samples={samples.shape[0]} L={L0a:.9e} repeat-render noise {abs(L0a - L0b):.2e}")

    L = loss_of(render(g)["cells"])
    params = {"xyz": g._xyz, "scaling": g._scaling, "rotation": g._rotation,
              "value": g._values, "weight": g._weight}
    grads = torch.autograd.grad(L, list(params.values()))
    grads = dict(zip(params, grads))

    with torch.no_grad():
        s = g.get_scaling
        aniso = (s.max(1).values / s.min(1).values)
        print(f"axis ratio median {aniso.median():.2f} p99 {aniso.quantile(0.99):.2f} max {aniso.max():.1f}")
        # Gaussians with a non-negligible gradient: half random, half the most anisotropic.
        gn = sum(grads[k].abs().sum(1) for k in grads)
        cand = torch.nonzero(gn > gn.quantile(0.5)).squeeze(1)
        rnd = cand[torch.randperm(cand.numel(), device="cuda")[: n_per // 2]]
        top = cand[aniso[cand].argsort(descending=True)[: n_per // 2]]
        idx = torch.cat([rnd, top])

    hs = {"xyz": 1e-4, "scaling": 1e-3, "rotation": 1e-3, "value": 1e-3, "weight": 1e-3}
    noise = abs(L0a - L0b)
    for name, p in params.items():
        an, fd, tag = [], [], []
        # This parameter's own largest gradients, so FD has signal: half the top by |grad|
        # among all, half the top among the most anisotropic decile.
        with torch.no_grad():
            gm = grads[name].abs().sum(1)
            top_any = gm.argsort(descending=True)[: n_per // 2]
            an_mask = aniso >= aniso.quantile(0.9)
            gm_an = torch.where(an_mask, gm, torch.zeros_like(gm))
            top_an = gm_an.argsort(descending=True)[: n_per // 2]
            idx = torch.cat([top_any, top_an])
        for k, i in enumerate(idx.tolist()):
            for c in range(p.shape[1]):
                h = hs[name]
                with torch.no_grad():
                    old = p[i, c].item()
                    p[i, c] = old + h
                    Lp = float(loss_of(render(g)["cells"]))
                    p[i, c] = old - h
                    Lm = float(loss_of(render(g)["cells"]))
                    p[i, c] = old
                fd.append((Lp - Lm) / (2 * h))
                an.append(float(grads[name][i, c]))
                tag.append("aniso" if k >= n_per // 2 else "top")
        an, fd = np.array(an), np.array(fd)
        for grp in ("top", "aniso"):
            m = np.array(tag) == grp
            a, f = an[m], fd[m]
            sig = np.abs(f) > 20 * noise / (2 * hs[name]) + 1e-12
            ratio = np.median(a[sig] / f[sig]) if sig.any() else float("nan")
            relerr = np.median(np.abs(a[sig] - f[sig]) / np.abs(f[sig])) if sig.any() else float("nan")
            cos = float(a @ f / (np.linalg.norm(a) * np.linalg.norm(f) + 1e-30))
            print(f"{name:8s} {grp:5s}: cos(autograd, FD) {cos:+.4f}  median autograd/FD {ratio:+.3f}  "
                  f"median rel err {relerr:.3f}  ({int(sig.sum())}/{len(a)} above noise)")


if __name__ == "__main__":
    main()
