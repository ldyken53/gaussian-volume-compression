"""FD check of margin_fp (margin.cu + conic6 autograd) on random Gaussians/exterior points."""
import torch
from scene.gaussian_model import GaussianModel
from gaussian_renderer import margin_set_exterior, margin_fp
torch.manual_seed(0)
P, N, T = 400, 20000, 1e-4
g = GaussianModel(); g.max_scale = 0.08
g._xyz = torch.nn.Parameter(torch.rand(P, 3, device="cuda"))
g._scaling = torch.nn.Parameter(torch.log(0.01 + 0.04 * torch.rand(P, 3, device="cuda")))
g._rotation = torch.nn.Parameter(torch.randn(P, 4, device="cuda"))
g._weight = torch.nn.Parameter(torch.randn(P, 1, device="cuda"))
g._values = torch.nn.Parameter(torch.zeros(P, 1, device="cuda"))
margin_set_exterior(torch.rand(N, 3, device="cuda"), 0)
F = margin_fp(g, 0, T)
params = {"xyz": g._xyz, "scaling": g._scaling, "rotation": g._rotation, "weight": g._weight}
grads = dict(zip(params, torch.autograd.grad(F, list(params.values()))))
print(f"F = {float(F):.4f}")
for name, p in params.items():
    an, fd = [], []
    idx = grads[name].abs().sum(1).argsort(descending=True)[:20]
    for i in idx.tolist():
        for c in range(p.shape[1]):
            h = 1e-3
            with torch.no_grad():
                o = p[i, c].item(); p[i, c] = o + h; Fp = float(margin_fp(g, 0, T))
                p[i, c] = o - h; Fm = float(margin_fp(g, 0, T)); p[i, c] = o
            fd.append((Fp - Fm) / (2 * h)); an.append(float(grads[name][i, c]))
    a, f = torch.tensor(an), torch.tensor(fd)
    cos = float(a @ f / (a.norm() * f.norm() + 1e-30))
    print(f"{name:8s} cos {cos:+.5f}  median autograd/FD {float((a / f)[f.abs() > 1e-3].median()):+.4f}")
