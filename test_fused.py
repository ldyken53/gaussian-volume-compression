"""Equivalence check: fused_ops.cu / multi_adam.cu against the torch paths they replace.
Run with PYTHONPATH pointing at a rast_* build. Prints max relative errors."""
import itertools
import torch
from bvh_diff_gaussian_rasterization import _C
from bvh_diff_gaussian_rasterization.fused import fused_activations, fused_volume_loss

torch.manual_seed(0)
dev = "cuda"


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


# ---- activations ----
N, M = 20000, 0.03
rs = (torch.randn(N, 3, device=dev) * 1.5 - 5).requires_grad_()
rq = torch.randn(N, 4, device=dev).requires_grad_()
rv = torch.randn(N, 1, device=dev).requires_grad_()
rw = torch.randn(N, 1, device=dev).requires_grad_()
gs, gq, gv, gw = (torch.randn(N, 3, device=dev), torch.randn(N, 4, device=dev),
                  torch.randn(N, 1, device=dev), torch.randn(N, 1, device=dev))


def torch_act(rs, rq, rv, rw):
    s = torch.exp(rs)
    r = torch.linalg.norm(s, dim=1, keepdim=True) + 1e-8
    s = s * (M * torch.tanh(r / M) / r)
    return s, torch.nn.functional.normalize(rq), torch.sigmoid(rv), torch.sigmoid(rw)


outs_t = torch_act(rs, rq, rv, rw)
gr_t = torch.autograd.grad(sum((o * g).sum() for o, g in zip(outs_t, (gs, gq, gv, gw))), (rs, rq, rv, rw))
outs_f = fused_activations(rs, rq, rv, rw, M)
gr_f = torch.autograd.grad(sum((o * g).sum() for o, g in zip(outs_f, (gs, gq, gv, gw))), (rs, rq, rv, rw))
worst = 0.0
for name, a, b in zip(["scale", "rot", "value", "weight"], outs_f, outs_t):
    worst = max(worst, rel(a, b))
for name, a, b in zip(["d_scale", "d_rot", "d_value", "d_weight"], gr_f, gr_t):
    worst = max(worst, rel(a, b))
print(f"activations: max rel err fwd+bwd {worst:.2e}")
assert worst < 1e-4

# ---- loss ----
S = 300000
c = torch.rand(S, device=dev)
g = torch.rand(S, device=dev)
w = torch.rand(S, device=dev) * 0.05
g[torch.rand(S, device=dev) < 0.05] = -1.0
dead = torch.rand(S, device=dev) < 0.03
c[dead] = -1.0
w[dead] = 0.0
w[:100] = 0.011  # exactly at the hinge threshold: clamp passes the gradient
w[100:200] = 0.0105


def torch_loss(c, g, w, l1, masked, fn_mode, tau, fn_reg, fp_reg, fp_t=0.0, fp_mode="lin", norm_all=False, rel_pow=0.0):
    if masked:
        m = torch.logical_and(g != -1, c != -1)
        den = m.sum().clamp_min(1)
        lv = (((c - g).abs() if l1 else (c - g) ** 2) * m).sum() / den
    else:
        lv = ((c - g).abs() if l1 else (c - g) ** 2).mean()
    fm = torch.logical_and(g != -1, w > 0.0)
    if fn_mode == "log":
        fv = torch.clamp(torch.log(tau / (w + 0.01 * tau)), min=0) * fm
    else:
        fv = torch.clamp(tau - w, min=0) * fm
    hs = lv.detach().clamp_min(0) ** rel_pow if rel_pow > 0 else 1.0
    sc = (fn_reg * lv.detach() if fn_mode == "rel" else fn_reg) * hs
    fn = sc * fv.sum() / (((g != -1).sum() if norm_all else (fv > 0).sum()).float() + 1e-8)
    if fp_reg > 0:
        pm = torch.logical_and(g == -1, w > 0.0)
        if fp_mode == "hinge":
            pv = torch.clamp(w - fp_t, min=0) * pm
            pm = pv > 0
        else:
            pv = (torch.clamp(torch.log(torch.clamp_min(w, 1e-30) / fp_t), min=0) if fp_mode == "log" else w) * pm
        fp = hs * fp_reg * pv.sum() / (((g == -1).sum() if norm_all else pm.sum()).float() + 1e-8)
    else:
        fp = torch.zeros((), device=dev)
    return lv + fn + fp, lv, fn, fp


worst_f = worst_b = 0.0
for l1, masked, fn_mode, fp_reg, (fp_mode, fp_t) in itertools.product([0, 1], [0, 1], ["abs", "rel", "log"], [0.0, 0.5],
                                                                     [("lin", 0.0), ("log", 1e-4), ("hinge", 0.02)]):
    if fp_reg == 0 and fp_mode != "lin": continue
    for norm_all, rel_pow in ([(False, 0.0), (True, 0.0), (True, 0.5)] if fp_mode == "hinge" else [(False, 0.0)]):
        tau = 0.002 if fn_mode == "log" else 0.011
        fn_reg = 0.02 if fn_mode != "rel" else 3.0
        ct, wt = c.clone().requires_grad_(), w.clone().requires_grad_()
        ref = torch_loss(ct, g, wt, l1, masked, fn_mode, tau, fn_reg, fp_reg, fp_t, fp_mode, norm_all, rel_pow)
        dct, dwt = torch.autograd.grad(ref[0], (ct, wt))
        cf, wf = c.clone().requires_grad_(), w.clone().requires_grad_()
        out = fused_volume_loss(cf, g, wf, l1=l1, masked=masked, fn_mode=fn_mode, fn_thresh=tau,
                                fn_reg=fn_reg, fp_reg=fp_reg, fp_t=fp_t, fp_mode=fp_mode, norm_all=norm_all, rel_pow=rel_pow)
        dcf, dwf = torch.autograd.grad(out[0], (cf, wf))
        ef = max(rel(out[i], ref[i].reshape(())) if float(ref[i]) != 0 else float(out[i].abs()) for i in range(4))
        eb = max(rel(dcf, dct), rel(dwf, dwt))
        worst_f, worst_b = max(worst_f, ef), max(worst_b, eb)
        print(f"loss l1={l1} masked={masked} fn={fn_mode:3s} fp={fp_reg} {fp_mode} norm_all={norm_all} rel_pow={rel_pow}: fwd {ef:.1e} bwd {eb:.1e}")
print(f"loss: max rel err fwd {worst_f:.2e} bwd {worst_b:.2e}")
assert worst_f < 1e-4 and worst_b < 1e-4

# ---- multi-tensor Adam (+ fused EMA) vs torch fused Adam ----
shapes = [(5000, 3), (5000, 1), (5000, 3), (5000, 4), (5000, 1)]
lrs = [1.6e-4, 0.025, 0.005, 0.001, 0.0025]
pa = [torch.randn(s, device=dev).requires_grad_() for s in shapes]
pb = [p.detach().clone().requires_grad_() for p in pa]
oa = torch.optim.Adam([{"params": [p], "lr": lr} for p, lr in zip(pa, lrs)], eps=1e-15, fused=True)
ob = torch.optim.Adam([{"params": [p], "lr": lr} for p, lr in zip(pb, lrs)], eps=1e-15, fused=True)
ema_a = [p.detach().clone() for p in pa]
ema_b = [p.detach().clone() for p in pb]
for t in range(1, 51):
    grads = [torch.randn(s, device=dev) for s in shapes]
    for p, gr in zip(pa, grads):
        p.grad = gr.clone()
    for p, gr in zip(pb, grads):
        p.grad = gr.clone()
    oa.step()
    torch._foreach_lerp_(ema_a, [p.detach() for p in pa], 0.01)
    if t == 1:
        ob.step()
        torch._foreach_lerp_(ema_b, [p.detach() for p in pb], 0.01)
    else:
        st = [ob.state[p] for p in pb]
        _C.multi_adam_ema_([p.data for p in pb], [p.grad for p in pb], [s["exp_avg"] for s in st],
                           [s["exp_avg_sq"] for s in st], lrs, t, 0.9, 0.999, 1e-15, ema_b, 0.01)
        torch._foreach_add_([s["step"] for s in st], 1.0)
ea = max(rel(a, b) for a, b in zip(pb, pa))
ee = max(rel(a, b) for a, b in zip(ema_b, ema_a))
print(f"multi_adam_ema_: 50 steps, params rel err {ea:.2e}, EMA rel err {ee:.2e}")
assert ea < 1e-5 and ee < 1e-5
print("ALL OK")
