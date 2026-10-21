import os

import numpy as np
import pyvista as pv
import torch
from diff_gaussian_rasterization import _C as _dgr_C
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from utils.general_utils import build_rotation, get_expon_lr_func, inverse_sigmoid
from utils.graphics_utils import BasicPointCloud

# Values are clamped this far inside (0, 1) before inverse_sigmoid: an exact 0 or 1 would map
# to -inf/+inf, where the sigmoid has no gradient.
VALUE_EPS = 1e-6
INIT_WEIGHT = 0.01


class GaussianModel:
    # Parameter attributes, in the order the EMA buffers use.
    EMA_NAMES = ("_xyz", "_scaling", "_rotation", "_weight", "_values")
    _GROUP_ATTR = {
        "xyz": "_xyz",
        "weight": "_weight",
        "scaling": "_scaling",
        "rotation": "_rotation",
        "value": "_values",
    }

    def __init__(self):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._weight = torch.empty(0)
        self._values = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.mesh = None
        self.max_scale = 0.02
        # Render rule: a cell is empty where aw <= occ_t, else S / max(aw, floor). Training uses
        # occ_t = 0; a trained model's rule is in its render_rule file.
        self.occ_t = 0.0
        self.floor = 0.01
        self.ema_buffers = None
        self._adam_t = None

    def _apply_cap(self, s):
        """Soft norm cap: ||s|| -> max_scale * tanh(||s|| / max_scale)."""
        r = torch.linalg.norm(s, dim=1, keepdim=True) + 1e-8
        return s * (self.max_scale * torch.tanh(r / self.max_scale) / r)

    @property
    def get_scaling(self):
        return self._apply_cap(torch.exp(self._scaling))

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_weight(self):
        return torch.sigmoid(self._weight)

    @property
    def get_values(self):
        return torch.sigmoid(self._values)

    def _set_bounds(self, mesh):
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds if mesh else (0, 1, 0, 1, 0, 1)
        self.mins = [xmin, ymin, zmin]
        self.maxes = [xmax, ymax, zmax]

    def create_from_pcd(self, pcd: BasicPointCloud, mesh: pv.PolyData):
        """Initial Gaussians at the sampled nodes, sized by their 3-NN distance."""
        points = torch.tensor(np.asarray(pcd.points)).float().cuda()
        self._set_bounds(mesh)
        print(f"Number of points at initialisation: {points.shape[0]}")
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((points.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        weights = inverse_sigmoid(
            INIT_WEIGHT
            * torch.ones((points.shape[0], 1), dtype=torch.float, device="cuda")
        )
        values = inverse_sigmoid(
            torch.clamp(
                torch.tensor(pcd.values, dtype=torch.float, device="cuda"),
                VALUE_EPS,
                1.0 - VALUE_EPS,
            )
        )
        self._xyz = nn.Parameter(points.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._weight = nn.Parameter(weights.requires_grad_(True))
        self._values = nn.Parameter(values.requires_grad_(True))
        self.mesh = mesh

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        groups = [
            ("xyz", self._xyz, training_args.position_lr_init),
            ("weight", self._weight, training_args.weight_lr),
            ("scaling", self._scaling, training_args.scaling_lr),
            ("rotation", self._rotation, training_args.rotation_lr),
            ("value", self._values, training_args.values_lr),
        ]
        # torch owns the Adam state and takes the first step; optimizer_step does the rest.
        self.optimizer = torch.optim.Adam(
            [{"params": [p], "lr": lr, "name": name} for name, p, lr in groups],
            lr=0.0,
            eps=1e-15,
            fused=True,
        )
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def optimizer_step(self, ema=None, ema_w=0.0):
        """One Adam step over all five parameters in a single kernel, plus the EMA if given."""
        opt = self.optimizer
        params = [g["params"][0] for g in opt.param_groups]
        has_grad = [p.grad is not None for p in params]
        ema_done = False
        if not any(has_grad):
            pass  # densification replaced every parameter this iteration: no Adam step
        elif self._adam_t is None:
            opt.step()
            self._adam_t = int(opt.state[params[0]]["step"])
        else:
            assert all(has_grad), "partial Adam step"
            self._adam_t += 1
            states = [opt.state[p] for p in params]
            g0 = opt.param_groups[0]
            emas = (
                [ema[self._GROUP_ATTR[g["name"]]] for g in opt.param_groups]
                if ema
                else []
            )
            _dgr_C.multi_adam_(
                [p.data for p in params],
                [p.grad for p in params],
                [st["exp_avg"] for st in states],
                [st["exp_avg_sq"] for st in states],
                [float(g["lr"]) for g in opt.param_groups],
                self._adam_t,
                float(g0["betas"][0]),
                float(g0["betas"][1]),
                float(g0["eps"]),
                emas,
                float(ema_w),
            )
            ema_done = ema is not None
        if ema is not None and not ema_done:
            torch._foreach_lerp_(
                [ema[n] for n in self.EMA_NAMES],
                [getattr(self, n).detach() for n in self.EMA_NAMES],
                ema_w,
            )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def save_ply(self, path):
        """Raw (pre-activation) parameters, 12 floats per Gaussian."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        attributes = ["x", "y", "z", "value", "weight"]
        attributes += [f"scale_{i}" for i in range(self._scaling.shape[1])]
        attributes += [f"rot_{i}" for i in range(self._rotation.shape[1])]
        data = np.concatenate(
            [
                t.detach().cpu().numpy()
                for t in (
                    self._xyz,
                    self._values,
                    self._weight,
                    self._scaling,
                    self._rotation,
                )
            ],
            axis=1,
        )
        elements = np.empty(data.shape[0], dtype=[(a, "f4") for a in attributes])
        elements[:] = list(map(tuple, data))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_ply(self, path, mesh=None):
        el = PlyData.read(path).elements[0]
        print(f"Number of points at initialisation : {el['x'].shape[0]}")
        self._set_bounds(mesh)

        def cols(prefix):
            names = sorted(
                (p.name for p in el.properties if p.name.startswith(prefix)),
                key=lambda x: int(x.split("_")[-1]),
            )
            return np.stack([np.asarray(el[n]) for n in names], axis=1)

        def param(a):
            return nn.Parameter(
                torch.tensor(a, dtype=torch.float, device="cuda").requires_grad_(True)
            )

        self._xyz = param(np.stack([np.asarray(el[c]) for c in "xyz"], axis=1))
        self._weight = param(np.asarray(el["weight"])[..., np.newaxis])
        self._scaling = param(cols("scale_"))
        self._rotation = param(cols("rot"))
        self._values = param(np.asarray(el["value"])[..., np.newaxis])
        self.mesh = mesh

    def _sync_params(self, tensors):
        for name, t in tensors.items():
            setattr(self, self._GROUP_ATTR[name], t)

    def _swap_param(self, group, tensor, state_fn):
        """Replace a group's parameter by tensor, carrying its Adam state through state_fn."""
        old = group["params"][0]
        state = self.optimizer.state.pop(old, None)
        group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
        if state is not None:
            state_fn(state)
            self.optimizer.state[group["params"][0]] = state
        return group["params"][0]

    def replace_tensors_to_optimizer(self, inds):
        """Re-wrap every parameter after in-place edits, zeroing the Adam moments at inds."""

        def reset(st):
            st["exp_avg"][inds] = 0
            st["exp_avg_sq"][inds] = 0

        self._sync_params(
            {
                g["name"]: self._swap_param(
                    g, getattr(self, self._GROUP_ATTR[g["name"]]), reset
                )
                for g in self.optimizer.param_groups
            }
        )

    def _prune_optimizer(self, mask):
        def prune(st):
            st["exp_avg"] = st["exp_avg"][mask]
            st["exp_avg_sq"] = st["exp_avg_sq"][mask]

        return {
            g["name"]: self._swap_param(g, g["params"][0][mask], prune)
            for g in self.optimizer.param_groups
        }

    def cat_tensors_to_optimizer(self, tensors_dict):
        out = {}
        for g in self.optimizer.param_groups:
            ext = tensors_dict[g["name"]]

            def extend(st):
                st["exp_avg"] = torch.cat((st["exp_avg"], torch.zeros_like(ext)), dim=0)
                st["exp_avg_sq"] = torch.cat(
                    (st["exp_avg_sq"], torch.zeros_like(ext)), dim=0
                )

            out[g["name"]] = self._swap_param(
                g, torch.cat((g["params"][0], ext), dim=0), extend
            )
        return out

    # The EMA buffers must see every row edit (prune, append, overwrite), or the saved
    # average mixes unrelated Gaussians.
    def _ema_prune(self, valid_points_mask):
        if self.ema_buffers is None:
            return
        for n in self.EMA_NAMES:
            self.ema_buffers[n] = self.ema_buffers[n][valid_points_mask]

    def _ema_extend(self, n_new):
        if self.ema_buffers is None or n_new <= 0:
            return
        for n in self.EMA_NAMES:
            self.ema_buffers[n] = torch.cat(
                (self.ema_buffers[n], getattr(self, n)[-n_new:].detach().clone()), dim=0
            )

    def _ema_reset(self, idx):
        if self.ema_buffers is None or idx is None or idx.numel() == 0:
            return
        for n in self.EMA_NAMES:
            self.ema_buffers[n][idx] = getattr(self, n)[idx].detach().clone()

    def prune_points(self, mask):
        valid = ~mask
        self._sync_params(self._prune_optimizer(valid))
        self._ema_prune(valid)

    def densification_postfix(
        self, new_xyz, new_weights, new_scaling, new_rotation, new_values
    ):
        d = {
            "xyz": new_xyz,
            "weight": new_weights,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "value": new_values,
        }
        self._sync_params(self.cat_tensors_to_optimizer(d))
        self._ema_extend(new_xyz.shape[0])

    # --use_mcmc: 3DGS-MCMC relocation of dead Gaussians onto live ones.
    def _update_params(self, idxs, ratio):
        N = (ratio[idxs, 0] + 1).float()
        new_weight = self.get_weight[idxs, 0] / N
        new_weight = torch.clamp(
            new_weight.unsqueeze(-1),
            max=1.0 - torch.finfo(torch.float32).eps,
            min=0.005,
        )
        new_weight = inverse_sigmoid(new_weight)
        scaling = self.get_scaling[idxs]
        rotation = build_rotation(self._rotation[idxs])
        noise = torch.randn_like(scaling) * scaling * 0.5
        xyz = self._xyz[idxs] + torch.bmm(rotation, noise.unsqueeze(-1)).squeeze(-1)
        return (
            xyz,
            new_weight,
            self._scaling[idxs],
            self._rotation[idxs],
            self._values[idxs],
        )

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask):
        if dead_mask.sum() == 0:
            return
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
        if alive_indices.shape[0] <= 0:
            return
        probs = self.get_weight[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(
            alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0]
        )
        (
            self._xyz[dead_indices],
            self._weight[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices],
            self._values[dead_indices],
        ) = self._update_params(reinit_idx, ratio=ratio)
        self._weight[reinit_idx] = self._weight[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]
        # Both the sources and the destinations now hold different Gaussians.
        reset_idx = torch.cat([reinit_idx, dead_indices]).unique()
        self._ema_reset(reset_idx)
        self.replace_tensors_to_optimizer(inds=reset_idx)

    def add_new_gs(self, cap_max):
        current = self._weight.shape[0]
        num_gs = max(0, min(cap_max, int(1.5 * current)) - current)
        if num_gs <= 0:
            return 0
        add_idx, ratio = self._sample_alives(
            probs=self.get_weight.squeeze(-1), num=num_gs
        )
        new_xyz, new_weight, new_scaling, new_rotation, new_values = (
            self._update_params(add_idx, ratio=ratio)
        )
        self._weight[add_idx] = new_weight
        self._scaling[add_idx] = new_scaling
        self._ema_reset(add_idx)
        self.densification_postfix(
            new_xyz, new_weight, new_scaling, new_rotation, new_values
        )
        self.replace_tensors_to_optimizer(inds=add_idx)
        return num_gs

    # --densify_3dgs: clone/split on the position gradient.
    def _select_by_grad(self, grads, grad_threshold, large, scene_extent, max_parents):
        grad_mag = torch.zeros(self.get_xyz.shape[0], device=self.get_xyz.device)
        grad_mag[: grads.shape[0]] = torch.linalg.norm(grads, dim=-1)
        big = (
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent
        )
        mask = (grad_mag >= grad_threshold) & (big if large else ~big)
        idx = mask.nonzero(as_tuple=True)[0]
        if max_parents is not None:
            if max_parents <= 0 or idx.numel() == 0:
                return None
            if idx.numel() > max_parents:
                idx = idx[
                    torch.topk(grad_mag[idx], k=max_parents, sorted=False).indices
                ]
            mask = torch.zeros_like(mask)
            mask[idx] = True
        return mask

    def densify_and_split(
        self, grads, grad_threshold, scene_extent, N=2, max_new_points=None
    ):
        mask = self._select_by_grad(
            grads,
            grad_threshold,
            True,
            scene_extent,
            None if max_new_points is None else int(max_new_points) // N,
        )
        if mask is None or N * mask.sum().item() == 0:
            return 0
        num_added = N * mask.sum().item()
        stds = self.get_scaling[mask].repeat(N, 1)
        samples = torch.normal(
            mean=torch.zeros((stds.size(0), 3), device=self.get_xyz.device), std=stds
        )
        rots = build_rotation(self._rotation[mask]).repeat(N, 1, 1)
        self.densification_postfix(
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[mask].repeat(N, 1),
            self._weight[mask].repeat(N, 1),
            torch.log(self.get_scaling[mask].repeat(N, 1) / (0.8 * N)),
            self._rotation[mask].repeat(N, 1),
            self._values[mask].repeat(N, 1),
        )
        self.prune_points(
            torch.cat(
                (
                    mask,
                    torch.zeros(
                        N * mask.sum(), device=self.get_xyz.device, dtype=torch.bool
                    ),
                )
            )
        )
        return num_added

    def densify_and_clone(
        self, grads, grad_threshold, scene_extent, max_new_points=None
    ):
        mask = self._select_by_grad(
            grads,
            grad_threshold,
            False,
            scene_extent,
            None if max_new_points is None else int(max_new_points),
        )
        if mask is None or mask.sum().item() == 0:
            return 0
        num_added = mask.sum().item()
        self.densification_postfix(
            self._xyz[mask],
            self._weight[mask],
            self._scaling[mask],
            self._rotation[mask],
            self._values[mask],
        )
        return num_added

    # Default densification: new Gaussians at high-error cells.
    def _new_point_knn(self, new_pts):
        """distCUDA2 for the new points only, against existing + new points.

        Bins all points on a grid of cell size R and searches each query's 27 neighbouring
        cells; a query whose 3 nearest all lie within R has the exact answer. Misses retry
        at 2R, then fall back to the full pass, as do large batches.
        """
        old = self._xyz.detach()
        n_old, k = old.shape[0], new_pts.shape[0]

        def full():
            return distCUDA2(torch.cat([old, new_pts], dim=0))[n_old:]

        if k == 0 or 8 * k > n_old:
            return full()
        dev = old.device
        pts = torch.cat([old, new_pts], dim=0)
        P = pts.shape[0]
        lo = pts.min(dim=0).values
        ext = (pts.max(dim=0).values - lo).clamp_min(1e-12)
        spacing = float((torch.prod(ext) / P) ** (1.0 / 3.0))
        offs = torch.stack(
            torch.meshgrid(*[torch.arange(-1, 2, device=dev)] * 3, indexing="ij"), -1
        ).reshape(-1, 3)
        out = torch.empty(k, device=dev)
        todo = torch.arange(k, device=dev)
        for R in (2.0 * spacing, 4.0 * spacing):
            G = (ext / R).floor().long() + 1
            c = ((pts - lo) / R).floor().long().clamp_min(0).minimum(G - 1)
            lin = (c[:, 2] * G[1] + c[:, 1]) * G[0] + c[:, 0]
            order = torch.argsort(lin)
            counts = torch.bincount(lin, minlength=int(torch.prod(G)))
            starts = torch.cumsum(counts, 0) - counts
            # Chunked so the (chunk, 27, M) candidate tensors stay under ~6e7 entries.
            chunk = max(1024, int(6e7 // (27 * max(1, int(counts.max())))))
            unresolved = []
            for ch in torch.split(todo, chunk):
                raw = c[n_old + ch][:, None, :] + offs[None]
                # Drop out-of-grid neighbours; clamping would count a boundary cell twice.
                inb = ((raw >= 0) & (raw < G)).all(-1)
                qc = raw.clamp_min(0).minimum(G - 1)
                nb = (qc[..., 2] * G[1] + qc[..., 1]) * G[0] + qc[..., 0]
                cnt = counts[nb] * inb
                M = int(cnt.max())
                if M == 0:
                    unresolved.append(ch)
                    continue
                ar = torch.arange(M, device=dev)
                slot = (starts[nb][..., None] + ar).clamp(max=P - 1)
                idx = order[slot].reshape(ch.shape[0], -1)
                valid = (ar < cnt[..., None]).reshape(ch.shape[0], -1)
                valid &= idx != (n_old + ch)[:, None]
                d2 = ((pts[idx] - new_pts[ch][:, None, :]) ** 2).sum(-1)
                d2 = d2.masked_fill(~valid, float("inf"))
                if d2.shape[1] < 3:
                    unresolved.append(ch)
                    continue
                best = torch.topk(d2, 3, dim=1, largest=False, sorted=True).values
                ok = best[:, 2] <= R * R
                out[ch[ok]] = (best[ok, 0] + best[ok, 1] + best[ok, 2]) / 3.0
                unresolved.append(ch[~ok])
            todo = torch.cat(unresolved)
            if todo.numel() == 0:
                return out
        out[todo] = full()[todo]
        return out

    def _new_gaussians(self, empty_points, empty_values, new_weight):
        dist2 = torch.clamp_min(self._new_point_knn(empty_points) * 0.1, 0.0000001)
        n = empty_points.shape[0]
        rot = torch.zeros((n, 4), device="cuda")
        rot[:, 0] = 1
        return {
            "xyz": empty_points,
            "weight": inverse_sigmoid(
                new_weight * torch.ones((n, 1), dtype=torch.float, device="cuda")
            ),
            "scaling": torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3),
            "rotation": rot,
            "value": inverse_sigmoid(
                torch.clamp(empty_values, VALUE_EPS, 1.0 - VALUE_EPS)
            ),
        }

    def _write_rows(self, idx, d):
        """Overwrite rows idx in place, zeroing their Adam moments and restarting their EMA.
        Clears all grads, which skips this iteration's Adam step as a prune/append would.
        """
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                p = group["params"][0]
                p.data[idx] = d[group["name"]].to(p.dtype)
                st = self.optimizer.state.get(p, None)
                if st:
                    st["exp_avg"][idx] = 0.0
                    st["exp_avg_sq"][idx] = 0.0
                p.grad = None
        self._ema_reset(idx)

    def densify_in_place(self, empty_points, empty_values, new_weight, min_weight):
        """New Gaussians take the rows of pruned ones; only the overflow is appended."""
        d = self._new_gaussians(empty_points, empty_values, new_weight)
        k = empty_points.shape[0]
        slots = torch.nonzero((self.get_weight <= min_weight).squeeze(-1)).squeeze(-1)
        m = min(k, slots.numel())
        if m > 0:
            self._write_rows(slots[:m], {n: v[:m] for n, v in d.items()})
        if k > m:
            self.densification_postfix(
                *(d[n][m:] for n in ("xyz", "weight", "scaling", "rotation", "value"))
            )
        elif slots.numel() > m:
            mask = torch.zeros(
                self._xyz.shape[0], dtype=torch.bool, device=self._xyz.device
            )
            mask[slots[m:]] = True
            self.prune_points(mask)

    def densify_and_prune(
        self,
        max_grad,
        min_weight,
        new_weight,
        empty_points,
        empty_values,
        prune_only=False,
        num_densify=None,
        use_3dgs=False,
    ):
        if not prune_only:
            if not use_3dgs:
                self.densify_in_place(
                    empty_points, empty_values, new_weight, min_weight
                )
                return
            # Called before the Adam step, so .grad is this iteration's.
            if self._xyz.grad is not None:
                grads = self._xyz.grad.detach().clone()
                remaining = num_densify
                added = self.densify_and_clone(
                    grads, max_grad, 1.0, max_new_points=remaining
                )
                if remaining is not None:
                    remaining -= added
                if remaining is None or remaining > 0:
                    self.densify_and_split(
                        grads, max_grad, 1.0, max_new_points=remaining
                    )
        self.prune_points((self.get_weight < min_weight).squeeze())
