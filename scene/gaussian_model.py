import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from bvh_diff_gaussian_rasterization import _C as _dgr_C
from gvcommon.general_utils import build_rotation, get_expon_lr_func, inverse_sigmoid

# Values are clamped this far from 0 and 1 before inverse_sigmoid, which is infinite there.
VALUE_EPS = 1e-6
INIT_WEIGHT = 0.1
NEW_WEIGHT = 0.01


class GaussianModel:
    EMA_NAMES = ("_xyz", "_scaling", "_rotation", "_weight", "_values")
    _EMA_ATTR = {
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
        # name -> per-Gaussian Polyak average, installed by train.py; rows follow every mutation.
        self.ema_buffers = None
        self.scaling_activation = torch.exp
        self.inverse_scaling_activation = torch.log
        self.weight_activation = torch.sigmoid
        self.inverse_weight_activation = inverse_sigmoid
        self.values_activation = torch.sigmoid
        self.inverse_value_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def _apply_cap(self, s):
        """Soft cap on the scale norm: max_scale * tanh(|s| / max_scale)."""
        r = torch.linalg.norm(s, dim=1, keepdim=True) + 1e-8
        r_soft = self.max_scale * torch.tanh(r / self.max_scale)
        return s * (r_soft / r)

    @property
    def get_scaling(self):
        return self._apply_cap(self.scaling_activation(self._scaling))

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_weight(self):
        return self.weight_activation(self._weight)

    @property
    def get_values(self):
        return self.values_activation(self._values)

    def create_from_pcd(self, pcd, mesh):
        """One Gaussian per seed point, sized by the mean distance to its 3 nearest seeds."""
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        self.mins = [xmin, ymin, zmin]
        self.maxes = [xmax, ymax, zmax]
        print(f"Number of points at initialisation : {fused_point_cloud.shape[0]}")

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = self.inverse_scaling_activation(torch.sqrt(dist2))[..., None].repeat(
            1, 3
        )
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        weights = self.inverse_weight_activation(
            INIT_WEIGHT
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        values = self.inverse_value_activation(
            torch.clamp(
                torch.tensor(pcd.values, dtype=torch.float, device="cuda"),
                VALUE_EPS,
                1.0 - VALUE_EPS,
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._weight = nn.Parameter(weights.requires_grad_(True))
        self._values = nn.Parameter(values.requires_grad_(True))
        self.mesh = mesh

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        optimizer_params = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init,
                "name": "xyz",
            },
            {"params": [self._weight], "lr": training_args.weight_lr, "name": "weight"},
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
            {"params": [self._values], "lr": training_args.values_lr, "name": "value"},
        ]
        self.optimizer = torch.optim.Adam(
            optimizer_params, lr=0.0, eps=1e-15, fused=True
        )
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def optimizer_step(self, ema=None, ema_w=0.0):
        """One Adam launch over all five tensors, updating the EMA in the same pass if given.

        torch keeps owning the state (densification edits exp_avg/exp_avg_sq in place); its
        own step creates that state and handles any partial step. Returns whether ema was updated.
        """
        opt = self.optimizer
        if getattr(self, "_adam_torch_only", False):
            opt.step()
            return False
        groups = [g for g in opt.param_groups if g["params"][0].grad is not None]
        if not groups:
            # Densification just replaced every parameter; torch's step is a no-op here.
            return False
        states = [opt.state.get(g["params"][0]) for g in groups]
        if (
            len(groups) != len(opt.param_groups)
            or not all(states)
            or getattr(self, "_adam_t", None) is None
        ):
            # Per-group step counts that ever diverge cannot share one bias correction.
            opt.step()
            steps = {
                int(st["step"])
                for st in (opt.state[g["params"][0]] for g in opt.param_groups)
                if st
            }
            self._adam_t = steps.pop() if len(steps) == 1 else None
            self._adam_torch_only = self._adam_t is None
            return False
        ps = [g["params"][0] for g in groups]
        if not all(p.grad.is_contiguous() for p in ps):
            opt.step()
            self._adam_t += 1
            return False
        self._adam_t += 1
        b1, b2 = groups[0]["betas"]
        emas = (
            [ema[self._EMA_ATTR[g["name"]]] for g in groups] if ema is not None else []
        )
        _dgr_C.multi_adam_(
            [p.data for p in ps],
            [p.grad for p in ps],
            [st["exp_avg"] for st in states],
            [st["exp_avg_sq"] for st in states],
            [float(g["lr"]) for g in groups],
            self._adam_t,
            float(b1),
            float(b2),
            float(groups[0]["eps"]),
            emas,
            float(ema_w),
        )
        torch._foreach_add_([st["step"] for st in states], 1.0)
        return ema is not None

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        attributes = ["x", "y", "z", "value", "weight"]
        attributes += [f"scale_{i}" for i in range(self._scaling.shape[1])]
        attributes += [f"rot_{i}" for i in range(self._rotation.shape[1])]
        return attributes

    def save_ply(self, path):
        """Raw (pre-activation) parameters, one vertex per Gaussian."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        weights = self._weight.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        values = self._values.detach().cpu().numpy()
        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, values, weights, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_ply(self, path):
        el = PlyData.read(path).elements[0]

        def cols(names):
            t = np.stack([np.asarray(el[n]) for n in names], axis=1)
            return nn.Parameter(
                torch.tensor(t, dtype=torch.float, device="cuda").requires_grad_(True)
            )

        props = [p.name for p in el.properties]
        by_index = lambda prefix: sorted(
            (n for n in props if n.startswith(prefix)),
            key=lambda x: int(x.split("_")[-1]),
        )
        self._xyz = cols(["x", "y", "z"])
        self._weight = cols(["weight"])
        self._scaling = cols(by_index("scale_"))
        self._rotation = cols(by_index("rot"))
        self._values = cols(["value"])

    # ---- MCMC densification (--use_mcmc) ----

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "weight": self._weight,
            "value": self._values,
        }
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
            del self.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group["params"][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._weight = optimizable_tensors["weight"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._values = optimizable_tensors["value"]

    def _update_params(self, idxs, ratio):
        N = (ratio[idxs, 0] + 1).float()
        new_weight = self.get_weight[idxs, 0] / N
        new_weight = torch.clamp(
            new_weight.unsqueeze(-1),
            max=1.0 - torch.finfo(torch.float32).eps,
            min=0.005,
        )
        new_weight = self.inverse_weight_activation(new_weight)
        new_scaling = self._scaling[idxs]
        scaling = self.get_scaling[idxs]
        rotation = build_rotation(self._rotation[idxs])
        noise = torch.randn_like(scaling) * scaling * 0.5
        perturbed_xyz = self._xyz[idxs] + torch.bmm(
            rotation, noise.unsqueeze(-1)
        ).squeeze(-1)
        return (
            perturbed_xyz,
            new_weight,
            new_scaling,
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
        """Move dead Gaussians onto live ones sampled by weight, splitting their weight."""
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
        # Both the sources and the destinations hold new parameters: reset their moments.
        self.replace_tensors_to_optimizer(
            inds=torch.cat([reinit_idx, dead_indices]).unique()
        )

    def add_new_gs(self, cap_max):
        current_num_points = self._weight.shape[0]
        target_num = min(cap_max, int(1.5 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0
        probs = self.get_weight.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)
        new_xyz, new_weight, new_scaling, new_rotation, new_values = (
            self._update_params(add_idx, ratio=ratio)
        )
        self._weight[add_idx] = new_weight
        self._scaling[add_idx] = new_scaling
        self.densification_postfix(
            new_xyz, new_weight, new_scaling, new_rotation, new_values
        )
        self.replace_tensors_to_optimizer(inds=add_idx)
        return num_gs

    # ---- Row bookkeeping shared by every densification path ----

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _ema_prune(self, valid_points_mask):
        if self.ema_buffers is None:
            return
        for n in self.EMA_NAMES:
            self.ema_buffers[n] = self.ema_buffers[n][valid_points_mask]

    def _ema_extend(self, n_new):
        if self.ema_buffers is None or n_new <= 0:
            return
        for n in self.EMA_NAMES:
            # A new Gaussian's average starts at its current value.
            self.ema_buffers[n] = torch.cat(
                (self.ema_buffers[n], getattr(self, n)[-n_new:].detach().clone()), dim=0
            )

    def _ema_reset(self, idx):
        """Restart the average of rows overwritten in place."""
        if self.ema_buffers is None or idx is None or idx.numel() == 0:
            return
        for n in self.EMA_NAMES:
            self.ema_buffers[n][idx] = getattr(self, n)[idx].detach().clone()

    def prune_points(self, mask):
        valid_points_mask = ~mask
        self._ema_prune(valid_points_mask)
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._weight = optimizable_tensors["weight"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._values = optimizable_tensors["value"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

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
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._weight = optimizable_tensors["weight"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._values = optimizable_tensors["value"]
        self._ema_extend(new_xyz.shape[0])

    # ---- 3DGS clone/split densification (--densify_3dgs) ----

    def densify_and_split(
        self, grads, grad_threshold, scene_extent, N=2, max_new_points=None
    ):
        n_points = self.get_xyz.shape[0]
        grad_mag = torch.zeros(n_points, device=self.get_xyz.device)
        grad_mag[: grads.shape[0]] = torch.linalg.norm(grads, dim=-1)
        selected_pts_mask = grad_mag >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )
        selected_idx = selected_pts_mask.nonzero(as_tuple=True)[0]

        if max_new_points is not None:
            max_parents = int(max_new_points) // N
            if max_parents <= 0 or selected_idx.numel() == 0:
                return 0
            if selected_idx.numel() > max_parents:
                keep = torch.topk(
                    grad_mag[selected_idx], k=max_parents, sorted=False
                ).indices
                selected_idx = selected_idx[keep]
            selected_pts_mask = torch.zeros_like(selected_pts_mask)
            selected_pts_mask[selected_idx] = True

        num_added = N * selected_pts_mask.sum().item()
        if num_added == 0:
            return 0

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.get_xyz.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.inverse_scaling_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_weight = self._weight[selected_pts_mask].repeat(N, 1)
        new_values = self._values[selected_pts_mask].repeat(N, 1)
        self.densification_postfix(
            new_xyz, new_weight, new_scaling, new_rotation, new_values
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(),
                    device=self.get_xyz.device,
                    dtype=torch.bool,
                ),
            )
        )
        self.prune_points(prune_filter)
        return num_added

    def densify_and_clone(
        self, grads, grad_threshold, scene_extent, max_new_points=None
    ):
        n_points = self.get_xyz.shape[0]
        grad_mag = torch.zeros(n_points, device=self.get_xyz.device)
        grad_mag[: grads.shape[0]] = torch.linalg.norm(grads, dim=-1)
        selected_pts_mask = grad_mag >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )
        selected_idx = selected_pts_mask.nonzero(as_tuple=True)[0]

        if max_new_points is not None:
            max_new_points = int(max_new_points)
            if max_new_points <= 0 or selected_idx.numel() == 0:
                return 0
            if selected_idx.numel() > max_new_points:
                keep = torch.topk(
                    grad_mag[selected_idx], k=max_new_points, sorted=False
                ).indices
                selected_idx = selected_idx[keep]
            selected_pts_mask = torch.zeros_like(selected_pts_mask)
            selected_pts_mask[selected_idx] = True

        num_added = selected_pts_mask.sum().item()
        if num_added == 0:
            return 0
        self.densification_postfix(
            self._xyz[selected_pts_mask],
            self._weight[selected_pts_mask],
            self._scaling[selected_pts_mask],
            self._rotation[selected_pts_mask],
            self._values[selected_pts_mask],
        )
        return num_added

    # ---- Error-based densification (default) ----

    def _new_point_knn(self, new_pts):
        """distCUDA2 (mean squared distance to the 3 nearest other points) for new points only.

        Grid search over the 27 cells around each query, exact whenever its 3 nearest lie
        within the cell size; misses retry at twice the size, then fall back to distCUDA2.
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
            # Chunks sized to the densest cell keep the (chunk, 27, M) candidates bounded.
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

    def _write_rows(self, idx, d):
        """Overwrite rows idx in place, zero their Adam moments and restart their EMA.

        Clears every grad, so this iteration's Adam step is skipped as after prune/cat.
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

    def _new_gaussians(self, empty_points, empty_values):
        dist2 = torch.clamp_min(self._new_point_knn(empty_points) * 0.1, 0.0000001)
        n = empty_points.shape[0]
        rot = torch.zeros((n, 4), device="cuda")
        rot[:, 0] = 1
        return {
            "xyz": empty_points,
            "weight": self.inverse_weight_activation(
                NEW_WEIGHT * torch.ones((n, 1), dtype=torch.float, device="cuda")
            ),
            "scaling": self.inverse_scaling_activation(torch.sqrt(dist2))[
                ..., None
            ].repeat(1, 3),
            "rotation": rot,
            "value": self.inverse_value_activation(
                torch.clamp(empty_values, VALUE_EPS, 1.0 - VALUE_EPS)
            ),
        }

    def densify_in_place(self, empty_points, empty_values, min_weight):
        """Add Gaussians at the given points and prune those under min_weight.

        New Gaussians take the dead ones' rows instead of being appended after a full-size
        prune, so a full model only rewrites the replaced rows.
        """
        d = self._new_gaussians(empty_points, empty_values)
        k = empty_points.shape[0]
        slots = torch.nonzero((self.get_weight < min_weight).squeeze(-1)).squeeze(-1)
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
