"""A VEGGIES model: Gaussians with a mean, scale, rotation, value and weight each."""

import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gvraster import ops
from veggies.utils import exponential_lr, inverse_sigmoid, morton_codes

VALUE_EPS = 1e-6
INIT_WEIGHT = 0.1
NEW_WEIGHT = 0.01


def scale_cap(max_scale, k, bounds_min, bounds_max, n):
    """max(max_scale, k (V / n)^(1/3)), V the bounding-box volume; k <= 0 keeps max_scale."""
    if k <= 0:
        return max_scale
    vol = float(np.prod(np.array(bounds_max) - np.array(bounds_min)))
    cap = max(max_scale, k * (vol / n) ** (1.0 / 3.0))
    print(f"max_scale {cap:.4f} (k={k}, N={n}, V={vol:.3f})")
    return cap


class VeggiesModel:
    # Raw parameters, in the order the EMA and the ply use them.
    PARAMS = ("_means", "_scales", "_rotations", "_weights", "_values")
    GROUPS = {
        "means": "_means",
        "weights": "_weights",
        "scales": "_scales",
        "rotations": "_rotations",
        "values": "_values",
    }

    def __init__(self):
        for name in self.PARAMS:
            setattr(self, name, torch.empty(0))
        self.optimizer = None
        self.percent_dense = 0
        self.volume = None
        self.max_scale = 0.02
        self.bounds_min, self.bounds_max = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
        self.ema = None
        self._adam_steps = None
        self._adam_torch_only = False
        self._order, self._order_state = None, (0, 0)

    def _soft_cap(self, s):
        r = torch.linalg.norm(s, dim=1, keepdim=True) + 1e-8
        return s * (self.max_scale * torch.tanh(r / self.max_scale) / r)

    @property
    def means(self):
        return self._means

    @property
    def scales(self):
        return self._soft_cap(torch.exp(self._scales))

    @property
    def rotations(self):
        return torch.nn.functional.normalize(self._rotations)

    @property
    def weights(self):
        return torch.sigmoid(self._weights)

    @property
    def values(self):
        return torch.sigmoid(self._values)

    @property
    def count(self):
        return self._means.shape[0]

    def set_bounds(self, volume):
        xmin, xmax, ymin, ymax, zmin, zmax = volume.bounds
        self.bounds_min, self.bounds_max = [xmin, ymin, zmin], [xmax, ymax, zmax]

    def locality_order(self):
        """Morton order of the means, refreshed every 100 calls or when the count changes."""
        n = self.count
        prev_n, age = self._order_state
        if self._order is None or prev_n != n or age >= 100:
            self._order = torch.argsort(morton_codes(self._means))
            self._order_state = (n, 0)
        else:
            self._order_state = (n, age + 1)
        return self._order

    def init_from_seeds(self, seeds, volume):
        """One isotropic Gaussian per seed point, sized by the mean distance to its 3 neighbours."""
        points = torch.tensor(np.asarray(seeds.points)).float().cuda()
        self.set_bounds(volume)
        print(f"Seeds: {points.shape[0]}")
        dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rotations = torch.zeros((points.shape[0], 4), device="cuda")
        rotations[:, 0] = 1
        weights = inverse_sigmoid(
            INIT_WEIGHT
            * torch.ones((points.shape[0], 1), dtype=torch.float, device="cuda")
        )
        values = inverse_sigmoid(
            torch.clamp(
                torch.tensor(seeds.values, dtype=torch.float, device="cuda"),
                VALUE_EPS,
                1.0 - VALUE_EPS,
            )
        )
        self._means = nn.Parameter(points.requires_grad_(True))
        self._scales = nn.Parameter(scales.requires_grad_(True))
        self._rotations = nn.Parameter(rotations.requires_grad_(True))
        self._weights = nn.Parameter(weights.requires_grad_(True))
        self._values = nn.Parameter(values.requires_grad_(True))
        self.volume = volume

    def setup_optimizer(self, opts):
        self.percent_dense = opts.percent_dense
        groups = [
            ("means", self._means, opts.position_lr_init),
            ("weights", self._weights, opts.weight_lr),
            ("scales", self._scales, opts.scaling_lr),
            ("rotations", self._rotations, opts.rotation_lr),
            ("values", self._values, opts.values_lr),
        ]
        self.optimizer = torch.optim.Adam(
            [{"params": [p], "lr": lr, "name": name} for name, p, lr in groups],
            lr=0.0,
            eps=1e-15,
            fused=True,
        )
        self.means_lr = exponential_lr(
            lr_init=opts.position_lr_init,
            lr_final=opts.position_lr_final,
            lr_delay_mult=opts.position_lr_delay_mult,
            max_steps=opts.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        for group in self.optimizer.param_groups:
            if group["name"] == "means":
                group["lr"] = self.means_lr(iteration)
                return group["lr"]

    def step(self, ema_decay, ema_on):
        """Adam step, then the parameter average (EMA), which starts at the parameters the first
        time and restarts should its rows ever fall out of step with the model's."""
        ema = self.ema if ema_on else None
        if ema is not None and ema["_means"].shape[0] != self.count:
            print("[ema] buffer desync; restarting")
            ema = self.ema = None
        ema_w = 1.0 - ema_decay
        if not self._adam_step(ema, ema_w) and ema is not None:
            torch._foreach_lerp_(
                [ema[n] for n in self.PARAMS],
                [getattr(self, n).detach() for n in self.PARAMS],
                ema_w,
            )
        self.optimizer.zero_grad(set_to_none=True)
        if ema_on and self.ema is None:
            self.ema = {n: getattr(self, n).detach().clone() for n in self.PARAMS}

    def _adam_step(self, ema, ema_w):
        """One launch of the fused Adam kernel (which also updates the EMA); returns whether it
        ran. torch owns the Adam state and takes the first step and any the kernel cannot.
        """
        opt = self.optimizer
        if self._adam_torch_only:
            opt.step()
            return False
        groups = [g for g in opt.param_groups if g["params"][0].grad is not None]
        if not groups:
            return False
        states = [opt.state.get(g["params"][0]) for g in groups]
        if (
            len(groups) != len(opt.param_groups)
            or not all(states)
            or self._adam_steps is None
        ):
            opt.step()
            steps = {
                int(st["step"])
                for st in (opt.state[g["params"][0]] for g in opt.param_groups)
                if st
            }
            self._adam_steps = steps.pop() if len(steps) == 1 else None
            self._adam_torch_only = self._adam_steps is None
            return False
        params = [g["params"][0] for g in groups]
        if not all(p.grad.is_contiguous() for p in params):
            opt.step()
            self._adam_steps += 1
            return False
        self._adam_steps += 1
        b1, b2 = groups[0]["betas"]
        emas = [ema[self.GROUPS[g["name"]]] for g in groups] if ema is not None else []
        ops.multi_adam_(
            [p.data for p in params],
            [p.grad for p in params],
            [st["exp_avg"] for st in states],
            [st["exp_avg_sq"] for st in states],
            [float(g["lr"]) for g in groups],
            self._adam_steps,
            float(b1),
            float(b2),
            float(groups[0]["eps"]),
            emas,
            float(ema_w),
        )
        torch._foreach_add_([st["step"] for st in states], 1.0)
        return ema is not None

    def save_ply(self, path, ema=None):
        """Raw parameters, 12 floats per Gaussian; the EMA's instead if given."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        src = ema if ema is not None else {n: getattr(self, n) for n in self.PARAMS}
        names = ["x", "y", "z", "value", "weight"]
        names += [f"scale_{i}" for i in range(src["_scales"].shape[1])]
        names += [f"rot_{i}" for i in range(src["_rotations"].shape[1])]
        data = np.concatenate(
            [
                src[n].detach().cpu().numpy()
                for n in ("_means", "_values", "_weights", "_scales", "_rotations")
            ],
            axis=1,
        )
        elements = np.empty(data.shape[0], dtype=[(a, "f4") for a in names])
        elements[:] = list(map(tuple, data))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_ply(self, path, volume=None):
        el = PlyData.read(path).elements[0]

        def columns(prefix):
            names = sorted(
                (p.name for p in el.properties if p.name.startswith(prefix)),
                key=lambda x: int(x.split("_")[-1]),
            )
            return np.stack([np.asarray(el[n]) for n in names], axis=1)

        def param(a):
            return nn.Parameter(
                torch.tensor(a, dtype=torch.float, device="cuda").requires_grad_(True)
            )

        self._means = param(np.stack([np.asarray(el[c]) for c in "xyz"], axis=1))
        self._weights = param(np.asarray(el["weight"])[..., np.newaxis])
        self._scales = param(columns("scale_"))
        self._rotations = param(columns("rot"))
        self._values = param(np.asarray(el["value"])[..., np.newaxis])
        if volume is not None:
            self.set_bounds(volume)
            self.volume = volume

    # Row edits. Each keeps the Adam state and the EMA aligned with the parameters, or the saved
    # average would mix unrelated Gaussians.

    def _set_params(self, params):
        for name, p in params.items():
            setattr(self, self.GROUPS[name], p)

    def _swap(self, group, tensor, update_state):
        old = group["params"][0]
        state = self.optimizer.state.pop(old, None)
        group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
        if state is not None:
            update_state(state)
            self.optimizer.state[group["params"][0]] = state
        return group["params"][0]

    def rewrap(self, rows=None):
        """Re-wrap the parameters after in-place edits, zeroing the Adam moments of rows (all if
        None)."""

        def reset(st):
            if rows is None:
                st["exp_avg"] = torch.zeros_like(st["exp_avg"])
                st["exp_avg_sq"] = torch.zeros_like(st["exp_avg_sq"])
            else:
                st["exp_avg"][rows] = 0
                st["exp_avg_sq"][rows] = 0

        self._set_params(
            {
                g["name"]: self._swap(g, getattr(self, self.GROUPS[g["name"]]), reset)
                for g in self.optimizer.param_groups
            }
        )

    def prune(self, mask):
        keep = ~mask

        def prune_state(st):
            st["exp_avg"] = st["exp_avg"][keep]
            st["exp_avg_sq"] = st["exp_avg_sq"][keep]

        self._set_params(
            {
                g["name"]: self._swap(g, g["params"][0][keep], prune_state)
                for g in self.optimizer.param_groups
            }
        )
        if self.ema is not None:
            for n in self.PARAMS:
                self.ema[n] = self.ema[n][keep]

    def append(self, means, weights, scales, rotations, values):
        new = {
            "means": means,
            "weights": weights,
            "scales": scales,
            "rotations": rotations,
            "values": values,
        }
        params = {}
        for g in self.optimizer.param_groups:
            ext = new[g["name"]]

            def extend(st):
                st["exp_avg"] = torch.cat((st["exp_avg"], torch.zeros_like(ext)), dim=0)
                st["exp_avg_sq"] = torch.cat(
                    (st["exp_avg_sq"], torch.zeros_like(ext)), dim=0
                )

            params[g["name"]] = self._swap(
                g, torch.cat((g["params"][0], ext), dim=0), extend
            )
        self._set_params(params)
        n_new = means.shape[0]
        if self.ema is not None and n_new > 0:
            for n in self.PARAMS:
                self.ema[n] = torch.cat(
                    (self.ema[n], getattr(self, n)[-n_new:].detach().clone()), dim=0
                )

    def overwrite(self, rows, new):
        """Write rows in place (new: group name -> rows); clears every grad, so this iteration's
        Adam step is skipped as after a prune or an append."""
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                p = group["params"][0]
                p.data[rows] = new[group["name"]].to(p.dtype)
                st = self.optimizer.state.get(p, None)
                if st:
                    st["exp_avg"][rows] = 0.0
                    st["exp_avg_sq"][rows] = 0.0
                p.grad = None
        self.restart_ema(rows)

    def restart_ema(self, rows):
        if self.ema is None or rows is None or rows.numel() == 0:
            return
        for n in self.PARAMS:
            self.ema[n][rows] = getattr(self, n)[rows].detach().clone()
