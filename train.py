import math
import os
import sys
import time
import uuid
from argparse import ArgumentParser, Namespace

import numpy as np
import pyvista as pv
import torch
from diff_gaussian_rasterization.fused import fused_volume_loss
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import build_scaling_rotation, safe_state

CELL_COUNT = 128
# Auto densification: enough growth events that each grows the model by about this factor.
DENSIFY_GROWTH = 1.5


class NodeSampler:
    """Ground truth drawn from the native grid, resident on the GPU.

    Node mode (--jitter_gt 0) reads raw node values on an integer-strided subgrid with a fresh
    offset each iteration. Jitter mode precomputes K continuous sets: the full-domain lattice
    plus a uniform in-cell jitter, with the trilinear interpolant as GT.
    """

    def __init__(self, mesh, cell_count):
        nx, ny, nz = mesh.dimensions
        lo = np.array(mesh.bounds[0::2])
        hi = np.array(mesh.bounds[1::2])
        self.n = np.array([nx, ny, nz])
        self.lo = lo
        self.d = (hi - lo) / (self.n - 1)
        self.cc = cell_count
        self.stride = np.maximum((self.n - 1) // (cell_count - 1), 1)
        self.span = (cell_count - 1) * self.stride
        self.noff = self.n - self.span
        self.jittered = False
        # Structured points are stored [z, y, x], the rasterizer's cell order. fp16 so the
        # largest volumes fit next to the model; its step is far below the models' error.
        vals = mesh.point_data[mesh.point_data.keys()[0]]
        self.vol = torch.tensor(
            np.ascontiguousarray(vals.reshape(nz, ny, nx), dtype=np.float16),
            device="cuda",
        )
        self.off = np.zeros(3, dtype=np.int64)
        print(
            f"Node sampling: stride {self.stride.tolist()}, "
            f"{int(np.prod(self.noff))} offsets, {self.vol.numel() * 2 / 1e9:.1f} GB resident"
        )

    def draw(self, random_offset):
        self.off = (
            np.array([np.random.randint(m) for m in self.noff], dtype=np.int64)
            if random_offset
            else np.zeros(3, dtype=np.int64)
        )
        ox, oy, oz = self.off
        sx, sy, sz = self.stride
        return (
            self.vol[
                oz : oz + self.span[2] + 1 : sz,
                oy : oy + self.span[1] + 1 : sy,
                ox : ox + self.span[0] + 1 : sx,
            ]
            .contiguous()
            .float()
        )

    def build_jitter_sets(self, k):
        """Precompute k (jitter, trilinear GT) sets, then release the resident volume."""
        cc = self.cc
        ar = torch.arange(cc, device="cuda", dtype=torch.float32)
        zi, yi, xi = torch.meshgrid(ar, ar, ar, indexing="ij")
        base = [xi, yi, zi]
        n = [int(v) for v in self.n]
        st = [(n[a] - 1) / (cc - 1) for a in range(3)]
        self.jittered = True
        self.jit_step = st
        vol_flat = self.vol.reshape(-1)
        gts, jits = [], []
        for _ in range(k):
            i0, frac, jit = [], [], []
            for a in range(3):
                lattice = base[a] * st[a]
                f = (lattice + (torch.rand_like(base[a]) - 0.5) * st[a]).clamp_(
                    0.0, n[a] - 1.0
                )
                lo = f.floor().long().clamp_(max=n[a] - 2)
                i0.append(lo)
                frac.append(f - lo)
                jit.append((f - lattice) * self.d[a])
            x0, y0, z0 = i0
            tx, ty, tz = frac

            def corner(dx, dy, dz):
                return vol_flat[
                    ((z0 + dz) * n[1] + (y0 + dy)) * n[0] + (x0 + dx)
                ].float()

            def lerp(t, a, b):
                return a + t * (b - a)

            g = lerp(
                tz,
                lerp(
                    ty,
                    lerp(tx, corner(0, 0, 0), corner(1, 0, 0)),
                    lerp(tx, corner(0, 1, 0), corner(1, 1, 0)),
                ),
                lerp(
                    ty,
                    lerp(tx, corner(0, 0, 1), corner(1, 0, 1)),
                    lerp(tx, corner(0, 1, 1), corner(1, 1, 1)),
                ),
            )
            gts.append(g.half())
            jits.append(torch.stack(jit, dim=-1).reshape(-1).half())
        self.jit_gt = torch.stack(gts)
        self.jit_jitter = torch.stack(jits)
        vol_gb = self.vol.numel() * 2 / 1e9
        self.vol = None
        torch.cuda.empty_cache()
        print(
            f"Jittered GT: {k} sets, "
            f"{(self.jit_gt.numel() + self.jit_jitter.numel()) * 2 / 1e9:.2f} GB resident; "
            f"volume ({vol_gb:.1f} GB) released"
        )

    def draw_jit(self, idx):
        self.cur_jit = idx
        return self.jit_gt[idx].float(), self.jit_jitter[idx].float()

    def bounds(self):
        if self.jittered:
            return list(self.lo), list(self.lo + (self.n - 1) * self.d)
        mins = self.lo + self.off * self.d
        return list(mins), list(mins + self.span * self.d)

    def __getitem__(self, idx):
        """Sample positions of flat cell ids (z-major), for densification."""
        a = torch.div(idx, self.cc * self.cc, rounding_mode="floor")
        b = torch.div(idx % (self.cc * self.cc), self.cc, rounding_mode="floor")
        c = idx % self.cc
        node = torch.stack([c, b, a], dim=1).float()

        def t(v):
            return torch.tensor(v, dtype=torch.float, device=idx.device)

        if self.jittered:
            # The jittered position the error was measured at: seeding on the bare lattice
            # node stacks every event's Gaussians on the same points.
            jit = self.jit_jitter[getattr(self, "cur_jit", 0)].float().view(-1, 3)[idx]
            return t(self.lo) + node * t(self.jit_step) * t(self.d) + jit
        return t(self.lo) + (t(self.off) + node * t(self.stride)) * t(self.d)


def probe_gt(gaussians, cell_count):
    """(gt, jitter, samples) for --probe_gt: the mesh probed at 100 fixed jitters."""
    t0 = time.time()
    spacing = [
        (gaussians.maxes[i] - gaussians.mins[i]) / (cell_count - 1)
        for i in range(3)
    ]
    axes = [
        np.linspace(gaussians.mins[i], gaussians.maxes[i], cell_count)
        for i in range(3)
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    samples = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    samples = samples.reshape(cell_count, cell_count, cell_count, 3)
    # Reorder to the rasterizer's [z, y, x] cell layout.
    samples = np.flip(np.rot90(samples, k=1, axes=(2, 0)), axis=2).reshape(-1, 3)
    num_jitters = 100
    big_samples = np.tile(samples, (num_jitters, 1))
    big_jitter = np.random.uniform(-0.5, 0.5, big_samples.shape)
    big_jitter *= np.array(spacing)[None, :]
    big_jitter[: cell_count**3, :] = 0
    big_samples = np.clip(
        big_samples + big_jitter,
        np.array(gaussians.mins),
        np.array(gaussians.maxes),
    )
    big_gt = pv.PolyData(big_samples).sample(gaussians.mesh).point_data["value"]
    big_gt = big_gt.reshape(num_jitters, cell_count**3)
    big_samples = big_samples.reshape(num_jitters, cell_count**3, 3)
    big_jitter = big_jitter.reshape(num_jitters, cell_count**3, 3)
    print(f"Time to sample gt: {time.time() - t0}")

    def cuda(a):
        return torch.tensor(a, dtype=torch.float, device="cuda")

    return cuda(big_gt), cuda(big_jitter), cuda(big_samples)


def training(dataset, opt, args):
    prepare_output(dataset)
    if args.max_scale_k > 0:
        # The cap grows with the Gaussian spacing N^(-1/3); --max_scale is the floor.
        args.max_scale = max(
            args.max_scale, args.max_scale_k * args.cap_max ** (-1.0 / 3.0)
        )
        print(
            f"max_scale {args.max_scale:.4f} (k={args.max_scale_k}, N={args.cap_max})"
        )
    with open(os.path.join(dataset.model_path, "max_scale"), "w") as f:
        f.write(f"{args.max_scale}\n")
    gaussians = GaussianModel()
    gaussians.max_scale = args.max_scale
    scene = Scene(
        dataset,
        gaussians,
        normalized=args.is_scaled,
        fraction=args.fraction,
        max_init=(
            (args.init_share_max * args.cap_max) if args.init_share_max > 0 else None
        ),
    )
    gaussians.training_setup(opt)

    # Growth events: ~DENSIFY_GROWTH each, at least 3, done by half-way so the last ones train.
    latest = max(
        3, (opt.iterations // 2 - opt.densify_from_iter) // opt.densification_interval
    )
    n0 = max(1, gaussians.get_xyz.shape[0])
    want = math.ceil(math.log(max(args.cap_max, n0) / n0) / math.log(DENSIFY_GROWTH))
    densify_events = int(min(latest, max(3, want)))
    print(
        f"densify_events: {densify_events} ({n0:,} -> {args.cap_max:,} Gaussians, "
        f"at most {latest})"
    )
    scene.save(0)

    if args.probe_gt:
        node_sampler = None
        big_gt, big_jitter, big_samples = probe_gt(gaussians, CELL_COUNT)
        num_jitters = big_gt.shape[0]
    else:
        node_sampler = NodeSampler(gaussians.mesh, CELL_COUNT)
        samples = node_sampler
        if args.jitter_gt > 0:
            node_sampler.build_jitter_sets(args.jitter_gt)
        else:
            gt = node_sampler.draw(False)
            jitter = torch.zeros(3 * CELL_COUNT**3, dtype=torch.float, device="cuda")

    # Timed: the iteration loop only, minus saves.
    save_seconds = 0.0
    torch.cuda.synchronize()
    train_t0 = time.perf_counter()
    ema_names = GaussianModel.EMA_NAMES
    ema = None
    densifies = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):
        # Saved iterations use sample set 0, so their printed PSNR is comparable.
        fresh = iteration not in args.save_iterations
        if node_sampler is not None:
            if args.jitter_gt > 0:
                gt, jitter = node_sampler.draw_jit(
                    np.random.randint(args.jitter_gt) if fresh else 0
                )
            else:
                gt = node_sampler.draw(fresh)
            gaussians.mins, gaussians.maxes = node_sampler.bounds()
        else:
            jit_idx = np.random.randint(0, num_jitters) if fresh else 0
            jitter = big_jitter[jit_idx].ravel()
            gt = big_gt[jit_idx].reshape(CELL_COUNT, CELL_COUNT, CELL_COUNT)
            samples = big_samples[jit_idx]

        xyz_lr = gaussians.update_learning_rate(iteration)
        render_pkg = render(gaussians, jitter, CELL_COUNT)
        cells, weights = render_pkg["cells"], render_pkg["weights"]
        out = fused_volume_loss(cells, gt, weights, opt.fn_reg, args.fn_thresh)
        loss, l_v, false_negative = out[0], out[1], out[2]
        if args.use_mcmc:
            loss = loss + opt.weight_reg * torch.abs(gaussians.get_weight).mean()
            loss = loss + opt.scale_reg * torch.abs(gaussians.get_scaling).mean()
        loss.backward()

        with torch.no_grad():
            if iteration % 500 == 0:
                psnr = -10 * torch.log10(torch.mean((cells - gt) ** 2))
                progress_bar.set_postfix({"Loss": f"{loss.item():.5f}"})
                progress_bar.update(500)
                print(
                    f"Num Gaussians: {gaussians.get_values.shape[0]}, psnr: {psnr.item():.3f}, "
                    f"l_v: {l_v.item():.3e}, fn: {false_negative.item():.3e}, "
                    f"empty: {int(torch.count_nonzero((cells == -1) & (gt != -1)))}"
                )
            if iteration == opt.iterations:
                progress_bar.close()

            if (
                opt.densify_from_iter <= iteration <= opt.prune_until_iter
                and iteration % opt.densification_interval == 0
            ):
                if args.use_mcmc:
                    gaussians.relocate_gs(
                        dead_mask=(gaussians.get_weight <= 0.005).squeeze(-1)
                    )
                    gaussians.add_new_gs(cap_max=args.cap_max)
                else:
                    err_flat = torch.abs(cells.ravel() - gt.ravel())
                    budget = int(
                        args.cap_max
                        - gaussians.get_values.shape[0]
                        + torch.count_nonzero(gaussians.get_weight <= args.min_weight)
                    )
                    remaining = max(1, densify_events - densifies)
                    k = min(-(-budget // remaining), budget)
                    densifies += 1
                    k = max(0, min(k, err_flat.numel()))
                    if k > 0:
                        if args.densify_alpha <= 0:
                            loss_idx = torch.topk(err_flat, k).indices
                        else:
                            # Sample cells without replacement with probability ~ err^alpha
                            # (Gumbel top-k): plain top-k spends the budget on a thin tail.
                            logits = args.densify_alpha * torch.log(err_flat + 1e-12)
                            u = torch.rand_like(err_flat).clamp_min(1e-20)
                            loss_idx = torch.topk(
                                logits - torch.log(-torch.log(u)), k
                            ).indices
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            args.min_weight,
                            0.01,
                            samples[loss_idx],
                            gt.ravel()[loss_idx].reshape(-1, 1),
                            iteration > opt.densify_until_iter,
                            k,
                            args.densify_3dgs,
                        )

            if iteration < opt.iterations:
                use_ema = args.ema > 0 and iteration >= args.ema_from
                gaussians.optimizer_step(
                    ema=ema if use_ema else None, ema_w=1.0 - args.ema
                )
                gaussians.optimizer.zero_grad(set_to_none=True)
                if use_ema and ema is None:
                    ema = {n: getattr(gaussians, n).detach().clone() for n in ema_names}
                    # Shared with the model, which mirrors every row edit onto it.
                    gaussians.ema_buffers = ema

                if args.use_mcmc:
                    L = build_scaling_rotation(
                        gaussians.get_scaling, gaussians.get_rotation
                    )
                    covariance = L @ L.transpose(1, 2)
                    gate = 1 / (
                        1 + torch.exp(-100 * ((1 - gaussians.get_weight) - 0.995))
                    )
                    noise = (
                        torch.randn_like(gaussians._xyz) * gate * opt.noise_lr * xyz_lr
                    )
                    noise = torch.bmm(covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)

            if iteration in args.save_iterations:
                torch.cuda.synchronize()
                save_t0 = time.perf_counter()
                print(f"\n[ITER {iteration}] Saving Gaussians")
                # Ship the parameter average, not the last iterate.
                if ema is not None:
                    backup = {
                        n: getattr(gaussians, n).detach().clone() for n in ema_names
                    }
                    for n in ema_names:
                        getattr(gaussians, n).copy_(ema[n])
                scene.save(iteration)
                if ema is not None:
                    for n in ema_names:
                        getattr(gaussians, n).copy_(backup[n])
                torch.cuda.synchronize()
                save_seconds += time.perf_counter() - save_t0

    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_t0 - save_seconds
    print(f"TRAIN_SECONDS: {train_seconds:.2f} (save {save_seconds:.2f})", flush=True)
    with open(os.path.join(scene.model_path, "train_seconds.txt"), "w") as f:
        f.write(f"{train_seconds:.3f}\n")


def prepare_output(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))


def vtk_node_count(path):
    """Node count of a legacy structured-points .vtk, from its text header."""
    with open(path, "rb") as f:
        for _ in range(32):
            tok = f.readline().decode("ascii", "replace").split()
            if tok and tok[0].upper() == "DIMENSIONS":
                return int(tok[1]) * int(tok[2]) * int(tok[3])
    raise SystemExit(f"{path}: no DIMENSIONS line in the header; pass --cap_max")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.0,
        help="compression ratio: train to nodes / (12 * ratio) Gaussians (sets --cap_max)",
    )
    parser.add_argument("--fraction", type=float, default=0.000005)
    parser.add_argument("--min_weight", type=float, default=0.005)
    parser.add_argument(
        "--init_share_max",
        type=float,
        default=0.25,
        help="cap the initial Gaussian count at this share of the final count; 0 = no cap",
    )
    parser.add_argument(
        "--fn_thresh",
        type=float,
        default=0.002,
        help="tau: accumulated weight below which the FN barrier pushes a cell up",
    )
    parser.add_argument("--is_scaled", action="store_true")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--jitter_gt",
        type=int,
        default=400,
        help="train on this many precomputed continuous-jittered GT sets; "
        "0 = native node values on a strided subgrid",
    )
    parser.add_argument(
        "--probe_gt",
        action="store_true",
        help="GT probed on the 128^3 lattice at 100 fixed jitters",
    )
    parser.add_argument(
        "--use_mcmc", action="store_true", help="densify by 3DGS-MCMC relocation"
    )
    parser.add_argument(
        "--densify_3dgs", action="store_true", help="densify by 3DGS clone/split"
    )
    parser.add_argument(
        "--densify_alpha",
        type=float,
        default=2.5,
        help="place new Gaussians at cells sampled ~ err^alpha; 0 = top-k error",
    )
    parser.add_argument("--max_scale", type=float, default=0.02)
    parser.add_argument(
        "--max_scale_k",
        type=float,
        default=1.25,
        help="if > 0, max_scale = max(--max_scale, k * cap_max^(-1/3))",
    )
    parser.add_argument(
        "--ema",
        type=float,
        default=0.995,
        help="Polyak decay of the saved parameters; 0 = off",
    )
    parser.add_argument("--ema_from", type=int, default=1000)
    args = parser.parse_args(sys.argv[1:])
    if args.ratio > 0:
        args.cap_max = vtk_node_count(args.source_path) // int(round(12 * args.ratio))
        print(f"ratio {args.ratio:g}: cap_max {args.cap_max} Gaussians")
    elif args.cap_max <= 0:
        parser.error("give --ratio, or --cap_max (the final Gaussian count)")
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    training(lp.extract(args), op.extract(args), args)
    print("\nTraining complete.")
