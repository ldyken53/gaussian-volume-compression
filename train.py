import os
import sys
import uuid
import json
import time
from argparse import ArgumentParser, Namespace
from random import randint
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pyvista as pv

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from gpu_mesh_sampling import gpu_sample, gpu_sampleu
from scene import GaussianModel, Scene
from utils.debug_utils import tensor_to_vtk, analyze_array
from utils.general_utils import get_expon_lr_func, safe_state, build_scaling_rotation
from utils.image_utils import psnr
from utils.loss_utils import bounding_box_regularization, create_window, l1_loss, l2_loss

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

DEBUG = True


class NodeSampler:
    """Ground truth read straight off the native grid, on an integer-strided subgrid.

    Default since Sep 2026; --probe_gt restores what came before, `probe.sample` on a
    continuous cell_count^3 lattice offset by one of 100 precomputed jitters. That target
    is defensible -- a renderer samples the trilinear interpolant too -- but the offset set
    is fixed, and at low compression the model has the capacity to fit it: miranda 64x read
    55.2 dB on a training offset and 53.5 dB on a fresh one.

    Here the subgrid is s_d = (n_d - 1) // (cell_count - 1) native nodes apart per axis,
    with a fresh integer offset drawn every iteration. GT is raw node values: no
    interpolation, no probe (so no startup cost), and s_x*s_y*s_z distinct offsets -- 512
    for a 1024^3 volume against the old 100 -- which between them cover every node, so
    there is nothing left to overfit. Measured over all nine dataset/ratio configs it is
    -0.04 to +1.9 dB against the raw voxel array and +0.15 to -1.5 dB at continuous
    positions; both extremes are miranda 64x, the only config with the capacity for it to
    matter.
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
        # The rasterizer's own cell order is [z,y,x], which is also how a structured-points
        # volume is stored, so the raw array needs no permutation.
        vals = mesh.point_data[mesh.point_data.keys()[0]]
        # fp16 residency. The volume is the single largest allocation -- richt is 8.05e9
        # nodes, 32.2 GB in fp32, which does not leave room on a 40 GB A100 for a 64x
        # model. Values are min-max normalised to [0,1], where fp16's step is at worst
        # 4.9e-4 near 1.0: an RMS quantisation error of ~1.4e-4, i.e. a 77 dB ceiling,
        # far above the 31-53 dB these models reach. The drawn subgrid is cast back to
        # fp32 before it is ever used in the loss.
        self.vol = torch.tensor(
            np.ascontiguousarray(vals.reshape(nz, ny, nx), dtype=np.float16), device="cuda")
        self.off = np.zeros(3, dtype=np.int64)
        print(f"Node sampling: stride {self.stride.tolist()}, "
              f"{int(np.prod(self.noff))} offsets, {self.vol.numel() * 2 / 1e9:.1f} GB resident")

    def draw(self, random_offset):
        self.off = (np.array([np.random.randint(m) for m in self.noff], dtype=np.int64)
                    if random_offset else np.zeros(3, dtype=np.int64))
        ox, oy, oz = self.off
        sx, sy, sz = self.stride
        return self.vol[oz:oz + self.span[2] + 1:sz,
                        oy:oy + self.span[1] + 1:sy,
                        ox:ox + self.span[0] + 1:sx].contiguous().float()


    def build_jitter_sets(self, k):
        """Precompute k (offset, jitter, GT) sets for continuous-jittered training.

        Positions are the fixed cc^3 lattice spanning the whole volume (the old
        --probe_gt lattice, no subgrid offsets) plus a fresh uniform jitter within
        +/- 0.5 lattice cells per axis, clamped to the volume; GT is the trilinear
        interpolant of the native grid at those positions -- the field a renderer
        actually asks the model for -- gathered on-GPU from the resident fp16 volume
        instead of the old startup pyvista probe. Everything is built before the
        timed loop; per-iteration cost is one index plus the same fp16->fp32 cast
        draw() already pays. Storage is fp16, k * (4.2 + 12.6) MB on-GPU for cc=128.
        """
        cc = self.cc
        ar = torch.arange(cc, device="cuda", dtype=torch.float32)
        zi, yi, xi = torch.meshgrid(ar, ar, ar, indexing="ij")
        base = [xi, yi, zi]
        n = [int(v) for v in self.n]
        # Full-domain lattice step in native-node units; fractional in general.
        st = [(n[a] - 1) / (cc - 1) for a in range(3)]
        self.jit_full = True
        self.jit_step = st
        vol_flat = self.vol.reshape(-1)
        gts, jits = [], []
        for _ in range(k):
            i0, frac, jit = [], [], []
            for a in range(3):
                lattice = base[a] * st[a]
                f = (lattice + (torch.rand_like(base[a]) - 0.5) * st[a]).clamp_(0.0, n[a] - 1.0)
                lo = f.floor().long().clamp_(max=n[a] - 2)
                i0.append(lo)
                frac.append(f - lo)
                jit.append((f - lattice) * self.d[a])
            x0, y0, z0 = i0
            tx, ty, tz = frac
            def corner(dx, dy, dz):
                return vol_flat[((z0 + dz) * n[1] + (y0 + dy)) * n[0] + (x0 + dx)].float()
            lerp = lambda t, a, b: a + t * (b - a)
            g = lerp(tz,
                     lerp(ty, lerp(tx, corner(0, 0, 0), corner(1, 0, 0)),
                              lerp(tx, corner(0, 1, 0), corner(1, 1, 0))),
                     lerp(ty, lerp(tx, corner(0, 0, 1), corner(1, 0, 1)),
                              lerp(tx, corner(0, 1, 1), corner(1, 1, 1))))
            gts.append(g.half())
            jits.append(torch.stack(jit, dim=-1).reshape(-1).half())
        self.jit_gt = torch.stack(gts)
        self.jit_jitter = torch.stack(jits)
        # The resident volume exists only to build these sets: in jitter mode nothing
        # after this point reads it (reporting draws set 0, densification uses lattice
        # positions), so release the 2-16 GB instead of carrying it through training.
        vol_gb = self.vol.numel() * 2 / 1e9
        self.vol = None
        torch.cuda.empty_cache()
        print(f"Jittered GT: {k} sets, "
              f"{(self.jit_gt.numel() + self.jit_jitter.numel()) * 2 / 1e9:.2f} GB resident; "
              f"volume ({vol_gb:.1f} GB) released")

    def draw_jit(self, idx):
        return self.jit_gt[idx].float(), self.jit_jitter[idx].float()

    def bounds(self):
        if getattr(self, "jit_full", False):
            return list(self.lo), list(self.lo + (self.n - 1) * self.d)
        mins = self.lo + self.off * self.d
        return list(mins), list(mins + self.span * self.d)

    def __getitem__(self, idx):
        """Positions of flat cell ids, for densification. cell_id = a*cc^2 + b*cc + c with
        a = z, b = y, c = x, matching the rasterizer's indexing."""
        a = torch.div(idx, self.cc * self.cc, rounding_mode="floor")
        b = torch.div(idx % (self.cc * self.cc), self.cc, rounding_mode="floor")
        c = idx % self.cc
        node = torch.stack([c, b, a], dim=1).float()
        t = lambda v: torch.tensor(v, dtype=torch.float, device=idx.device)
        if getattr(self, "jit_full", False):
            return t(self.lo) + node * t(self.jit_step) * t(self.d)
        return t(self.lo) + (t(self.off) + node * t(self.stride)) * t(self.d)


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    log_to_file,
    fraction,
    min_weight,
    is_scaled,
    precomputed_samples,
    use_mcmc
):
    print(use_mcmc)
    vtk_files = []
    vtk_files_loss = []
    log_data = []
    first_iter = 0
    prepare_output(dataset)
    gaussians = GaussianModel()
    gaussians.max_scale = args.max_scale
    scene = Scene(
        dataset, 
        gaussians, 
        load_iteration=0 if args.model_path else None, 
        normalized=is_scaled, 
        fraction=fraction)
    gaussians.training_setup(opt)
    scene.save(0)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    ema_lv_for_log = 0.0
    ema_lfp_for_log = 0.0
    ema_lfn_for_log = 0.0
    ema_lpsnr_for_log = 0.0
    error_thresh = 0.05
    new_scale = 0.006 # 4 * 100^3 cell?
    densifies = 0

    # Make ground truth
    cell_count = 128
    spacing = [
        (gaussians.maxes[0] - gaussians.mins[0]) / (cell_count - 1),
        (gaussians.maxes[1] - gaussians.mins[1]) / (cell_count - 1),
        (gaussians.maxes[2] - gaussians.mins[2]) / (cell_count - 1)
    ]
    start = time.time()
    # Node sampling is the default; --probe_gt restores the continuous jittered lattice,
    # and --precomputed_samples implies it since those .npy files are lattice samples.
    node_sampler = (None if args.probe_gt or precomputed_samples
                    else NodeSampler(gaussians.mesh, cell_count))
    if node_sampler is not None:
        gt = node_sampler.draw(False)
        jitter_cuda = torch.zeros(3 * cell_count ** 3, dtype=torch.float, device="cuda")
        samples_cuda = node_sampler
        if args.jitter_gt > 0:
            node_sampler.build_jitter_sets(args.jitter_gt)
    elif precomputed_samples:
        big_gt = np.load("richtmyer_meshkov_big_gt.npy")
        num_jitters = big_gt.shape[0]
        size = big_gt.shape[1]
        big_samples = np.load("richtmyer_meshkov_big_samples.npy")
        big_jitter = np.load("richtmyer_meshkov_big_jitter.npy")
    else:
        x = np.linspace(gaussians.mins[0], gaussians.maxes[0], cell_count)
        y = np.linspace(gaussians.mins[1], gaussians.maxes[1], cell_count)
        z = np.linspace(gaussians.mins[2], gaussians.maxes[2], cell_count)
        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        samples = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
        samples_3d = samples.reshape(cell_count, cell_count, cell_count, 3)
        rot = np.rot90(samples_3d, k=1, axes=(2,0))
        samples_tf = np.flip(rot, axis=2)
        samples_tf_flat = samples_tf.reshape(-1, 3)
        num_jitters = 100
        big_samples = np.tile(samples_tf_flat, (num_jitters, 1))
        big_jitter = np.random.uniform(-0.5, 0.5, big_samples.shape)
        big_jitter *= np.array(spacing)[None, :]
        big_jitter[: cell_count**3, :] = 0
        big_samples = np.clip(
            big_samples + big_jitter,
            np.array(gaussians.mins),
            np.array(gaussians.maxes)
        )
        probe = pv.PolyData(big_samples)
        sampled = probe.sample(gaussians.mesh)
        big_gt = sampled.point_data['value']
        # big_gt = gpu_sample(
        #     gaussians.mesh.dimensions,
        #     gaussians.mesh.origin,
        #     gaussians.mesh.spacing,
        #     gaussians.mesh.point_data['value'],
        #     big_samples
        # )
        # big_gt = gpu_sampleu(
        #     gaussians.mesh.points, 
        #     gaussians.mesh.cell_connectivity.astype(np.int64),
        #     gaussians.mesh.point_data['value'],
        #     big_samples
        # )
        big_gt = big_gt.reshape(num_jitters, cell_count**3)
        # big_gt_weights = big_gt.copy()
        # big_gt_weights[big_gt_weights != -1] = 1
        # big_gt_weights[big_gt_weights == -1] = 0
        big_samples = big_samples.reshape(num_jitters, cell_count**3, 3)
        big_jitter = big_jitter.reshape(num_jitters, cell_count**3, 3)
        end = time.time()
        print(f"Time to sample gt: {end - start}")
    if node_sampler is None:
        big_jitter_cuda = torch.tensor(big_jitter, dtype=torch.float, device="cuda")
        big_gt_cuda = torch.tensor(big_gt, dtype=torch.float, device="cuda")
        big_samples_cuda = torch.tensor(big_samples, dtype=torch.float, device="cuda")
        gt = big_gt_cuda[0].reshape(cell_count, cell_count, cell_count)
        print(f"Number of invalid samples: {torch.count_nonzero(gt == -1)}")
        # tensor_to_vtk(gt.cpu().numpy(), "test_gt.vtk", spacing)
        # gt_weights = big_gt_weights[0].reshape(cell_count, cell_count, cell_count)
        # gt_weights = torch.tensor(gt_weights).cuda()
        jitter_cuda = big_jitter_cuda[0].ravel()
    loss_samples = np.empty((0, 3))
    loss_vals = np.empty((0, 1))

    # Training-time instrumentation. Wall clock over the iteration loop only:
    # excludes mesh loading, GT construction and checkpoint writes, which is the
    # convention the baseline table uses ("training time only").
    save_seconds = 0.0
    torch.cuda.synchronize()
    train_t0 = time.perf_counter()
    # --ema state: dict of buffers, (re)built lazily so densification cannot desync it.
    ema_params = None
    ema_names = ["_xyz", "_scaling", "_rotation", "_weight", "_values"]
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    done = 0
    n = False
    for iteration in range(first_iter, opt.iterations + 1):
        deb = False
        iter_start.record()
        fresh = (iteration not in saving_iterations and iteration not in testing_iterations
                 and done != 1)
        if node_sampler is not None:
            if args.jitter_gt > 0:
                # Set 0 on reporting iterations so the logged PSNR is a fixed sample set.
                jit_idx = np.random.randint(args.jitter_gt) if fresh else 0
                gt, jitter_cuda = node_sampler.draw_jit(jit_idx)
            else:
                # Offset 0 on reporting iterations so the logged PSNR is a fixed subgrid.
                gt = node_sampler.draw(fresh)
            gaussians.mins, gaussians.maxes = node_sampler.bounds()
        else:
            jit_idx = np.random.randint(0, num_jitters) if fresh else 0
            jitter_cuda = big_jitter_cuda[jit_idx].ravel()
            gt = big_gt_cuda[jit_idx].reshape(cell_count, cell_count, cell_count)
            # gt_weights = big_gt_weights[jit_idx].reshape(cell_count, cell_count, cell_count)
            # gt_weights = torch.tensor(gt_weights).cuda()
            samples_cuda = big_samples_cuda[jit_idx]

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if iteration % 2000 == 0:
            deb = True

        render_pkg = render(
            gaussians,
            pipe,
            jitter_cuda,
            cell_count,
            debug=deb
        )
        cells, weights, visibility_filter, radii = (
            render_pkg["cells"],
            render_pkg["weights"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        # l1_lv = l1_loss(cells, gt)
        # PSNR is MSE-based, so L2 optimises the reported metric directly.
        err = (cells - gt) ** 2 if args.loss == "l2" else torch.abs(cells - gt)
        l1_lv = err.mean()
        # l1_lv = torch.mean((cells - gt) ** 2)
        # delta = 1.0
        # residual = cells - gt
        # loss = torch.where(
        #     residual.abs() <= delta,
        #     0.5 * residual ** 2,
        #     delta * (residual.abs() - 0.5 * delta)
        # )
        # l1_lv = loss.mean()
        # TODO: FIX FP AND FN FOR CHANGING CELL COUNTS
        k = 600  # Adjust this to control decay rate
        # fn_mask = torch.logical_and(gt != -1, weights < 0.02)
        # false_negative = torch.exp(-k * weights[fn_mask])
        # false_negative = false_negative[false_negative > 0].mean()
            # false_negative = torch.exp(-k * torch.clamp(weights[fn_mask] - 0.01, 0.0))
        # False-negative loss. A cell whose accumulated weight drops under the
        # rasterizer cutoff (1e-2) is undefined, so penalise cells approaching it.
        # Cells already at W == 0 are excluded: the rasterizer zeroes their weight
        # and skips them in the backward pass, so they carry no gradient and would
        # only dilute the mean, weakening the push on the cells still recoverable.
        fn_mask = torch.logical_and(gt != -1, weights > 0.0)
        fn_vals = torch.clamp(0.011 - weights, min=0) * fn_mask
        # "rel": fn_reg multiplies the current data loss instead of being an absolute
        # weight, so the FN term keeps a fixed ratio to the data term regardless of the
        # PSNR regime the dataset sits in. Ported from gaussian-volume to test whether a
        # single setting can span both repos. Default "abs" leaves behaviour unchanged.
        fn_scale = args.fn_reg * l1_lv.detach() if args.fn_mode == "rel" else args.fn_reg
        false_negative = fn_scale * fn_vals.sum() / ((fn_vals > 0).sum().float() + 1e-8)
        # overlap_mask = torch.logical_and(gt != -1, weights > 1.0)
        # if overlap_mask.any():
            # overlap_loss = (torch.exp(weights[overlap_mask] - 1) - 1).mean()
        # else:
        #     overlap_loss = torch.tensor(0., device="cuda")
        # fp_mask = torch.logical_and(gt == -1, weights > 0.0)
        # fp_vals = weights[fp_mask]        
        # false_positive = args.fp_reg * fp_vals.sum() / ((fp_vals > 0).sum().float() + 1e-8)
        # if mask.any():
        #     false_positive = (2 * (1 - torch.exp(-k * weights[mask]))).mean()
        # else:
        #     false_positive = torch.tensor(0., device="cuda")
        # min_allowed_scale = min(spacing) / 6.0  # One cell worth of extent
        # mask = gaussians.get_scaling < min_allowed_scale
        # if mask.any():
        #     false_positive = torch.relu(min_allowed_scale - gaussians.get_scaling[mask]).mean()
        # else:
        #     false_positive = torch.tensor(0., device="cuda")
        # loss = l1_lv + false_negative + 0.0000 * overlap_loss
        loss = l1_lv + false_negative
        if gaussians.get_values.shape[0] > args.cap_max:
            n = True
        if use_mcmc:
            loss = loss + args.weight_reg * torch.abs(gaussians.get_weight).mean()
            loss = loss + args.scale_reg * torch.abs(gaussians.get_scaling).mean()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Logging
            if args.fn_debug and iteration % args.fn_debug == 0:
                print(f"[fndbg] iter {iteration} N {gaussians.get_values.shape[0]} "
                      f"dead {int(torch.count_nonzero(cells == -1))} "
                      f"near {int(torch.count_nonzero((weights > 0) & (weights <= 0.011)))} "
                      f"of {cells.numel()}", flush=True)
            if log_to_file and iteration % 100 == 0:
                mse = torch.mean((cells - gt) ** 2)
                psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
                mse2 = torch.mean((cells[torch.logical_and(weights > 0, gt != -1)] - gt[torch.logical_and(weights > 0, gt != -1)]) ** 2)
                psnr2 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse2 + 1e-8)
                num_gaussians = gaussians.get_values.shape[0]
                log_data.append({
                    "iteration": iteration,
                    "loss": loss.item(),
                    "l_v": l1_lv.item(),
                    # "false_positive": false_positive.item(),
                    "psnr": psnr.item(),
                    # "psnr2": psnr2.item(),
                    "num_gaussians": num_gaussians
                })
            
            # Progress bar
            if iteration % 500 == 0:
                ema_loss_for_log = 0.9 * loss.item() + 0.1 * ema_loss_for_log
                mse = torch.mean((cells - gt) ** 2)
                psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
                mse2 = torch.mean((cells[torch.logical_and(weights > 0, gt != -1)] - gt[torch.logical_and(weights > 0, gt != -1)]) ** 2)
                psnr2 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse2 + 1e-8)
                # ema_lv_for_log = 0.4 * l1_lv + 0.6 * ema_lv_for_log
                # ema_lfp_for_log = 0.4 * false_positive + 0.6 * ema_lfp_for_log
                # ema_lfn_for_log = 0.4 * false_negative + 0.6 * ema_lfn_for_log
                # ema_lpsnr_for_log = 0.4 * psnr + 0.6 * ema_lpsnr_for_log
                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.{5}f}",
                        # "L_v": f"{ema_lv_for_log:.{5}f}",
                        # "L_fp": f"{ema_lfp_for_log:.{5}f}",
                        # "L_fn": f"{ema_lfn_for_log:.{5}f}",
                        # "PSNR": f"{ema_lpsnr_for_log:.{5}f}"
                    }
                )
                progress_bar.update(500)
                print(f"Num Gaussians: {gaussians.get_values.shape[0]}, psnr: {psnr}, psnr2: {psnr2}, l_v: {l1_lv.item()}")
                # print(f"Num Gaussians: {gaussians.get_values.shape[0]}")
                print(f"w= {(args.weight_reg * torch.abs(gaussians.get_weight).mean()).item()}, {(args.scale_reg * torch.abs(gaussians.get_scaling).mean()).item()}, fn: {false_negative}")
                print(f"Gaussian weight: {torch.mean(gaussians.get_weight)}, gaussian scale: {torch.mean(gaussians.get_scaling)}, scale var: {torch.std(gaussians.get_scaling)}")
                print(f"False negative: {torch.count_nonzero(torch.logical_and(cells == -1, gt != -1))}, fn_reg: {args.fn_reg}")
                # print(f"Overlaps: {torch.count_nonzero(torch.logical_and(gt != -1, weights > 1.0))}")
                print(f"Number of Gaussians to prune: {torch.count_nonzero((gaussians.get_weight < min_weight))}")
               # print(f"Loss samples: {loss_samples.shape}")
                # x = cells * weights
                # low = (x) / (weights + mean_weight)
                # high = (x + mean_weight) / (weights + mean_weight)
                # mm = torch.logical_and(
                #     gt >= low,
                #     gt <= high
                # )
                # print(f"Num between range1: {torch.count_nonzero(gt < low)}, range2: {torch.count_nonzero(gt > high)}")
                # print(f"Num between range1: {torch.count_nonzero(mm)}")
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification
            if (iteration <= opt.prune_until_iter and
                iteration >= opt.densify_from_iter and
                iteration % opt.densification_interval == 0 and
                iteration not in testing_iterations
            ):
                # if gaussians.get_values.shape[0] > args.cap_max:
                if use_mcmc:
                    # pass
                    dead_mask = (gaussians.get_weight <= 0.005).squeeze(-1)
                    gaussians.relocate_gs(dead_mask=dead_mask, cells=cells, gt=gt)
                    gaussians.add_new_gs(cap_max=args.cap_max)
                else:
                    err_flat = torch.abs(cells.ravel() - gt.ravel())
                    budget = int(args.cap_max - gaussians.get_values.shape[0]
                                 + torch.count_nonzero(gaussians.get_weight <= min_weight) + 1000)
                    if args.densify_batch > 0:
                        k = min(args.densify_batch, budget)
                    else:
                        # Auto: spread the remaining budget over a fixed number of
                        # densification events. The optimum sits at ~3 events on every
                        # dataset/ratio tested, and unlike a fixed Gaussian count this
                        # scales automatically with cap_max, init size and iterations
                        # (a fixed count silently becomes all-at-once at high
                        # compression, which is the worst regime).
                        remaining = max(1, args.densify_events - densifies)
                        k = min(-(-budget // remaining), budget)
                    densifies += 1
                    k = max(0, min(k, err_flat.numel()))
                    if k > 0:
                        if args.densify_alpha <= 0:
                            # --densify_alpha 0 restores the old pure-topk placement.
                            loss_idx = torch.topk(err_flat, k).indices
                        else:
                            # Sample cells without replacement with probability
                            # proportional to err**alpha, via the Gumbel top-k trick
                            # (still a single topk). Plain topk aims the whole budget
                            # at a thin tail -- the worst 1% of cells hold only ~13% of
                            # the squared error -- so spreading placement over the
                            # error *mass* uses the budget better. alpha=1.5 gains
                            # +0.14 to +1.79 dB across chameleon/miranda at 64/256/1024x
                            # (1.5-2 is a broad optimum; <=1 is too weak for chameleon).
                            # Costs ~7-10% iteration time: the sampling itself is free,
                            # but spread-out Gaussians grow larger and AABB cost goes
                            # as (scale*m)**3.
                            logits = args.densify_alpha * torch.log(err_flat + 1e-12)
                            u = torch.rand_like(err_flat).clamp_min(1e-20)
                            gumbel = -torch.log(-torch.log(u))
                            loss_idx = torch.topk(logits + gumbel, k).indices
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            min_weight,
                            torch.mean(gaussians.get_scaling) / 6.0,
                            0.01,
                            samples_cuda[loss_idx],
                            gt.ravel()[loss_idx].reshape(-1, 1),
                            iteration > opt.densify_until_iter,
                            k,
                            args.densify_3dgs
                        )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                if args.ema > 0 and iteration >= args.ema_from:
                    cur = {n: getattr(gaussians, n) for n in ema_names}
                    if (ema_params is None
                            or ema_params["_xyz"].shape[0] != cur["_xyz"].shape[0]):
                        # First use. A shape mismatch here means a row mutation was not
                        # mirrored onto the buffers (see GaussianModel._ema_prune /
                        # _ema_extend / _ema_reset); restarting is safe but silently
                        # shortens the window, so it is worth knowing about.
                        if ema_params is not None:
                            print(f"[ema] buffer desync at iter {iteration}: "
                                  f"{ema_params['_xyz'].shape[0]} -> {cur['_xyz'].shape[0]}; "
                                  f"restarting the average")
                        ema_params = {n: cur[n].detach().clone() for n in ema_names}
                        gaussians.ema_buffers = ema_params
                    else:
                        with torch.no_grad():
                            # One fused kernel for all five tensors: the per-group
                            # mul_/add_ pair was 10 launches and cost ~0.3s over a
                            # 4000-iteration run, which matters at the time margins
                            # these models are scored on.
                            torch._foreach_lerp_(
                                [ema_params[n] for n in ema_names],
                                [cur[n].detach() for n in ema_names],
                                1.0 - args.ema)
                    ema_params = gaussians.ema_buffers

                # if gaussians.get_values.shape[0] > args.cap_max and iteration % 10 == 0:
                if use_mcmc:
                    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)

                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))
                    
                    noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_weight)) * args.noise_lr * xyz_lr
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)

            # Save
            if iteration in saving_iterations or done == 1:
                torch.cuda.synchronize()
                _save_t0 = time.perf_counter()
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                if ema_params is not None:
                    # Ship the average, not the last iterate. Swap in-place so a mid-run
                    # save leaves training state untouched afterwards.
                    _bak = {n: getattr(gaussians, n).detach().clone() for n in ema_names}
                    with torch.no_grad():
                        for n in ema_names:
                            getattr(gaussians, n).copy_(ema_params[n])
                scene.save(iteration)
                if ema_params is not None:
                    with torch.no_grad():
                        for n in ema_names:
                            getattr(gaussians, n).copy_(_bak[n])
                cpu_cells = cells.cpu().numpy()
                # tensor_to_vtk(cpu_cells, f"out_vtk/test_{iteration}.vtk", spacing)
                # tensor_to_vtk(torch.abs((cells - gt)).cpu().numpy(), f"out_vtk/test_{iteration}_loss.vtk", spacing)
                vtk_files.append({
                    "name": f"test_{iteration}.vtk",
                    "time": float(iteration)
                })                
                vtk_files_loss.append({
                    "name": f"test_{iteration}_loss.vtk",
                    "time": float(iteration)
                })
                torch.cuda.synchronize()
                save_seconds += time.perf_counter() - _save_t0

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    os.path.join(scene.model_path, "/chkpnt{iteration}.pth"),
                )

    # series = {
    #     "file-series-version": "1.0",
    #     "files": vtk_files
    # }
    # with open("out_vtk/test.vtk.series", "w") as jf:
    #     json.dump(series, jf, indent=2)

    # series_loss = {
    #     "file-series-version": "1.0",
    #     "files": vtk_files_loss
    # }
    # with open("out_vtk/test_loss.vtk.series", "w") as jf:
    #     json.dump(series_loss, jf, indent=2)

    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_t0 - save_seconds
    print(f"TRAIN_SECONDS: {train_seconds:.2f} (save {save_seconds:.2f})", flush=True)
    with open(os.path.join(scene.model_path, "train_seconds.txt"), "w") as f:
        f.write(f"{train_seconds:.3f}\n")

    if log_to_file:
        log_file_path = os.path.join(scene.model_path, 'training_log.json')
        with open(log_file_path, 'w') as log_file:
            json.dump(log_data, log_file, indent=4)



def prepare_output(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

if __name__ == "__main__":
    window = create_window()
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--min_weight", type=float, default=0.005)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--is_scaled", action="store_true")
    # parser.add_argument(
    #     "--test_iterations", nargs="+", type=int, default=[i * 1000 for i in range(20)]
    # )
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[]
    )
    # parser.add_argument(
    #     "--save_iterations", nargs="+", type=int, default=[1, 16, 32, 64, 125, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000]
    # )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log_to_file", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--precomputed_samples", action="store_true")
    # Restore the old target: probe.sample on a continuous lattice at one of 100 fixed
    # jitters, instead of native node values (see NodeSampler). Worth it only when the
    # model is for rendering: it is up to 1.5 dB better at continuous positions (6 of the
    # 9 configs tested, the rest a wash) and up to 1.9 dB worse against the raw voxel
    # array, and it spends 41-55 s probing at startup.
    parser.add_argument("--probe_gt", action="store_true")
    parser.add_argument("--jitter_gt", type=int, default=0,
        help="precompute this many continuous-jittered GT sets (trilinear interpolant at "
             "subgrid nodes + U(-.5,.5)-cell jitter, fresh integer offset per set) and "
             "train on those instead of native nodes; 0 = plain node sampling")
    parser.add_argument("--use_mcmc", action="store_true")
    parser.add_argument("--densify_3dgs", action="store_true",
        help="place new Gaussians by 3DGS clone/split on the position gradient instead "
             "of seeding at high-error cells")
    # Default 0 = auto: spread the budget over --densify_events events instead of a
    # fixed count per event. A fixed 80000 silently became all-at-once at high
    # compression -- every config with a total budget under 80k (all three 1024x, and
    # vertebra 256x) placed every Gaussian it would ever have in a single event at
    # iteration 500, siting them from the error map of a 500-iteration model. Pass a
    # positive value to restore the old fixed-batch behaviour.
    parser.add_argument("--densify_batch", type=int, default=0)
    parser.add_argument("--max_scale", type=float, default=0.02)
    parser.add_argument("--densify_events", type=int, default=3)
    parser.add_argument("--densify_alpha", type=float, default=1.5)
    parser.add_argument("--ema", type=float, default=0.0,
                        help="Polyak-average the parameters and SHIP THE AVERAGE: keep an "
                             "EMA of all five parameter groups with this decay (0 = off, "
                             "try 0.999) and write it, not the last iterate, at save time. "
                             "The saved model is still exactly 12 floats/Gaussian; the EMA "
                             "is a transient training buffer. Targets late-training Adam "
                             "noise: at 64x compression each Gaussian sees ~1.4 cells per "
                             "iteration, and the 8000-iter curve DECLINES past 6000.")
    parser.add_argument("--ema_from", type=int, default=1000,
                        help="start the EMA here (after densification placement ends, so "
                             "the buffers never need remapping). Restarted automatically "
                             "if the Gaussian count changes after it began.")
    parser.add_argument("--loss", type=str, default="l2", choices=["l1","l2"])
    parser.add_argument("--fn_mode", type=str, default="abs", choices=["abs", "rel"],
                        help="abs: fn_reg is an absolute weight. rel: fn_reg multiplies "
                             "the data loss.")
    parser.add_argument("--fn_debug", type=int, default=0,
                        help="print dead/near-cutoff cell counts every N iterations (0 = off)")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.log_to_file,
        args.fraction,
        args.min_weight,
        args.is_scaled,
        args.precomputed_samples,
        args.use_mcmc
    )

    # All done
    print("\nTraining complete.")
