"""Fit Gaussians to a volume mesh's interior field, or with --encode_surface to its interior and
exterior (a sample outside the mesh must render empty)."""

import math
import os
import sys
import time
import uuid
from argparse import ArgumentParser

import numpy as np
import pyvista as pv
import torch
from tqdm import tqdm

from bvh_diff_gaussian_rasterization.fused import fused_volume_loss
from gaussian_renderer import (
    build_bvh,
    build_bvh_cached,
    INTERIOR_RULE,
    SURFACE_RULE,
    init_rasterizer,
    margin_fp,
    margin_set_exterior,
    morton3d_unit,
    render,
    sample_bvh_prim_ids,
)
from gvcommon import densify, model_files
from gvcommon.arguments import ModelParams, OptimizationParams
from gvcommon.general_utils import build_scaling_rotation, safe_state
from sampling import sample_exterior_points, sample_mesh_points
from scene import GaussianModel, read_mesh

# Fixed seed for the sample points, so the cached ground truth matches every run's points.
SAMPLE_SEED = 20260903
BATCH_SIZE = 128**3
NUM_BATCHES = 100
# The rasterizer's per-Gaussian support threshold.
SUPPORT_T = 1e-4
# Gaussians under this weight count as dead when sizing a densification event.
DEAD_WEIGHT = 0.005

# Per-mode defaults for flags left unset.
# Interior: the log barrier keeps every sample above the render threshold (0.002) with margin.
INTERIOR_DEFAULTS = dict(fn_mode="log", fn_reg=0.0001, fp_reg=0.0)
# Surface: FN/FP hinges around the render threshold, scaled by the data loss (MAE), and L2
# values with the interior's light log barrier (fn_thresh).
SURFACE_DEFAULTS = dict(
    fn_mode="abs",
    fn_reg=1200.0,
    fp_reg=12.0,
    hinge_rel_pow=1.0,
    fp_mode="hinge",
    fp_margin=0.2,
    fn_margin=0.05,
    fnfp_norm="all",
    interior_skin=0.1,
)


def sample_cache_path(cache_dir, source_path, encode_surface, interior_skin):
    """Path of the cached ground truth, or None if caching is disabled."""
    if not cache_dir:
        return None
    stem = os.path.splitext(os.path.basename(source_path))[0]
    tag = f"{stem}_c128_b{NUM_BATCHES}_s{int(bool(encode_surface))}"
    if encode_surface and interior_skin > 0:
        tag += f"_skin{interior_skin:g}"
    return os.path.join(cache_dir, tag + "_gt.npy")


def load_samples(args, mesh):
    """(gt, samples) as (B, S) and (B, S, 3) arrays; gt is -1 outside the mesh."""
    if args.precomputed_samples:
        # <prefix>_gt.npy and <prefix>_samples.npy, for meshes too large to triangulate.
        big_gt = np.load(args.precomputed_samples + "_gt.npy")
        big_samples = np.load(args.precomputed_samples + "_samples.npy")
        assert big_samples.shape[:2] == big_gt.shape
        print(
            f"Loaded precomputed samples {big_gt.shape} from {args.precomputed_samples}"
        )
        return big_gt, big_samples

    # Drawing the points takes ~1 s; probing them ~9 CPU-minutes, so only the GT is cached.
    size = BATCH_SIZE
    cache_g = sample_cache_path(
        args.sample_cache_dir, args.source_path, args.encode_surface, args.interior_skin
    )
    torch.manual_seed(SAMPLE_SEED)
    if args.encode_surface:
        # 20% just outside the surface and --interior_skin just inside it, so both sides of
        # the boundary are sampled at the same density per unit area.
        size2 = int(math.floor(size * 0.2))
        size3 = int(math.floor(size * args.interior_skin))
        size1 = size - size2 - size3
        parts = [
            sample_mesh_points(mesh, NUM_BATCHES, size1).cpu().numpy(),
            sample_exterior_points(mesh, NUM_BATCHES, size2).cpu().numpy(),
        ]
        if size3 > 0:
            parts.append(
                sample_exterior_points(mesh, NUM_BATCHES, size3, inward=True)
                .cpu()
                .numpy()
            )
        big_samples = np.concatenate(parts, axis=1)
    else:
        big_samples = sample_mesh_points(mesh, NUM_BATCHES, size).cpu().numpy()

    if cache_g is not None and os.path.exists(cache_g):
        print(f"Loaded cached ground truth from {cache_g}")
        return np.load(cache_g), big_samples.reshape(NUM_BATCHES, size, 3)

    big_samples = big_samples.reshape(NUM_BATCHES * size, 3)
    probed = pv.PolyData(big_samples).sample(mesh)
    big_gt = probed[mesh.array_names[0]]
    big_gt[~probed["vtkValidPointMask"].astype(bool)] = -1
    big_gt = big_gt.reshape(NUM_BATCHES, size)
    big_samples = big_samples.reshape(NUM_BATCHES, size, 3)
    if cache_g is not None:
        os.makedirs(os.path.dirname(cache_g), exist_ok=True)
        tmp = cache_g + f".tmp{os.getpid()}.npy"
        np.save(tmp, np.ascontiguousarray(big_gt, dtype=np.float32))
        os.replace(tmp, cache_g)
        print(f"Wrote ground-truth cache to {cache_g}")
    return big_gt, big_samples


def drop_exterior_samples(big_gt, big_samples):
    """Replace -1 targets with other in-mesh samples of the same batch (interior models)."""
    bad = big_gt == -1
    if not bad.any():
        return big_gt, big_samples
    rng = np.random.default_rng(SAMPLE_SEED)
    big_gt = np.array(big_gt, copy=True)
    big_samples = np.array(big_samples, copy=True)
    for b in range(big_gt.shape[0]):
        idx = np.flatnonzero(bad[b])
        if idx.size == 0:
            continue
        good = np.flatnonzero(~bad[b])
        repl = rng.choice(good, size=idx.size, replace=False)
        big_gt[b, idx] = big_gt[b, repl]
        big_samples[b, idx] = big_samples[b, repl]
    print(f"replaced {int(bad.sum())} out-of-mesh samples with in-mesh ones")
    return big_gt, big_samples


def training(dataset, opt, args):
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), "w") as f:
        f.write(str(args))
    n_final = args.cap_max
    surface = args.encode_surface

    gaussians = GaussianModel()
    caps = [c for c in (args.init_share_max * n_final, args.init_max) if c > 0]
    mesh, pcd = read_mesh(
        dataset.source_path, args.fraction, min(caps) if caps else None
    )
    gaussians.create_from_pcd(pcd, mesh)
    # The cap grows with the Gaussian spacing (V / N)^(1/3), V the mesh's bounding box.
    args.max_scale = model_files.size_cap(
        args.max_scale, args.max_scale_k, gaussians.mins, gaussians.maxes, n_final
    )
    # Rendering a model needs the max_scale it was trained with.
    model_files.write_max_scale(dataset.model_path, args.max_scale)
    gaussians.max_scale = args.max_scale
    gaussians.training_setup(opt)

    densify_events = 3
    if args.densify_growth > 1.0:
        # Each event grows the model by about densify_growth, finishing by half-way.
        n0 = max(1, gaussians.get_xyz.shape[0])
        densify_events, _ = densify.event_count(
            n0,
            n_final,
            opt.iterations,
            opt.densify_from_iter,
            opt.densification_interval,
            args.densify_growth,
        )
        print(f"densify_events {densify_events} ({n0:,} -> {n_final:,} Gaussians)")

    big_gt, big_samples = load_samples(args, mesh)
    if not surface:
        big_gt, big_samples = drop_exterior_samples(big_gt, big_samples)
    num_batches, size = big_gt.shape
    big_gt_cuda = torch.tensor(big_gt, dtype=torch.float, device="cuda")
    big_samples_cuda = torch.tensor(big_samples, dtype=torch.float, device="cuda")

    # Morton-sort each batch: BVH builds and subsampling rely on the locality.
    morton = morton3d_unit(
        torch.clamp(big_samples_cuda, 0.0, 1.0).reshape(-1, 3)
    ).reshape(num_batches, size)
    order = morton.argsort(dim=1)
    big_samples_cuda = big_samples_cuda.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))
    big_gt_cuda = big_gt_cuda.gather(1, order)

    # Every model trains with the soft render (empty only where aw == 0) and is deployed under
    # its own rule, which is stored with it.
    rule = SURFACE_RULE if surface else INTERIOR_RULE
    occ_t = rule[0] if args.occ_t is None else args.occ_t
    floor = rule[1] if args.denom_floor is None else args.denom_floor
    model_files.write_render_rule(dataset.model_path, occ_t, floor)
    init_rasterizer(0.0, floor)
    if args.samples_per_node > 0:
        # About this many samples per mesh node per iteration, at most the whole batch.
        args.sample_frac = min(1.0, args.samples_per_node * int(mesh.n_points) / size)
        print(
            f"samples_per_node {args.samples_per_node:g}: sample_frac {args.sample_frac:.4f}"
        )
    full_batch = not (0.0 < args.sample_frac < 1.0)

    # Permute each batch into its sample BVH's leaf order, so render loads are near-sequential;
    # full-batch runs also keep each batch's BVH.
    for b in range(num_batches):
        build_bvh(big_samples_cuda[b])
        perm = sample_bvh_prim_ids().long()
        big_samples_cuda[b] = big_samples_cuda[b][perm]
        big_gt_cuda[b] = big_gt_cuda[b][perm]
        if full_batch:
            build_bvh_cached(big_samples_cuda[b], b)
    torch.cuda.synchronize()
    build_bvh(big_samples_cuda[0])

    margin_n_ext = None
    if surface and args.fp_mode == "margin":
        margin_n_ext = []
        for b in range(num_batches):
            ext = big_samples_cuda[b][big_gt_cuda[b] == -1]
            margin_set_exterior(ext, b)
            margin_n_ext.append(max(1, int(ext.shape[0])))

    sub_k, sub_stride, sub_base = 0, 1, None
    if not full_batch:
        if args.sample_mode == "stratified":
            # One draw from each run of sub_stride: ascending by construction, no gaps.
            sub_stride = max(2, int(round(1.0 / args.sample_frac)))
            sub_k = size // sub_stride
            sub_base = torch.arange(sub_k, device="cuda") * sub_stride
        else:
            sub_k = max(1, int(round(size * args.sample_frac)))
        print(f"rendering {sub_k} of {size} samples per iteration ({args.sample_mode})")

    if args.fn_mode == "log":
        fn_tau = args.fn_thresh
    else:
        fn_tau = ((1.0 + args.fn_margin) if surface else 1.1) * occ_t
    fp_t = (1.0 - args.fp_margin) * occ_t if args.fp_mode == "hinge" else SUPPORT_T
    fp_on = surface and args.fp_mode != "margin"

    torch.cuda.synchronize()
    train_t0 = time.perf_counter()
    save_seconds = 0.0
    hinge_frozen, hinge_lv_sum, hinge_lv_n = None, 0.0, 0
    densifies = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):
        jit_idx = np.random.randint(0, num_batches)
        gt = big_gt_cuda[jit_idx]
        current_samples = big_samples_cuda[jit_idx]
        if sub_k > 0:
            # Indices stay ascending, as the BVH build expects.
            if sub_base is not None:
                sub_idx = sub_base + torch.randint(
                    0, sub_stride, (sub_k,), device="cuda"
                )
            else:
                sub_idx = torch.randint(0, size, (sub_k,), device="cuda").sort().values
            gt = gt[sub_idx]
            current_samples = current_samples[sub_idx]
            build_bvh(current_samples)
        else:
            build_bvh_cached(current_samples, jit_idx)
        xyz_lr = gaussians.update_learning_rate(iteration)

        render_pkg = render(gaussians)
        cells, weights = render_pkg["cells"], render_pkg["weights"]

        # --hinge_rel_pow scales the hinges by the data loss; --hinge_freeze fixes that scale
        # at the mean data loss of the 50 iterations before it.
        hw_fn, hw_fp, hw_rel = args.fn_reg, args.fp_reg, args.hinge_rel_pow
        if hinge_frozen is not None:
            hw_fn, hw_fp, hw_rel = (
                args.fn_reg * hinge_frozen,
                args.fp_reg * hinge_frozen,
                0.0,
            )
        loss, l_v, false_negative, false_positive = fused_volume_loss(
            cells,
            gt,
            weights,
            l1=surface,
            fn_mode=args.fn_mode,
            fn_thresh=fn_tau,
            fn_reg=hw_fn,
            fp_reg=hw_fp if fp_on else 0.0,
            fp_t=fp_t,
            fp_mode=args.fp_mode if fp_on else "lin",
            norm_all=args.fnfp_norm == "all",
            rel_pow=hw_rel,
        )
        # The hinges' data-loss scale (L1 on surface models); l_v may become the L2 term below.
        l_v_hinge = l_v
        if surface:
            # Surface values are fit with L2 and the interior's light log barrier; the call
            # above only supplies the hinges.
            lv_out = fused_volume_loss(
                cells,
                gt,
                weights,
                l1=False,
                fn_mode="log",
                fn_thresh=args.fn_thresh,
                fn_reg=0.0001,
            )
            l_v = lv_out[1]
            loss = lv_out[0] + false_negative + false_positive
        if args.hinge_freeze > 0 and args.hinge_rel_pow > 0 and hinge_frozen is None:
            if iteration > args.hinge_freeze - 50:
                hinge_lv_sum = hinge_lv_sum + l_v_hinge.detach()
                hinge_lv_n += 1
            if iteration >= args.hinge_freeze:
                hinge_frozen = (
                    float(hinge_lv_sum / max(hinge_lv_n, 1)) ** args.hinge_rel_pow
                )
                print(
                    f"hinge weights frozen at iteration {iteration}: scale {hinge_frozen:.3e}"
                )
        if margin_n_ext is not None:
            false_positive = (
                args.fp_reg
                * margin_fp(gaussians, jit_idx, fp_t)
                / margin_n_ext[jit_idx]
            )
            loss = loss + false_positive
        if args.use_mcmc:
            loss = loss + opt.weight_reg * torch.abs(gaussians.get_weight).mean()
            loss = loss + opt.scale_reg * torch.abs(gaussians.get_scaling).mean()
        loss.backward()

        with torch.no_grad():
            if iteration % 250 == 0:
                live = torch.logical_and(cells != -1, gt != -1)
                mse = (
                    ((cells.double() - gt.double()) ** 2) * live
                ).sum() / live.sum().clamp_min(1)
                progress_bar.set_postfix({"Loss": f"{loss.item():.5f}"})
                progress_bar.update(250)
                print(
                    f"Num Gaussians: {gaussians.get_xyz.shape[0]}, "
                    f"psnr: {float(-10 * torch.log10(mse + 1e-16)):.3f}, "
                    f"fn {float(false_negative):.3e}, fp {float(false_positive):.3e}"
                )
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in args.save_iterations:
                torch.cuda.synchronize()
                save_t0 = time.perf_counter()
                print(f"\n[ITER {iteration}] Saving Gaussians")
                # Ship the parameter average, not the last iterate.
                gaussians.save_ply(
                    os.path.join(
                        dataset.model_path,
                        f"point_cloud/iteration_{iteration}/point_cloud.ply",
                    ),
                    ema=gaussians.ema_buffers,
                )
                torch.cuda.synchronize()
                save_seconds += time.perf_counter() - save_t0

            if (
                opt.densify_from_iter <= iteration <= opt.densify_until_iter
                and iteration % opt.densification_interval == 0
                and iteration not in args.save_iterations
            ):
                budget = densify.budget(gaussians, n_final, DEAD_WEIGHT)
                remaining = max(1, densify_events - densifies)
                if args.use_mcmc:
                    gaussians.relocate_gs(
                        (gaussians.get_weight < DEAD_WEIGHT).squeeze(-1)
                    )
                    gaussians.add_new_gs(cap_max=args.cap_max)
                elif args.densify_3dgs:
                    # Clone/split by position-gradient magnitude, threshold 0: 3DGS's image-space
                    # threshold never fires here.
                    k3 = max(0, densify.share(budget, remaining))
                    g = gaussians._xyz.grad
                    if k3 > 0 and g is not None:
                        n_cl = gaussians.densify_and_clone(
                            g, 0.0, 1.0, max_new_points=k3
                        )
                        left = max(0, k3 - int(n_cl or 0))
                        if left > 0:
                            # Cloning replaced _xyz, so reuse the gradient captured before it.
                            gaussians.densify_and_split(
                                g, 0.0, 1.0, max_new_points=left
                            )
                        densifies += 1
                        gaussians.prune_points(
                            (gaussians.get_weight < DEAD_WEIGHT).squeeze(-1)
                        )
                else:
                    mask = gt.ravel() != -1
                    diff = torch.abs(cells.ravel()[mask] - gt.ravel()[mask])
                    k = max(0, min(densify.share(budget, remaining), diff.numel()))
                    if k > 0:
                        pick = densify.pick(diff, k, args.densify_alpha)
                        loss_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)[pick]
                        gaussians.densify_in_place(
                            current_samples[loss_idx],
                            gt[loss_idx].reshape(-1, 1),
                            args.min_weight,
                        )
                densifies += 1

            if iteration < opt.iterations:
                gaussians.step(args.ema, args.ema > 0 and iteration >= args.ema_from)

                if args.use_mcmc:
                    L = build_scaling_rotation(
                        gaussians.get_scaling, gaussians.get_rotation
                    )
                    actual_covariance = L @ L.transpose(1, 2)
                    gate = 1 / (
                        1 + torch.exp(-100 * ((1 - gaussians.get_weight) - 0.995))
                    )
                    noise = (
                        torch.randn_like(gaussians._xyz) * gate * opt.noise_lr * xyz_lr
                    )
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(
                        -1
                    )
                    gaussians._xyz.add_(noise)

    torch.cuda.synchronize()
    model_files.write_train_seconds(
        dataset.model_path, time.perf_counter() - train_t0 - save_seconds, save_seconds
    )


def parse_args(argv):
    parser = ArgumentParser(description=__doc__)
    lp = ModelParams(parser)
    # This pipeline's densification window and MCMC weight term.
    op = OptimizationParams(parser, densify_until_iter=16_000, weight_reg=0.00001)
    parser.add_argument(
        "--fraction", type=float, default=1.0, help="share of mesh nodes seeded"
    )
    parser.add_argument(
        "--init_share_max", type=float, default=0.5, help="seed cap, x final count"
    )
    parser.add_argument("--init_max", type=int, default=20000, help="absolute seed cap")
    parser.add_argument("--min_weight", type=float, default=0.005)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--precomputed_samples",
        default="",
        help="prefix of <prefix>_gt.npy / _samples.npy",
    )
    parser.add_argument(
        "--sample_cache_dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "sample_cache"
        ),
        help="ground-truth cache; '' disables",
    )
    parser.add_argument("--encode_surface", action="store_true")
    parser.add_argument(
        "--interior_skin", type=float, help="surface: share sampled just inside"
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="share of each batch rendered per step (overrides --samples_per_node)",
    )
    parser.add_argument(
        "--samples_per_node",
        type=float,
        default=0.25,
        help="render about this many samples per mesh node per step",
    )
    parser.add_argument(
        "--sample_mode", default="stratified", choices=["stratified", "random"]
    )
    parser.add_argument("--fn_mode", choices=["abs", "rel", "log"])
    parser.add_argument(
        "--fn_thresh",
        type=float,
        default=0.0025,
        help="log barrier threshold (both modes; the structured pipeline's too)",
    )
    parser.add_argument(
        "--occ_t",
        type=float,
        help="render rule: empty where aw <= occ_t (default 0.002 interior, 0.01 surface)",
    )
    parser.add_argument(
        "--denom_floor",
        type=float,
        help="render rule: S / max(aw, floor) (default 0.01)",
    )
    parser.add_argument("--fn_reg", type=float)
    parser.add_argument("--fn_margin", type=float, default=0.05)
    parser.add_argument("--fp_reg", type=float)
    parser.add_argument(
        "--fp_mode", default="lin", choices=["lin", "log", "margin", "hinge"]
    )
    parser.add_argument("--fp_margin", type=float, default=0.05)
    parser.add_argument("--fnfp_norm", default="viol", choices=["viol", "all"])
    parser.add_argument("--hinge_rel_pow", type=float, default=0.0)
    parser.add_argument("--hinge_freeze", type=int, default=0)
    parser.add_argument(
        "--use_mcmc", action="store_true", help="MCMC relocate/add densification"
    )
    parser.add_argument("--densify_3dgs", action="store_true", help="3DGS clone/split")
    parser.add_argument(
        "--densify_alpha", type=float, default=2.5, help="<= 0: top-k by error"
    )
    parser.add_argument("--densify_growth", type=float, default=1.5)
    parser.add_argument(
        "--ema", type=float, default=0.995, help="0 ships the last iterate"
    )
    parser.add_argument(
        "--ema_from", type=int, default=1000, help="-1: last 100 iterations"
    )
    parser.add_argument("--max_scale", type=float, default=0.02)
    parser.add_argument("--max_scale_k", type=float, default=1.25)
    args = parser.parse_args(argv)

    given = {a.lstrip("-").split("=")[0] for a in argv if a.startswith("--")}
    defaults = SURFACE_DEFAULTS if args.encode_surface else INTERIOR_DEFAULTS
    for k, v in defaults.items():
        if k not in given:
            setattr(args, k, v)
    if "sample_frac" in given and "samples_per_node" not in given:
        args.samples_per_node = 0.0
    if args.interior_skin is None:
        args.interior_skin = 0.0
    if (args.use_mcmc or args.densify_3dgs) and args.ema > 0:
        print(
            "--use_mcmc/--densify_3dgs do not keep the EMA rows in step; using --ema 0"
        )
        args.ema = 0.0
    if args.ema_from < 0:
        args.ema_from = max(1, args.iterations - 100)
    args.save_iterations.append(args.iterations)
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[:10])
    return args, lp, op


if __name__ == "__main__":
    args, lp, op = parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    training(lp.extract(args), op.extract(args), args)
    print("\nTraining complete.")
