"""Fit Gaussians to the scalar field of a volume. The model type follows the input:

  structured  a regular grid (.vtk structured points): trained on jittered lattices over the
              grid with continuous ground truth, rendered by the tile rasterizer
  interior    a volume mesh (.vtu): trained on points sampled inside the mesh, rendered at
              them through a BVH
  surface     --encode_surface: a mesh's interior and its boundary, points just outside the
              mesh rendering empty

--kind overrides the choice (for example a .vtk grid trained as a mesh). Every type shares the
model, the schedule, the EMA and densification; the defaults that differ by type are in
KIND_DEFAULTS and RULES.
"""

import os
import sys
import time
import uuid
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams
from gaussian_renderer import RULES, init_rasterizer, margin_fp
from gvraster.ops import volume_loss
from scene import GaussianModel, densify, model_files, read_mesh, read_structured
from scene.samples import GridSamples, MeshSamples
from utils.general_utils import build_scaling_rotation, safe_state

KINDS = ("structured", "interior", "surface")
# The rasterizer's per-Gaussian support threshold.
SUPPORT_T = 1e-4

# Per-type defaults for flags left unset.
KIND_DEFAULTS = {
    # Seeds: 1e-5 of the grid nodes; L2 on every lattice cell plus the log FN barrier.
    "structured": dict(
        fraction=0.00001,
        init_share_max=0.25,
        fn_mode="log",
        fn_reg=0.0001,
        fp_reg=0.0,
        weight_reg=0.01,
        densify_grad_threshold=0.0002,
    ),
    # Seeds: every mesh node; the log barrier keeps samples above the render threshold.
    "interior": dict(
        fraction=1.0, init_share_max=0.5, fn_mode="log", fn_reg=0.0001, fp_reg=0.0
    ),
    # FN/FP hinges around the render threshold, scaled by the data loss (MAE), and L2 values
    # with the interior's light log barrier (fn_thresh).
    "surface": dict(
        fraction=1.0,
        init_share_max=0.5,
        fn_mode="abs",
        fn_reg=1200.0,
        fp_reg=12.0,
        hinge_rel_pow=1.0,
        fp_mode="hinge",
        fp_margin=0.2,
        fn_margin=0.05,
        fnfp_norm="all",
        interior_skin=0.1,
    ),
}
# Flags that only mean something for grids or for meshes.
GRID_ONLY = ("is_scaled", "jitter_gt", "probe_gt")
MESH_ONLY = (
    "precomputed_samples",
    "sample_cache_dir",
    "sample_frac",
    "samples_per_node",
    "sample_mode",
)


def training(dataset, opt, args):
    kind = args.kind
    surface = kind == "surface"
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), "w") as f:
        f.write(str(args))
    n_final = args.cap_max

    gaussians = GaussianModel()
    caps = [c for c in (args.init_share_max * n_final, args.init_max) if c > 0]
    max_init = min(caps) if caps else None
    if kind == "structured":
        mesh, pcd = read_structured(
            dataset.source_path, args.fraction, args.is_scaled, max_init
        )
    else:
        mesh, pcd = read_mesh(dataset.source_path, args.fraction, max_init)
    gaussians.create_from_pcd(pcd, mesh)
    # The cap grows with the Gaussian spacing (V / N)^(1/3), V the volume's bounding box.
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

    # Every model trains with the soft render (empty only where aw == 0) and is deployed under
    # its own rule, which is stored with it.
    occ_t = RULES[kind][0] if args.occ_t is None else args.occ_t
    floor = RULES[kind][1] if args.denom_floor is None else args.denom_floor
    model_files.write_render_rule(dataset.model_path, occ_t, floor)
    init_rasterizer(0.0, floor)
    if kind == "structured":
        data = GridSamples(args, gaussians)
    else:
        data = MeshSamples(args, mesh, surface)

    if args.fn_mode == "log":
        fn_tau = args.fn_thresh
    else:
        fn_tau = ((1.0 + args.fn_margin) if surface else 1.1) * occ_t
    fp_t = (1.0 - args.fp_margin) * occ_t if args.fp_mode == "hinge" else SUPPORT_T
    fp_on = surface and args.fp_mode != "margin"
    # Grid models fit every lattice cell, an empty one's -1 included; mesh models skip samples
    # without ground truth or coverage.
    mask_data = kind != "structured"

    torch.cuda.synchronize()
    train_t0 = time.perf_counter()
    save_seconds = 0.0
    hinge_frozen, hinge_lv_sum, hinge_lv_n = None, 0.0, 0
    densifies = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    for iteration in range(1, opt.iterations + 1):
        gt = data.next(gaussians)
        xyz_lr = gaussians.update_learning_rate(iteration)
        cells, weights = data.render(gaussians)

        # --hinge_rel_pow scales the hinges by the data loss; --hinge_freeze fixes that scale
        # at the mean data loss of the 50 iterations before it.
        hw_fn, hw_fp, hw_rel = args.fn_reg, args.fp_reg, args.hinge_rel_pow
        if hinge_frozen is not None:
            hw_fn, hw_fp, hw_rel = (
                args.fn_reg * hinge_frozen,
                args.fp_reg * hinge_frozen,
                0.0,
            )
        loss, l_v, false_negative, false_positive = volume_loss(
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
            mask_data=mask_data,
        )
        # The hinges' data-loss scale (L1 on surface models); l_v may become the L2 term below.
        l_v_hinge = l_v
        if surface:
            # Surface values are fit with L2 and the interior's light log barrier; the call
            # above only supplies the hinges.
            lv_out = volume_loss(
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
        if surface and data.margin_n_ext is not None:
            false_positive = (
                args.fp_reg
                * margin_fp(gaussians, data.batch, fp_t)
                / data.margin_n_ext[data.batch]
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
                budget = densify.budget(gaussians, n_final, args.min_weight)
                remaining = max(1, densify_events - densifies)
                if args.use_mcmc:
                    gaussians.relocate_gs(
                        (gaussians.get_weight < args.min_weight).squeeze(-1)
                    )
                    gaussians.add_new_gs(cap_max=args.cap_max)
                elif args.densify_3dgs:
                    # Clone/split by position-gradient magnitude (mesh models' threshold 0:
                    # 3DGS's image-space threshold never fires there), then prune the dead.
                    k3 = max(0, densify.share(budget, remaining))
                    g = gaussians._xyz.grad
                    if k3 > 0 and g is not None:
                        n_cl = gaussians.densify_and_clone(
                            g, opt.densify_grad_threshold, 1.0, max_new_points=k3
                        )
                        left = max(0, k3 - int(n_cl or 0))
                        if left > 0:
                            # Cloning replaced _xyz, so reuse the gradient captured before it.
                            gaussians.densify_and_split(
                                g, opt.densify_grad_threshold, 1.0, max_new_points=left
                            )
                    gaussians.prune_points(
                        (gaussians.get_weight < args.min_weight).squeeze(-1)
                    )
                else:
                    # New Gaussians at samples drawn ~ error^alpha among those with ground
                    # truth, taking dead Gaussians' rows.
                    mask = gt.ravel() != -1
                    diff = torch.abs(cells.ravel()[mask] - gt.ravel()[mask])
                    k = max(0, min(densify.share(budget, remaining), diff.numel()))
                    if k > 0:
                        pick = densify.pick(diff, k, args.densify_alpha)
                        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)[pick]
                        gaussians.densify_in_place(
                            data.positions(idx),
                            gt.ravel()[idx].reshape(-1, 1),
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


def vtk_node_count(path):
    """Node count of a legacy structured-points .vtk, from its text header."""
    with open(path, "rb") as f:
        for _ in range(32):
            tok = f.readline().decode("ascii", "replace").split()
            if tok and tok[0].upper() == "DIMENSIONS":
                return int(tok[1]) * int(tok[2]) * int(tok[3])
    raise SystemExit(f"{path}: no DIMENSIONS line in the header; pass --cap_max")


def parse_args(argv):
    # No abbreviations: the per-type defaults below key on the full flag names given.
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    lp = ModelParams(parser)
    op = OptimizationParams(parser)

    g = parser.add_argument_group("Model type")
    g.add_argument(
        "--kind",
        choices=KINDS,
        help="default: structured for a .vtk grid, interior for a .vtu mesh, "
        "surface with --encode_surface",
    )
    g.add_argument("--encode_surface", action="store_true", help="--kind surface")
    g.add_argument(
        "--ratio",
        type=float,
        default=0.0,
        help="grids: train to nodes / (12 * ratio) Gaussians (sets --cap_max)",
    )

    g = parser.add_argument_group("Seeds, densification and EMA")
    g.add_argument(
        "--fraction",
        type=float,
        help="share of nodes seeded (default 1 for meshes, 1e-5 for grids)",
    )
    g.add_argument(
        "--init_share_max",
        type=float,
        help="seed cap, x final count (default 0.5 meshes, 0.25 grids; 0 = none)",
    )
    g.add_argument(
        "--init_max", type=int, default=20000, help="absolute seed cap; 0 = none"
    )
    g.add_argument(
        "--min_weight", type=float, default=0.005, help="Gaussians below are dead"
    )
    g.add_argument(
        "--use_mcmc", action="store_true", help="MCMC relocate/add densification"
    )
    g.add_argument("--densify_3dgs", action="store_true", help="3DGS clone/split")
    g.add_argument(
        "--densify_alpha", type=float, default=2.5, help="<= 0: top-k by error"
    )
    g.add_argument("--densify_growth", type=float, default=1.5)
    g.add_argument("--ema", type=float, default=0.995, help="0 ships the last iterate")
    g.add_argument("--ema_from", type=int, default=1000, help="-1: last 100 iterations")
    g.add_argument("--max_scale", type=float, default=0.02)
    g.add_argument(
        "--max_scale_k",
        type=float,
        default=1.25,
        help="if > 0, max_scale = max(--max_scale, k (V / N)^(1/3)), V the bounding box",
    )

    g = parser.add_argument_group("Loss and render rule")
    g.add_argument("--fn_mode", choices=["abs", "rel", "log"])
    g.add_argument(
        "--fn_thresh",
        type=float,
        default=0.0025,
        help="log barrier threshold: accumulated weight below which a sample is pushed up",
    )
    g.add_argument("--fn_reg", type=float)
    g.add_argument(
        "--occ_t",
        type=float,
        help="render rule: empty where aw <= occ_t (default 0.002; surface 0.01)",
    )
    g.add_argument(
        "--denom_floor",
        type=float,
        help="render rule: S / max(aw, floor) (default 0.01)",
    )

    g = parser.add_argument_group("Grids (structured)")
    g.add_argument(
        "--is_scaled",
        action="store_true",
        help="the volume is already in the unit cube with values in [0, 1]",
    )
    g.add_argument(
        "--jitter_gt",
        type=int,
        default=400,
        help="train on this many precomputed jittered lattices with trilinear GT; "
        "0 = raw node values on a strided subgrid",
    )
    g.add_argument(
        "--probe_gt",
        action="store_true",
        help="GT probed on the lattice at 100 fixed jitters",
    )

    g = parser.add_argument_group("Meshes (interior and surface)")
    g.add_argument(
        "--precomputed_samples",
        default="",
        help="prefix of <prefix>_gt.npy / _samples.npy",
    )
    g.add_argument(
        "--sample_cache_dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "sample_cache"
        ),
        help="ground-truth cache; '' disables",
    )
    g.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="share of each batch rendered per step (overrides --samples_per_node)",
    )
    g.add_argument(
        "--samples_per_node",
        type=float,
        default=0.25,
        help="render about this many samples per mesh node per step",
    )
    g.add_argument(
        "--sample_mode", default="stratified", choices=["stratified", "random"]
    )

    g = parser.add_argument_group("Surface models")
    g.add_argument(
        "--interior_skin", type=float, help="share sampled just inside the surface"
    )
    g.add_argument("--fn_margin", type=float, default=0.05)
    g.add_argument("--fp_reg", type=float)
    g.add_argument(
        "--fp_mode", default="lin", choices=["lin", "log", "margin", "hinge"]
    )
    g.add_argument("--fp_margin", type=float, default=0.05)
    g.add_argument("--fnfp_norm", default="viol", choices=["viol", "all"])
    g.add_argument("--hinge_rel_pow", type=float, default=0.0)
    g.add_argument("--hinge_freeze", type=int, default=0)

    g = parser.add_argument_group("Run")
    g.add_argument("--save_iterations", nargs="+", type=int, default=[])
    g.add_argument("--quiet", action="store_true")
    g.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEED", "0")),
        help="RNG seed (default $SEED, else 0)",
    )
    args = parser.parse_args(argv)

    given = {a.lstrip("-").split("=")[0] for a in argv if a.startswith("--")}
    src = args.source_path.lower()
    if args.kind is None:
        if args.encode_surface:
            args.kind = "surface"
        else:
            args.kind = "structured" if src.endswith(".vtk") else "interior"
    elif args.encode_surface and args.kind != "surface":
        parser.error("--encode_surface trains a surface model")
    if args.kind == "structured" and not src.endswith(".vtk"):
        parser.error("structured models need a .vtk grid")
    args.encode_surface = args.kind == "surface"
    wrong = GRID_ONLY if args.kind != "structured" else MESH_ONLY
    bad = sorted(f"--{k}" for k in wrong if k in given)
    if bad:
        parser.error(f"{', '.join(bad)} do(es) not apply to {args.kind} models")
    for k, v in KIND_DEFAULTS[args.kind].items():
        if k not in given:
            setattr(args, k, v)
    if "sample_frac" in given and "samples_per_node" not in given:
        args.samples_per_node = 0.0
    if args.interior_skin is None:
        args.interior_skin = 0.0
    if args.ema_from < 0:
        args.ema_from = max(1, args.iterations - 100)
    if args.ratio > 0:
        if not src.endswith(".vtk"):
            parser.error("--ratio needs a .vtk grid; give meshes --cap_max")
        args.cap_max = vtk_node_count(args.source_path) // int(round(12 * args.ratio))
        print(f"ratio {args.ratio:g}: cap_max {args.cap_max} Gaussians")
    elif args.cap_max <= 0:
        parser.error(
            "give --cap_max (the final Gaussian count) or, for a grid, --ratio"
        )
    args.save_iterations.append(args.iterations)
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[:10])
    return args, lp, op


if __name__ == "__main__":
    args, lp, op = parse_args(sys.argv[1:])
    print("Optimizing " + args.model_path)
    safe_state(args.quiet, seed=args.seed)
    training(lp.extract(args), op.extract(args), args)
    print("\nTraining complete.")
