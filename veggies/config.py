"""Training options: the command line, and the defaults that depend on the model type."""

import os
import uuid
from argparse import SUPPRESS, ArgumentParser, RawDescriptionHelpFormatter

from veggies import volume

KINDS = ("structured", "interior", "surface")

DESCRIPTION = """Fit a VEGGIES model to the scalar field of a volume.

model types (--kind; the default follows the input):
  structured  a regular grid (.vti, legacy .vtk structured points)
  surface     a mesh, encoding its field and its domain (the paper's VEGGIES); default for meshes
  interior    a mesh, encoding its field only (the paper's VEGGIES+S): sampling it needs the mesh's
              exterior surface to tell where the field is defined

size: --ratio R trains (volume bytes) / (48 R) Gaussians, --cap_max N exactly N."""

# Defaults of options left unset, by model type.
KIND_DEFAULTS = {
    "structured": dict(
        fraction=0.00001,
        init_share_max=0.25,
        fn_mode="log",
        lambda_fn=0.0001,
        lambda_fp=0.0,
        weight_reg=0.01,
    ),
    "interior": dict(
        fraction=1.0, init_share_max=0.5, fn_mode="log", lambda_fn=0.0001, lambda_fp=0.0
    ),
    "surface": dict(
        fraction=1.0,
        init_share_max=0.5,
        fn_mode="abs",
        lambda_fn=1200.0,
        lambda_fp=12.0,
        hinge_rel_pow=1.0,
        fp_mode="hinge",
        fp_margin=0.2,
        fnfp_norm="all",
        interior_skin=0.1,
        remove_exterior_gaussians=0.5,
    ),
}
COMMON_DEFAULTS = dict(
    weight_reg=0.00001,
    hinge_rel_pow=0.0,
    fp_mode="lin",
    fp_margin=0.05,
    fnfp_norm="viol",
    interior_skin=0.0,
    densify="error",
)
GRID_ONLY = dict(is_scaled=False, jitter_gt=400, probe_gt=False)
MESH_ONLY = dict(
    precomputed_samples="",
    sample_cache_dir=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_cache"
    ),
    sample_frac=None,
    samples_per_node=None,
    sample_mode="stratified",
    importance=0.5,
    remove_exterior_gaussians=0.0,
)


def build_parser():
    p = ArgumentParser(
        description=DESCRIPTION,
        formatter_class=RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    g = p.add_argument_group("Input and output")
    g.add_argument("-s", "--source_path", required=True, help="the volume")
    g.add_argument(
        "-m", "--model_path", default="", help="output directory (default output/<id>)"
    )
    g.add_argument("--array", help="the scalar array to fit (default: the first)")
    g.add_argument("--kind", choices=KINDS)
    g.add_argument(
        "--encode_domain",
        "--encode_surface",
        dest="encode_domain",
        action="store_true",
        help="--kind surface",
    )
    g.add_argument("--ratio", type=float, default=0.0, help="compression ratio")
    g.add_argument("--cap_max", type=int, default=-1, help="final Gaussian count")

    g = p.add_argument_group("Schedule and learning rates")
    g.add_argument("--iterations", type=int, default=5_000)
    g.add_argument("--position_lr_init", type=float, default=0.00016)
    g.add_argument("--position_lr_final", type=float, default=0.0000016)
    g.add_argument("--position_lr_delay_mult", type=float, default=0.01)
    g.add_argument("--position_lr_max_steps", type=int, default=30_000)
    g.add_argument("--values_lr", type=float, default=0.005)
    g.add_argument("--weight_lr", type=float, default=0.025)
    g.add_argument("--scaling_lr", type=float, default=0.008)
    g.add_argument("--rotation_lr", type=float, default=0.005)
    g.add_argument(
        "--ema",
        type=float,
        default=0.995,
        help="parameter average decay; 0 ships the last iterate",
    )
    g.add_argument(
        "--ema_from", type=int, default=1000, help="-1: the last 100 iterations"
    )

    g = p.add_argument_group("Seeds and densification")
    g.add_argument(
        "--fraction",
        type=float,
        help="share of nodes seeded (1 for meshes, 1e-5 for grids)",
    )
    g.add_argument(
        "--init_share_max",
        type=float,
        help="seed cap, x the final count (0.5 meshes, 0.25 grids)",
    )
    g.add_argument(
        "--init_max", type=int, default=20000, help="absolute seed cap; 0 = none"
    )
    g.add_argument(
        "--densify", choices=["error", "3dgs", "mcmc"], help="strategy (default error)"
    )
    g.add_argument(
        "--use_mcmc", dest="densify", action="store_const", const="mcmc", help=SUPPRESS
    )
    g.add_argument(
        "--densify_3dgs",
        dest="densify",
        action="store_const",
        const="3dgs",
        help=SUPPRESS,
    )
    g.add_argument(
        "--densify_alpha",
        type=float,
        default=2.5,
        help="samples drawn ~ error^alpha; <= 0: top-k",
    )
    g.add_argument(
        "--densify_growth",
        type=float,
        default=1.5,
        help="growth per densification event",
    )
    g.add_argument("--densification_interval", type=int, default=100)
    g.add_argument("--densify_from_iter", type=int, default=500)
    g.add_argument("--densify_until_iter", type=int, default=16_000)
    g.add_argument(
        "--prune_threshold",
        "--min_weight",
        dest="prune_threshold",
        type=float,
        default=0.005,
        help="tau_2",
    )
    g.add_argument(
        "--densify_grad_threshold",
        type=float,
        default=0.0,
        help="3DGS clone/split position-gradient threshold",
    )
    g.add_argument(
        "--percent_dense", type=float, default=0.01, help="3DGS clone/split size limit"
    )
    g.add_argument("--noise_lr", type=float, default=5e5, help="MCMC")
    g.add_argument("--scale_reg", type=float, default=0.01, help="MCMC")
    g.add_argument("--weight_reg", type=float, help="MCMC (0.01 grids, 1e-5 meshes)")
    g.add_argument("--max_scale", type=float, default=0.02)
    g.add_argument(
        "--max_scale_k",
        type=float,
        default=1.25,
        help="if > 0, max_scale = max(--max_scale, k (V / N)^(1/3))",
    )

    g = p.add_argument_group("Loss and render rule")
    g.add_argument("--fn_mode", choices=["abs", "rel", "log"])
    g.add_argument(
        "--fn_thresh",
        type=float,
        default=0.0025,
        help="log barrier threshold on the influence",
    )
    g.add_argument("--lambda_fn", "--fn_reg", dest="lambda_fn", type=float)
    g.add_argument(
        "--defined_threshold",
        "--occ_t",
        dest="defined_threshold",
        type=float,
        help="T (0.002; surface 0.01)",
    )
    g.add_argument(
        "--floor", "--denom_floor", dest="floor", type=float, help="tau_1 (0.01)"
    )

    g = p.add_argument_group("Grids (structured)")
    g.add_argument(
        "--is_scaled",
        action="store_true",
        default=None,
        help="already in the unit cube, values in [0, 1]",
    )
    g.add_argument(
        "--jitter_gt",
        type=int,
        help="precomputed jittered lattices (400); 0 = raw nodes",
    )
    g.add_argument(
        "--probe_gt", action="store_true", default=None, help="probe 100 fixed jitters"
    )

    g = p.add_argument_group("Meshes (interior and surface)")
    g.add_argument(
        "--precomputed_samples", help="prefix of <prefix>_gt.npy / _samples.npy"
    )
    g.add_argument("--sample_cache_dir", help="ground-truth cache ('' disables)")
    g.add_argument(
        "--sample_frac", type=float, help="share of each batch sampled per step"
    )
    g.add_argument(
        "--samples_per_node", type=float, help="samples per mesh node per step (0.25)"
    )
    g.add_argument("--sample_mode", choices=["stratified", "random"])
    g.add_argument(
        "--importance",
        type=float,
        help="share of the in-mesh samples drawn with probability proportional to their tet's "
        "value range, the rest uniformly (0.5; precomputed sets carry their own)",
    )
    g.add_argument(
        "--remove_exterior_gaussians",
        type=float,
        help="share of training in which each densification event removes the Gaussians that "
        "reached no inside sample since the last, so their slots are re-placed (surface 0.5, "
        "interior 0 = off)",
    )

    g = p.add_argument_group("Surface models")
    g.add_argument(
        "--interior_skin",
        type=float,
        help="share of samples just inside the surface (0.1)",
    )
    g.add_argument("--fn_margin", type=float, default=0.05)
    g.add_argument("--lambda_fp", "--fp_reg", dest="lambda_fp", type=float)
    g.add_argument("--fp_mode", choices=["lin", "log", "margin", "hinge"])
    g.add_argument("--fp_margin", type=float)
    g.add_argument("--fnfp_norm", choices=["viol", "all"])
    g.add_argument("--hinge_rel_pow", type=float)
    g.add_argument("--hinge_freeze", type=int, default=0)

    g = p.add_argument_group("Run")
    g.add_argument("--save_iterations", nargs="+", type=int, default=[])
    g.add_argument("--quiet", action="store_true")
    g.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEED", "0")),
        help="default $SEED, else 0",
    )
    return p


def parse_args(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    grid = volume.is_grid_file(args.source_path)
    if args.kind is None:
        args.kind = "structured" if grid and not args.encode_domain else "surface"
    elif args.encode_domain and args.kind != "surface":
        parser.error("--encode_domain trains a surface model")
    if args.kind == "structured" and not grid:
        parser.error(
            "structured models need a regular grid (.vti or legacy .vtk structured points)"
        )
    args.encode_domain = args.kind == "surface"

    wrong = GRID_ONLY if args.kind != "structured" else MESH_ONLY
    given = sorted(f"--{k}" for k in wrong if getattr(args, k) is not None)
    if given:
        parser.error(f"{', '.join(given)} do(es) not apply to {args.kind} models")
    if args.sample_frac is not None and args.samples_per_node is None:
        args.samples_per_node = 0.0
    for defaults in (KIND_DEFAULTS[args.kind], COMMON_DEFAULTS, GRID_ONLY, MESH_ONLY):
        for k, v in defaults.items():
            if getattr(args, k) is None:
                setattr(args, k, v)
    if args.sample_frac is None:
        args.sample_frac = 1.0
    if args.samples_per_node is None:
        args.samples_per_node = 0.25

    if args.ema_from < 0:
        args.ema_from = max(1, args.iterations - 100)
    if args.ratio > 0:
        if grid:
            args.cap_max = volume.grid_gaussian_count(args.source_path, args.ratio)
            print(f"ratio {args.ratio:g}: cap_max {args.cap_max} Gaussians")
    elif args.cap_max <= 0:
        parser.error(
            "give --ratio (a compression ratio) or --cap_max (the final Gaussian count)"
        )
    args.save_iterations.append(args.iterations)
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[:10])
    return args


def to_argv(source, model_path, options):
    """Command-line arguments of keyword options (True for flags, lists for nargs)."""
    argv = ["-s", source, "-m", model_path]
    for k, v in options.items():
        if v is True:
            argv.append(f"--{k}")
        elif v is not False and v is not None:
            argv += [f"--{k}", *map(str, v if isinstance(v, list | tuple) else [v])]
    return argv
