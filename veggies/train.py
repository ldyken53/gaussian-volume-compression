"""Training a VEGGIES model on a volume."""

import os
import sys
import time

import torch
from tqdm import tqdm

from veggies import densify, files, volume
from veggies.config import parse_args, to_argv
from veggies.losses import TrainingLoss
from veggies.model import VeggiesModel, scale_cap
from veggies.sampling import RULES
from veggies.samples import GridSamples, MeshSamples
from veggies.utils import seed_everything, timestamp_stdout


def train(source, model_path, **options):
    """Train a model on the volume at source into model_path; options are the command-line
    options (python train.py -h), e.g. train("mesh.vtu", "out", ratio=256, kind="interior").
    """
    args = parse_args(to_argv(source, model_path, options))
    seed_everything(args.seed)
    run(args)
    return args.model_path


def run(args):
    os.makedirs(args.model_path, exist_ok=True)
    vol = volume.load(os.path.abspath(args.source_path), args.array, args.is_scaled)
    if args.cap_max <= 0:
        args.cap_max = volume.mesh_gaussian_count(vol, args.ratio)
        print(f"ratio {args.ratio:g}: cap_max {args.cap_max} Gaussians")
    files.write_cfg(args.model_path, args)
    files.write_info(args.model_path, args.kind, vol.normalization)
    n_final = args.cap_max

    model = VeggiesModel()
    caps = [c for c in (args.init_share_max * n_final, args.init_max) if c > 0]
    model.init_from_seeds(
        volume.seed_points(vol, args.fraction, min(caps) if caps else None), vol.dataset
    )
    args.max_scale = scale_cap(
        args.max_scale, args.max_scale_k, model.bounds_min, model.bounds_max, n_final
    )
    files.write_max_scale(args.model_path, args.max_scale)
    model.max_scale = args.max_scale
    model.setup_optimizer(args)
    events = densify.planned_events(model, n_final, args)

    threshold = (
        RULES[args.kind][0]
        if args.defined_threshold is None
        else args.defined_threshold
    )
    floor = RULES[args.kind][1] if args.floor is None else args.floor
    files.write_render_rule(args.model_path, threshold, floor)
    training_rule = (0.0, floor)
    if args.kind == "structured":
        data = GridSamples(args, model, training_rule)
    else:
        data = MeshSamples(args, vol.dataset, training_rule)
    loss_fn = TrainingLoss(args, threshold)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    save_seconds = 0.0
    densified = 0
    # Exterior Gaussians are removed at the densification events up to this iteration (0: off).
    remove_until = args.remove_exterior_gaussians * args.iterations
    progress = tqdm(range(args.iterations), desc="Training progress")
    for iteration in range(1, args.iterations + 1):
        removing = iteration <= remove_until
        if remove_until > 0:
            data.recording = removing
        gt = data.next(model)
        means_lr = model.update_learning_rate(iteration)
        values, influence = data.render(model)
        loss, _, fn, fp = loss_fn(values, gt, influence, model, data, iteration)
        loss.backward()

        with torch.no_grad():
            if iteration % 250 == 0:
                defined = torch.logical_and(values != -1, gt != -1)
                mse = (
                    ((values.double() - gt.double()) ** 2) * defined
                ).sum() / defined.sum().clamp_min(1)
                progress.set_postfix({"Loss": f"{loss.item():.5f}"})
                progress.update(250)
                print(
                    f"Gaussians: {model.count}, psnr: {float(-10 * torch.log10(mse + 1e-16)):.3f}, "
                    f"fn {float(fn):.3e}, fp {float(fp):.3e}"
                )
            if iteration == args.iterations:
                progress.close()

            if iteration in args.save_iterations:
                torch.cuda.synchronize()
                t_save = time.perf_counter()
                print(f"\n[ITER {iteration}] Saving Gaussians")
                path = os.path.join(
                    args.model_path,
                    f"point_cloud/iteration_{iteration}/point_cloud.ply",
                )
                model.save_ply(path, ema=model.ema)
                torch.cuda.synchronize()
                save_seconds += time.perf_counter() - t_save

            if (
                args.densify_from_iter <= iteration <= args.densify_until_iter
                and iteration % args.densification_interval == 0
                and iteration not in args.save_iterations
            ):
                exterior = data.exterior_gaussians(model, threshold) if removing else None
                densify.run_event(
                    model, args, data, values, gt, n_final, events - densified, exterior
                )
                densified += 1

            if iteration < args.iterations:
                model.step(args.ema, args.ema > 0 and iteration >= args.ema_from)
                if args.densify == "mcmc":
                    densify.mcmc_noise(model, args.noise_lr, means_lr)

    torch.cuda.synchronize()
    files.write_train_seconds(
        args.model_path, time.perf_counter() - t0 - save_seconds, save_seconds
    )


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    print("Optimizing " + args.model_path)
    timestamp_stdout(args.quiet)
    seed_everything(args.seed)
    run(args)
    print("\nTraining complete.")
