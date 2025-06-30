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
from scene import GaussianModel, Scene
from utils.debug_utils import tensor_to_vtk, analyze_array
from utils.general_utils import get_expon_lr_func, safe_state
from utils.image_utils import psnr
from utils.loss_utils import bounding_box_regularization, create_window, l1_loss, l2_loss

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

DEBUG = True


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
    min_weight
):
    log_data = []
    first_iter = 0
    prepare_output(dataset)
    gaussians = GaussianModel()
    scene = Scene(dataset, gaussians, fraction=fraction)
    gaussians.training_setup(opt)
    scene.save(0)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0

    # Make ground truth
    cell_count = 100
    spacing = [
        (gaussians.maxes[0] - gaussians.mins[0]) / (cell_count - 1),
        (gaussians.maxes[1] - gaussians.mins[1]) / (cell_count - 1),
        (gaussians.maxes[2] - gaussians.mins[2]) / (cell_count - 1)
    ]
    x = np.linspace(gaussians.mins[0], gaussians.maxes[0], cell_count)
    y = np.linspace(gaussians.mins[1], gaussians.maxes[1], cell_count)
    z = np.linspace(gaussians.mins[2], gaussians.maxes[2], cell_count)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    samples = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    samples_3d = samples.reshape(cell_count, cell_count, cell_count, 3)
    rot = np.rot90(samples_3d, k=1, axes=(2,0))
    samples_tf = np.flip(rot, axis=2)
    samples_tf_flat = samples_tf.reshape(-1, 3)
    start = time.time()
    gt_point_cloud = pv.PolyData(samples_tf_flat)
    probed = gt_point_cloud.sample(gaussians.mesh)
    end = time.time()
    print(f"Time to sample gt: {end - start}")
    gt_cells = probed.point_data['value']
    gt_cells[probed.point_data['vtkValidPointMask'] == 0] = -1.0
    gt_cells = gt_cells.reshape(cell_count, cell_count, cell_count)
    print(f"Number of invalid samples: {np.count_nonzero(probed.point_data['vtkValidPointMask'] == 0)}")
    tensor_to_vtk(gt_cells, "test_gt.vtk", spacing)
    gt = torch.tensor(gt_cells.copy()).cuda()
    gt_weights = probed.point_data['vtkValidPointMask'].astype(np.float32).copy().reshape(cell_count, cell_count, cell_count)
    tensor_to_vtk(gt_weights, "test_gt_weight.vtk", spacing)
    gt_weights = torch.tensor(gt_weights).cuda()

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # if iteration % 1000 == 0:
        #     start = time.time()
        #     jitter = np.random.uniform(-0.5, 0.5, samples_tf.shape)
        #     for i in range(3):
        #         jitter[:,i] *= spacing[2 - i]
        #     samples_tf_flat = (samples_tf + jitter).reshape(-1, 3)
        #     gt_point_cloud = pv.PolyData(samples_tf_flat)
        #     probed = gt_point_cloud.sample(gaussians.mesh)
        #     gt_cells = probed.point_data['value']
        #     gt_cells[probed.point_data['vtkValidPointMask'] == 0] = -1.0
        #     gt_cells = gt_cells.reshape(cell_count, cell_count, cell_count)
        #     gt = torch.tensor(gt_cells.copy()).cuda()
        #     gt_weights = probed.point_data['vtkValidPointMask'].astype(np.float32).copy().reshape(cell_count, cell_count, cell_count)
        #     gt_weights = torch.tensor(gt_weights).cuda()
        #     end = time.time()
        #     print(f"Jitter time: {end - start}")

        gaussians.update_learning_rate(iteration)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(
            gaussians,
            pipe,
            cell_count
        )
        cells, weights, visibility_filter, radii = (
            render_pkg["cells"],
            render_pkg["weights"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        l1_lv = l1_loss(cells[gt != -1], gt[gt != -1])
        k = 10  # Adjust this to control decay rate
        false_negative = torch.exp(-k * weights) * gt_weights
        # lw = ((1 - torch.exp(-k * weights)) * (1 - gt_weights) + false_negative).mean()
        lw = l1_loss(weights[gt == -1 ], gt_weights[gt == -1])
        loss = l1_lv
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Logging
            if log_to_file and iteration % 20 == 0:
                cpu_cells = cells.cpu().numpy()
                mse = torch.mean((cells - gt) ** 2)
                psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
                mse2 = torch.mean((cells[torch.logical_and(cells != -1, gt != -1)] - gt[torch.logical_and(cells != -1, gt != -1)]) ** 2)
                psnr2 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse2 + 1e-8)
                num_gaussians = gaussians.get_values.shape[0]
                log_data.append({
                    "iteration": iteration,
                    "loss": loss.item(),
                    "l_v": l1_lv,
                    "l_w": lw,
                    "psnr": psnr.item(),
                    "psnr2": psnr2.item(),
                    "num_gaussians": num_gaussians
                })
            
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 500 == 0:
                mse = torch.mean((cells - gt) ** 2)
                psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "L_v": f"{l1_lv:.{5}f}",
                        "L_w": f"{lw:.{5}f}",
                        "PSNR": f"{psnr:.{5}f}"
                    }
                )
                progress_bar.update(500)
                print("")
            if iteration == opt.iterations:
                progress_bar.close()

            # Save
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                cpu_cells = cells.cpu().numpy()
                tensor_to_vtk(cpu_cells, f"test_{iteration}.vtk", spacing)
                cpu_weights = weights.cpu().numpy()
                tensor_to_vtk(cpu_weights, f"test_{iteration}_weight.vtk", spacing)

            # Densification
            if (iteration <= opt.densify_until_iter and
                iteration >= opt.densify_from_iter and
                iteration % opt.densification_interval == 0
            ):
                cpu_cells = cells.cpu().numpy()
                print(f"Number of cells that weren't seen: {np.count_nonzero(np.logical_and(cpu_cells.ravel() == -1, gt_cells.ravel() != -1))}")
                mse = torch.mean((cells - gt) ** 2)
                psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
                mse2 = torch.mean((cells[torch.logical_and(cells != -1, gt != -1)] - gt[torch.logical_and(cells != -1, gt != -1)]) ** 2)
                psnr2 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse2 + 1e-8)
                print(f"Num Gaussians: {gaussians.get_values.shape[0]}, psnr: {psnr}, psnr without empty: {psnr2}")
                gaussians.densify_and_prune(
                    opt.densify_grad_threshold,
                    min_weight,
                    samples_tf_flat[np.logical_and(cpu_cells.ravel() == -1, gt_cells.ravel() != -1)],
                    gt_cells.ravel()[np.logical_and(cpu_cells.ravel() == -1, gt_cells.ravel() != -1)].reshape(-1, 1)
                )

                # if iteration % opt.weight_reset_interval == 0 or (
                #     dataset.white_background and iteration == opt.densify_from_iter
                # ):
                #     gaussians.reset_weight()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    os.path.join(scene.model_path, "/chkpnt{iteration}.pth"),
                )

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
    parser.add_argument("--min_weight", type=float, default=0.0001)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[1, 1_000, 2_000, 4_000, 6_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log_to_file", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
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
        args.min_weight
    )

    # All done
    print("\nTraining complete.")
