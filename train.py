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
from gaussian_renderer import init_rasterizer, render, build_bvh, intersect
from gpu_mesh_sampling import gpu_sample, gpu_sampleu
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
    min_weight,
    is_scaled
):
    vtk_files = []
    vtk_files_loss = []
    log_data = []
    first_iter = 0
    prepare_output(dataset)
    gaussians = GaussianModel()
    scene = Scene(dataset, gaussians, normalized=is_scaled, fraction=fraction)
    struct = dataset.source_path.lower().endswith('.vtk')
    gaussians.training_setup(opt)
    print("Before save")
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
    std = 0
    mean = 0
    avg = 0
    error_thresh = 0.05
    new_scale = 0.1
    densifies = 0
    lossy_frac = 0.0

    # Make ground truth
    print("Before cell")
    cell_count = 128
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
    save_cell = samples_tf.reshape(-1, 3)
    print("Save cell made")
    if struct:
        save_gt = gpu_sample(
            gaussians.mesh.dimensions,
            gaussians.mesh.origin,
            gaussians.mesh.spacing,
            gaussians.mesh.point_data['value'],
            save_cell
        )
    else:
        save_gt = gpu_sampleu(
            gaussians.mesh.points, 
            gaussians.mesh.cell_connectivity.astype(np.int64),
            gaussians.mesh.celltypes.astype(np.int64),
            gaussians.mesh.offset.astype(np.int64),
            gaussians.mesh.point_data[gaussians.mesh.array_names[0]],
            save_cell
        )
    num_batches = 100
    size = cell_count ** 3
    big_samples = np.tile(save_cell, (num_batches, 1))
    big_jitter = np.random.uniform(-0.5, 0.5, big_samples.shape)
    big_jitter *= np.array(spacing)[None, :]
    big_jitter[: cell_count**3, :] = 0
    big_samples = np.clip(
        big_samples + big_jitter, 
        np.array(gaussians.mins), 
        np.array(gaussians.maxes)
    )
    if struct:
        big_gt = gpu_sample(
            gaussians.mesh.dimensions,
            gaussians.mesh.origin,
            gaussians.mesh.spacing,
            gaussians.mesh.point_data['value'],
            big_samples
        )
    else:
        big_gt = gpu_sampleu(
            gaussians.mesh.points, 
            gaussians.mesh.cell_connectivity.astype(np.int64),
            gaussians.mesh.celltypes.astype(np.int64),
            gaussians.mesh.offset.astype(np.int64),
            gaussians.mesh.point_data[gaussians.mesh.array_names[0]],
            big_samples
        )
    big_gt = big_gt.reshape(num_batches, cell_count**3)
    big_samples = big_samples.reshape(num_batches, cell_count**3, 3)
    end = time.time()

    # size = 262144
    # start = time.time()
    # num_batches = 1000
    # idx = torch.randint(gaussians.mesh.n_points, (num_batches, size))
    # nx, ny, nz = gaussians.mesh.dimensions
    # ox, oy, oz = gaussians.mesh.origin
    # sx, sy, sz = gaussians.mesh.spacing
    # nxny = nx * ny
    # k, r = np.divmod(idx, nxny)
    # j, i = np.divmod(r, nx)
    # x = ox + i * sx
    # y = oy + j * sy
    # z = oz + k * sz
    # mesh_samples = np.stack((x, y, z), axis=-1)
    # # mesh_samples = gaussians.mesh.points[idx]
    # mesh_vals = gaussians.mesh.point_data[gaussians.mesh.array_names[0]][idx]
    # big_gt = mesh_vals.reshape(num_batches, size)
    # big_samples = mesh_samples.reshape(num_batches, size, 3)
    # end = time.time()
    # print(f"Time to sample gt: {end - start}")
    
    gt_cells = big_gt[0]
    print(f"Number of invalid samples: {np.count_nonzero(gt_cells == -1)}")
    # tensor_to_vtk(save_gt.reshape(cell_count, cell_count, cell_count), "test_gt.vtk", spacing)
    gt = torch.tensor(gt_cells).cuda()
    if debug_from == 0:
        pipe.debug = True
    use_gaussian_bvh = False
    init_rasterizer(
        gaussians,
        pipe,
        cell_count,
        use_gaussian_bvh=use_gaussian_bvh
    )
    build_bvh(torch.tensor(big_samples[0], dtype=torch.float, device="cuda"))

    loss_idx = (gt_cells == 2)
    loss_samples = big_samples[0][loss_idx]
    loss_gt = gt_cells[loss_idx]

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        deb = False
        if iteration % 1000 == 0:
            deb = True

        # if iteration in saving_iterations or iteration in testing_iterations:
        #     gt_cells = save_gt
        #     gt = torch.tensor(gt_cells).cuda()
        #     current_samples = save_cell
        # else:
        num_loss = loss_samples.shape[0]
        jit_idx = np.random.randint(0, num_batches)
        gt_cells = np.concatenate([
            loss_gt,
            big_gt[jit_idx][:(size - num_loss)]
        ])
        gt = torch.tensor(gt_cells).cuda()
        current_samples = np.concatenate([
            loss_samples,
            big_samples[jit_idx][:(size - num_loss)]
        ])
        build_bvh(torch.tensor(current_samples, dtype=torch.float, device="cuda"), deb)

        gaussians.update_learning_rate(iteration)

        # Render
        render_pkg = render(
            gaussians,
            deb
        )
        cells, weights= (
            render_pkg["cells"],
            render_pkg["weights"]
        )
        # intersect_pkg = intersect(
        #     gaussians,
        #     deb
        # )
        # intersections, intersection_weights = (
        #     intersect_pkg["intersections"],
        #     intersect_pkg["intersection_weight"]
        # )
        overlap_loss = torch.tensor(0, device="cuda")
        # overlap_loss = torch.mean(torch.pow(10, 100 * intersection_weights) - 1)
        # overlap_loss = torch.mean(intersection_weights)
        recon_mask = torch.logical_and(gt != -1, cells != -1)
        l1_lv = l1_loss(cells[recon_mask], gt[recon_mask])
        # TODO: FIX FP AND FN FOR CHANGING CELL COUNTS
        k = 600  # Adjust this to control decay rate
        fn_mask = torch.logical_and(gt != -1, weights < 0.03)
        t = 0.01
        delta = 0.002 
        # fn_mask = (gt != -1)
        if fn_mask.any():
            false_negative = torch.exp(-k * weights[fn_mask])
            # false_negative = torch.exp(-k * torch.clamp(weights[fn_mask] - 0.01, 0.0))
            # false_negative = torch.clamp(0.015 - weights[fn_mask], min=0)
            # false_negative = 0.01 * (torch.clamp((t + delta - weights[fn_mask]) / delta, min=0.0) ** 2).mean()
            # false_negative = (torch.pow(10, -1000 * (weights[fn_mask] - 0.01))).mean()
            mean_mask = (false_negative > 0.0)
            if mean_mask.any():
                false_negative = false_negative[mean_mask].mean()
            else:
                false_negative = torch.tensor(0., device="cuda")
        else:
            false_negative = torch.tensor(0., device="cuda")
        mask = torch.logical_and(gt == -1, weights > 0)
        if mask.any() and not struct:
            false_positive = (1 * (1 - torch.exp(-k * weights[mask]))).mean()
        else:
            false_positive = torch.tensor(0., device="cuda")
        loss = l1_lv + false_positive + false_negative + overlap_loss
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Compute the lossy samples where new Gaussians are needed
            recon_mask = torch.logical_and(cells != -1, gt != -1)
            # recon_mask = (gt != -1)
            if iteration >= opt.densify_from_iter and iteration not in saving_iterations and iteration not in testing_iterations:
                med = torch.median(torch.abs(cells - gt))
                stdn, meann = torch.std_mean(torch.abs(cells - gt))
                mean = (mean * avg + meann) / (avg + 1)
                avg += 1
                loss_idx = torch.logical_and(
                    torch.abs(cells - gt) > error_thresh,
                    recon_mask
                ).cpu().numpy()
                lossy_frac = 0.9 * lossy_frac + 0.1 * np.count_nonzero(loss_idx) / size
                # loss_samples = current_samples[loss_idx]
                # loss_gt = gt_cells[loss_idx]
                # if loss_idx.sum() < 1000:
                #     error_thresh *= 0.9
                #     print(f"Lossy samples low, new error thresh: {error_thresh}")

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
                    "l_v": l1_lv.item(),
                    "false_positive": false_positive.item(),
                    "psnr": psnr.item(),
                    "psnr2": psnr2.item(),
                    "num_gaussians": num_gaussians
                })
            
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            mse = torch.mean((cells - gt) ** 2)
            psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
            mse2 = torch.mean((cells[torch.logical_and(cells != -1, gt != -1)] - gt[torch.logical_and(cells != -1, gt != -1)]) ** 2)
            psnr2 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse2 + 1e-8)
            if iteration in testing_iterations:
                print(f"Testing PSNR at iteration {iteration}: {psnr}")
                print(f"Testing fraction of samples that are lossy: {np.count_nonzero(loss_idx) / size}, avg: {lossy_frac}")
                # if np.count_nonzero(loss_idx) / size < 0.01 and iteration > 1:
                #     error_thresh -= 0.1
                #     lossy_frac = 0
                #     print(f"Error thresh changed to {error_thresh}")
            ema_lv_for_log = 0.1 * l1_lv + 0.9 * ema_lv_for_log
            ema_lfp_for_log = 0.1 * false_positive + 0.9 * ema_lfp_for_log
            ema_lfn_for_log = 0.1 * false_negative + 0.9 * ema_lfn_for_log
            ema_lpsnr_for_log = 0.1 * psnr + 0.9 * ema_lpsnr_for_log
            if iteration % 250 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "L_v": f"{ema_lv_for_log:.{5}f}",
                        "L_fp": f"{ema_lfp_for_log:.{5}f}",
                        "L_fn": f"{ema_lfn_for_log:.{5}f}",
                        "PSNR": f"{ema_lpsnr_for_log:.{5}f}"
                    }
                )
                progress_bar.update(250)
                # print(f"0 cells: {torch.count_nonzero(gt == 0).cpu().numpy()}, -1: {torch.count_nonzero(gt == -1).cpu().numpy()}")
                # print(f"0 cells: {torch.count_nonzero(cells == 0).cpu().numpy()}, -1: {torch.count_nonzero(cells == -1).cpu().numpy()}")
                print(f"False negative: {torch.count_nonzero(torch.logical_and(cells== -1, gt != -1))}, false positive: {torch.count_nonzero(torch.logical_and(cells != -1, gt == -1))}")
                print(f"Num Gaussians: {gaussians.get_values.shape[0]}, psnr: {psnr}, psnr2: {psnr2}, weight: {torch.mean(weights)}")
                print(f"False negative mask: {fn_mask.sum()}")
                print(loss_samples.shape)
                # print(f"Overlap loss: {overlap_loss} mean {torch.mean(intersection_weights)} max: {torch.max(intersection_weights)} median: {torch.median(intersection_weights)} intersections: {torch.mean(intersections)}, max: {torch.max(intersections)}")
                # top5 = torch.topk(intersection_weights, 5)
                # print(f"Top 5: {top5.values}, weight: {gaussians.get_weight[top5.indices]}, scale: {gaussians.get_scaling[top5.indices]}")
                # print(f"scale: {torch.mean(gaussians.get_scaling)}, median: {torch.median(gaussians.get_scaling)}, std: {torch.std(gaussians.get_scaling)}")
                # print(f"Weights below 0.01: {torch.count_nonzero(torch.logical_and(weights > 0.0, weights < 0.01))}")
                # print(f"{torch.mean(gaussians.get_scaling[gaussians.get_values.squeeze(-1) != 0])}")
            if iteration == opt.iterations:
                progress_bar.close()

            # Save
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                cpu_cells = cells.cpu().numpy()
                tensor_to_vtk(cpu_cells.reshape(cell_count, cell_count, cell_count), f"out_vtk/test_{iteration}.vtk", spacing)
                tensor_to_vtk(torch.abs((cells - gt)).cpu().numpy().reshape(cell_count, cell_count, cell_count), f"out_vtk/test_{iteration}_loss.vtk", spacing)
                vtk_files.append({
                    "name": f"test_{iteration}.vtk",
                    "time": float(saving_iterations.index(iteration))
                })                
                vtk_files_loss.append({
                    "name": f"test_{iteration}_loss.vtk",
                    "time": float(saving_iterations.index(iteration))
                })

            # Densification
            if (iteration <= opt.densify_until_iter and
                iteration >= opt.densify_from_iter and
                iteration % opt.densification_interval == 0 and
                 iteration not in saving_iterations and
                 iteration not in testing_iterations
            ):
                # if densifies > 0 and densifies % 30 == 0 and error_thresh > 0.0125:
                #     error_thresh *= 0.5
                #     new_scale *= 0.5
                #     print(f"New thresh {error_thresh}, new scale {new_scale}")

                # cpu_cells = cells.cpu().numpy()
                # print(f"False negative: {np.count_nonzero(np.logical_and(cpu_cells.ravel() == -1, gt_cells.ravel() != -1))}, false positive: {np.count_nonzero(np.logical_and(cpu_cells.ravel() != -1, gt_cells.ravel() == -1))}")
                # mse = torch.mean((cells - gt) ** 2)
                # psnr = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse + 1e-8)
                # mse2 = torch.mean((cells[torch.logical_and(cells != -1, gt != -1)] - gt[torch.logical_and(cells != -1, gt != -1)]) ** 2)
                # psnr2 = 20 * torch.log10(torch.tensor(1.0)) - 10 * torch.log10(mse2 + 1e-8)
                gaussians.densify_and_prune(
                    opt.densify_grad_threshold,
                    min_weight,
                    # new_scale,
                    torch.mean(gaussians.get_scaling) / 6.0,
                    # current_samples[np.logical_and(current_samples == -1, gt != -1)],
                    # gt_cells.ravel()[np.logical_and(cpu_cells.ravel() == -1, gt_cells.ravel() != -1)].reshape(-1, 1)
                    current_samples[loss_idx],
                    gt_cells[loss_idx].reshape(-1, 1)
                )
                densifies += 1
                # error_thresh -= 0.002

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
    series = {
        "file-series-version": "1.0",
        "files": vtk_files
    }
    with open("out_vtk/test.vtk.series", "w") as jf:
        json.dump(series, jf, indent=2)

    series_loss = {
        "file-series-version": "1.0",
        "files": vtk_files_loss
    }
    with open("out_vtk/test_loss.vtk.series", "w") as jf:
        json.dump(series_loss, jf, indent=2)

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
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[]
    )
    # parser.add_argument(
    #     "--test_iterations", nargs="+", type=int, default=[1] + [i * 1000 for i in range(32)]
    # )
    # parser.add_argument(
    #     "--save_iterations", nargs="+", type=int, default=[1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 48_000, 64_000]
    # )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log_to_file", action="store_true")
    parser.add_argument("--is_scaled", action="store_true")
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
        args.min_weight,
        args.is_scaled
    )

    # All done
    print("\nTraining complete.")
