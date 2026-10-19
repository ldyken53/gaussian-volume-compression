import os
import sys
import uuid
import json
import time
import math
from argparse import ArgumentParser, Namespace
from random import randint
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pyvista as pv

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import init_rasterizer, render, build_bvh, build_bvh_cached, sample_bvh_prim_ids, margin_set_exterior, margin_fp
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

try:
    from bvh_diff_gaussian_rasterization.fused import fused_volume_loss as _fused_loss
except ImportError:  # a rasterizer build without fused_ops.cu
    _fused_loss = None

# Fixed seed for the sample points, so the cached ground truth stays valid and every
# run sees the same points.
SAMPLE_SEED = 20260903


def sample_cache_path(cache_dir, source_path, cell_count, num_batches, encode_surface):
    """Path of the cached ground truth, or None if caching is disabled."""
    if not cache_dir:
        return None
    stem = os.path.splitext(os.path.basename(source_path))[0]
    tag = f"{stem}_c{cell_count}_b{num_batches}_s{int(bool(encode_surface))}"
    if encode_surface and args.interior_skin > 0:
        tag += f"_skin{args.interior_skin:g}"
    return os.path.join(cache_dir, tag + "_gt.npy")


def sample_mesh_points(mesh, num_batches, batch_size, device="cuda"):
    points = torch.tensor(mesh.points, dtype=torch.float32, device=device)
    tet_mesh = mesh.triangulate()

    # Extract cell connectivity — assumes tets (4 verts per cell)
    cells = torch.tensor(
        tet_mesh.cells.reshape(-1, 5)[:, 1:], dtype=torch.long, device=device
    )  # (C, 4)

    total = num_batches * batch_size

    # Pick random cells (uniform = denser where cells are smaller = where points are denser)
    cell_idx = torch.randint(0, cells.shape[0], (total,), device=device)
    verts = points[cells[cell_idx]]  # (total, 4, 3)

    # Random barycentric coordinates inside a tetrahedron
    # Uniformly sample a tet: take 3 random values, sort, then differences give bary coords
    u = torch.rand(total, 3, device=device).sort(dim=1).values
    bary = torch.zeros(total, 4, device=device)
    bary[:, 0] = u[:, 0]
    bary[:, 1] = u[:, 1] - u[:, 0]
    bary[:, 2] = u[:, 2] - u[:, 1]
    bary[:, 3] = 1.0 - u[:, 2]

    # Interpolate: (total, 4, 1) * (total, 4, 3) summed over verts
    samples = (bary.unsqueeze(2) * verts).sum(dim=1)

    return samples.reshape(num_batches, batch_size, 3)

def sample_exterior_points(mesh, num_batches, batch_size, offset=0.01, device="cuda", inward=False):
    """Sample points just outside the mesh surface using properly oriented normals."""
    surf = mesh.extract_surface().triangulate().compute_normals(cell_normals=True, point_normals=False)
    print(surf.faces.shape)

    face_normals = torch.tensor(surf.cell_data["Normals"], dtype=torch.float32, device=device)
    points = torch.tensor(surf.points, dtype=torch.float32, device=device)
    faces = torch.tensor(
        surf.faces.reshape(-1, 4)[:, 1:], dtype=torch.long, device=device
    )

    total = num_batches * batch_size

    face_idx = torch.randint(0, faces.shape[0], (total,), device=device)
    v = points[faces[face_idx]]

    # Uniform barycentric coords on triangle
    u = torch.rand(total, 2, device=device)
    sqrt_u0 = u[:, 0].sqrt()
    bary = torch.stack([1 - sqrt_u0, sqrt_u0 * (1 - u[:, 1]), sqrt_u0 * u[:, 1]], dim=1)

    surf_pts = (bary.unsqueeze(2) * v).sum(dim=1)

    # Use PyVista's properly oriented outward normals
    normals = face_normals[face_idx]

    dist = torch.rand(total, 1, device=device) * offset
    # inward=True mirrors the exterior band just inside the surface (the probe labels any
    # point that a flipped normal or a concavity puts outside as exterior anyway).
    samples = surf_pts + normals * (-dist if inward else dist)

    return samples.reshape(num_batches, batch_size, 3)


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
    precompute_samples,
    precomputed_prefix,
    encode_surface,
    sample_cache_dir,
    stream_samples=False
):
    use_mcmc = args.use_mcmc
    vtk_files = []
    vtk_files_loss = []
    log_data = []
    first_iter = 0
    prepare_output(dataset)
    # Densification here has always ended at cap_max + 1000; --cap_exact makes cap_max the
    # final count, as struct d92929a does.
    n_final = args.cap_max if args.cap_exact else args.cap_max + 1000
    gaussians = GaussianModel()
    gaussians.init_weight = args.init_weight
    scene = Scene(dataset, gaussians, normalized=is_scaled, fraction=fraction,
                  max_init=_max_init(args, n_final))
    if args.max_scale_k > 0:
        # From struct (f3cc97d/5312eee): the scale cap grows with the Gaussian spacing
        # N^(-1/3); --max_scale is the floor. That spacing assumes the data fills the unit
        # cube, as struct's volumes do. A mesh normalised into it can fill far less (sf1's
        # bounding box is 10% of it, mito's 15%), so --max_scale_vol uses the spacing
        # (V_bbox / N)^(1/3) instead; identical when the box is the whole cube.
        vol = 1.0
        if args.max_scale_vol:
            vol = float(np.prod(np.array(gaussians.maxes) - np.array(gaussians.mins)))
        args.max_scale = max(args.max_scale, args.max_scale_k * (vol / n_final) ** (1.0 / 3.0))
        print(f"max_scale {args.max_scale:.4f} (k={args.max_scale_k}, N={n_final}, V={vol:.3f})")
    with open(os.path.join(dataset.model_path, "max_scale"), "w") as _msf:
        _msf.write(f"{args.max_scale}\n")
    gaussians.max_scale = args.max_scale
    struct = dataset.source_path.lower().endswith('.vtk')
    gaussians.training_setup(opt)
    if args.densify_growth > 1.0:
        # From struct 5312eee: enough growth events that each grows the model by about
        # densify_growth, at least 3 and finished by half-way so the last ones still train.
        n0 = max(1, gaussians.get_xyz.shape[0])
        latest = max(3, (opt.iterations // 2 - opt.densify_from_iter) // opt.densification_interval)
        want = math.ceil(math.log(max(n_final, n0) / n0) / math.log(args.densify_growth))
        args.densify_events = int(min(latest, max(3, want)))
        print(f"densify_events auto: {args.densify_events} ({n0:,} -> {n_final:,} Gaussians at "
              f"x{args.densify_growth:g} per event, at most {latest})")
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
    if precompute_samples:
        # <prefix>_gt.npy (num_batches, size) and <prefix>_samples.npy (num_batches,
        # size, 3), already in the mesh's coordinate space. Skips sampling and the
        # vtkProbeFilter entirely -- the only tractable path for meshes whose
        # triangulation would not fit on the GPU.
        t0 = time.time()
        big_gt = np.load(precomputed_prefix + "_gt.npy")
        num_batches = big_gt.shape[0]
        size = big_gt.shape[1]
        big_samples = np.load(precomputed_prefix + "_samples.npy")
        assert big_samples.shape[:2] == (num_batches, size), (
            f"shape mismatch: gt {big_gt.shape} vs samples {big_samples.shape}")
        print(f"Loaded precomputed samples from {precomputed_prefix}_* "
              f"({num_batches}x{size}, {int((big_gt == -1).sum())} sentinels) "
              f"in {time.time() - t0:.1f}s")
    else:
        # if struct:
        #     save_gt = gpu_sample(
        #         gaussians.mesh.dimensions,
        #         gaussians.mesh.origin,
        #         gaussians.mesh.spacing,
        #         gaussians.mesh.point_data['value'],
        #         save_cell
        #     )
        # else:
        #     save_gt = gpu_sampleu(
        #         gaussians.mesh.points,
        #         gaussians.mesh.cell_connectivity.astype(np.int64),
        #         gaussians.mesh.celltypes.astype(np.int64),
        #         gaussians.mesh.offset.astype(np.int64),
        #         gaussians.mesh.point_data[gaussians.mesh.array_names[0]],
        #         save_cell
        #     )
        # num_batches = 100
        # size = cell_count ** 3
        # big_samples = np.tile(save_cell, (num_batches, 1))
        # big_jitter = np.random.uniform(-0.5, 0.5, big_samples.shape)
        # big_jitter *= np.array(spacing)[None, :]
        # big_jitter[: cell_count**3, :] = 0
        # big_samples = np.clip(
        #     big_samples + big_jitter,
        #     np.array(gaussians.mins),
        #     np.array(gaussians.maxes)
        # )
        # # # Sort spatially via Morton code (Z-order curve)
        # # def part1by2(n):
        # #     n = n.astype(np.uint64) & 0x1fffff
        # #     n = (n | (n << 32)) & 0x1f00000000ffff
        # #     n = (n | (n << 16)) & 0x1f0000ff0000ff
        # #     n = (n | (n << 8))  & 0x100f00f00f00f00f
        # #     n = (n | (n << 4))  & 0x10c30c30c30c30c3
        # #     n = (n | (n << 2))  & 0x1249249249249249
        # #     return n

        # # scale = (1 << 21) - 1
        # # big_samples = big_samples.reshape(num_batches, size, 3)

        # # norm = np.clip(big_samples, 0.0, 1.0)
        # # q = (norm * scale).astype(np.uint64)

        # # morton = (
        # #     part1by2(q[:, :, 0])
        # #     | (part1by2(q[:, :, 1]) << 1)
        # #     | (part1by2(q[:, :, 2]) << 2)
        # # )
        # # order = np.argsort(morton, axis=1)
        # # big_samples = np.take_along_axis(big_samples, order[:, :, None], axis=1)
        # # big_samples = big_samples.reshape(num_batches * size, 3)
        # if struct:
        #     big_gt = gpu_sample(
        #         gaussians.mesh.dimensions,
        #         gaussians.mesh.origin,
        #         gaussians.mesh.spacing,
        #         gaussians.mesh.point_data['value'],
        #         big_samples
        #     )
        # else:
        #     big_gt = gpu_sampleu(
        #         gaussians.mesh.points,
        #         gaussians.mesh.cell_connectivity.astype(np.int64),
        #         gaussians.mesh.celltypes.astype(np.int64),
        #         gaussians.mesh.offset.astype(np.int64),
        #         gaussians.mesh.point_data[gaussians.mesh.array_names[0]],
        #         big_samples
        #     )
        # big_gt = big_gt.reshape(num_batches, cell_count**3)
        # big_samples = big_samples.reshape(num_batches, cell_count**3, 3)
        # end = time.time()

        size = cell_count ** 3
        # start = time.time()
        num_batches = 100
        # idx = torch.randint(gaussians.mesh.n_points, (num_batches, size))
        # idx = torch.arange(num_batches * size) % gaussians.mesh.n_points
        # idx = idx.view(num_batches, size)
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
        # Drawing the sample points costs ~1s on the GPU; probing them for ground truth
        # costs ~9 CPU-minutes (single-threaded vtkProbeFilter over 5.5M cells). So the
        # points are always redrawn from a fixed seed -- which makes them reproducible
        # across runs -- and only the ground truth is cached.
        cache_g = sample_cache_path(
            sample_cache_dir, dataset.source_path, cell_count, num_batches, encode_surface
        )
        t0 = time.time()
        torch.manual_seed(SAMPLE_SEED)
        if encode_surface:
            size2 = int(math.floor(size * 0.2))
            # --interior_skin: a share of the batch sampled in a band just INSIDE the surface,
            # mirroring the exterior band, so both sides of the boundary are sampled at the
            # same density per unit area. Otherwise the inside of the boundary only gets the
            # few uniform-volume samples that happen to land there -- a share set by the
            # domain's surface-to-volume ratio -- and the FN/FP balance shifts per dataset.
            size3 = int(math.floor(size * args.interior_skin))
            size1 = size - size2 - size3
            big_samples = sample_mesh_points(gaussians.mesh, 100, size1).cpu().numpy()
            big_samples2 = sample_exterior_points(gaussians.mesh, 100, size2).cpu().numpy()
            parts = [big_samples, big_samples2]
            if size3 > 0:
                parts.append(sample_exterior_points(gaussians.mesh, 100, size3, inward=True).cpu().numpy())
            big_samples = np.concatenate(parts, axis=1)
        else:
            big_samples = sample_mesh_points(gaussians.mesh, 100, size).cpu().numpy()
        print(f"sample_mesh_points: {time.time() - t0:.1f}s")
        print(big_samples.shape)
        cached = cache_g is not None and os.path.exists(cache_g)
        if cached:
            t0 = time.time()
            big_samples = big_samples.reshape(num_batches, size, 3)
            big_gt = np.load(cache_g)
            print(f"Loaded cached ground truth from {cache_g} in {time.time() - t0:.1f}s")
        if not cached:
        # big_samples = gaussians.mesh.points[idx]
        # big_jitter = np.random.uniform(-0.5, 0.5, big_samples.shape)
        # big_jitter *= np.array(spacing)[None, :]
        # # big_jitter[:size, :] = 0
        # big_samples = np.clip(
        #     big_samples + big_jitter,
        #     np.array(gaussians.mins),
        #     np.array(gaussians.maxes)
        # )
            big_samples = big_samples.reshape(num_batches * size, 3)
            t0 = time.time()
            probe_mesh = pv.PolyData(big_samples)
            print(f"pv.PolyData: {time.time() - t0:.1f}s")
            t0 = time.time()
            probed = probe_mesh.sample(gaussians.mesh)
            print(f"probe_mesh.sample: {time.time() - t0:.1f}s")
            big_gt = probed[gaussians.mesh.array_names[0]]
            valid_mask = probed['vtkValidPointMask'].astype(bool)
            big_gt[~valid_mask] = -1
        # big_gt = gpu_sampleu(
        #     gaussians.mesh.points,
        #     gaussians.mesh.cell_connectivity.astype(np.int64),
        #     gaussians.mesh.celltypes.astype(np.int64),
        #     gaussians.mesh.offset.astype(np.int64),
        #     gaussians.mesh.point_data[gaussians.mesh.array_names[0]],
        #     big_samples
        # )
        # big_gt = gaussians.mesh.point_data[gaussians.mesh.array_names[0]][idx]

            big_gt = big_gt.reshape(num_batches, size)
            big_samples = big_samples.reshape(num_batches, size, 3)
            if cache_g is not None:
                t0 = time.time()
                os.makedirs(os.path.dirname(cache_g), exist_ok=True)
                tmp = cache_g + f".tmp{os.getpid()}"
                np.save(tmp, np.ascontiguousarray(big_gt, dtype=np.float32))
                os.replace(tmp + ".npy" if not tmp.endswith(".npy") else tmp, cache_g)
                print(f"Wrote ground-truth cache to {cache_g} in {time.time() - t0:.1f}s")

    if not encode_surface:
        # A non-surface model has no exterior to encode, so a -1 target is not
        # something it can or should be fitted or scored against. Drop those samples
        # outright rather than masking them downstream: each is replaced by another
        # in-mesh sample drawn from the same batch, which keeps the arrays rectangular
        # and every reported metric sentinel-free. Almost always a no-op (the on-the-fly
        # sampler draws inside tetrahedra); it matters for precomputed sets, e.g.
        # earthquakebig carries 81 out-of-mesh points that otherwise dominate PSNR.
        bad = (big_gt == -1)
        n_bad = int(bad.sum())
        if n_bad:
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
            print(f"non-surface: replaced {n_bad} out-of-mesh samples with in-mesh ones")

    if stream_samples:
        big_gt_host = torch.from_numpy(np.ascontiguousarray(big_gt, dtype=np.float32)).pin_memory()
        big_samples_host = torch.from_numpy(np.ascontiguousarray(big_samples, dtype=np.float32)).pin_memory()
        print(f"stream_samples: {num_batches} batches pinned on host "
              f"({(big_gt_host.numel()*4 + big_samples_host.numel()*4)/1e9:.1f} GB)", flush=True)
    else:
        big_gt_cuda = torch.tensor(big_gt, dtype=torch.float, device="cuda")
        big_samples_cuda = torch.tensor(big_samples, dtype=torch.float, device="cuda")

    # Sort spatially via Morton code (Z-order curve)
    def part1by2_torch(n: torch.Tensor) -> torch.Tensor:
        # n: int64
        n = n & 0x1fffff
        n = (n | (n << 32)) & 0x1f00000000ffff
        n = (n | (n << 16)) & 0x1f0000ff0000ff
        n = (n | (n << 8))  & 0x100f00f00f00f00f
        n = (n | (n << 4))  & 0x10c30c30c30c30c3
        n = (n | (n << 2))  & 0x1249249249249249
        return n

    scale = (1 << 21) - 1

    if stream_samples:
        # Sort each batch on the GPU one at a time and write it back to host; the
        # wrapper below then serves batches with a per-iteration H2D copy so every
        # downstream `big_*_cuda[i]` reads unchanged.
        for _b in range(num_batches):
            _s = big_samples_host[_b].cuda(non_blocking=True)
            _g = big_gt_host[_b].cuda(non_blocking=True)
            _q = (torch.clamp(_s, 0.0, 1.0) * scale).to(torch.int64)
            _m = (part1by2_torch(_q[:, 0]) | (part1by2_torch(_q[:, 1]) << 1)
                  | (part1by2_torch(_q[:, 2]) << 2))
            _o = _m.argsort()
            big_samples_host[_b].copy_(_s[_o].cpu())
            big_gt_host[_b].copy_(_g[_o].cpu())
        class _HostBatches:
            def __init__(self, host): self.host = host
            def __getitem__(self, i): return self.host[i].cuda(non_blocking=True)
            @property
            def shape(self): return self.host.shape
        big_samples_cuda = _HostBatches(big_samples_host)
        big_gt_cuda = _HostBatches(big_gt_host)
    else:
        norm = torch.clamp(big_samples_cuda, 0.0, 1.0)
        q = (norm * scale).to(torch.int64)

        morton = (
            part1by2_torch(q[:, :, 0])
            | (part1by2_torch(q[:, :, 1]) << 1)
            | (part1by2_torch(q[:, :, 2]) << 2)
        )
        order = morton.argsort(dim=1)
        idx3 = order.unsqueeze(-1).expand(-1, -1, 3)  # [B, S, 3]
        big_samples_cuda = big_samples_cuda.gather(1, idx3)
        big_gt_cuda = big_gt_cuda.gather(1, order)
        # big_samples = np.take_along_axis(big_samples, order[:, :, None], axis=1)
        # big_gt = np.take_along_axis(big_gt, order, axis=1)
        # end = time.time()
        # print(f"Time to sample gt: {end - start}")

    gt = big_gt_cuda[0]
    print(f"Number of invalid samples: {torch.count_nonzero(gt == -1)}")
    # tensor_to_vtk(save_gt.reshape(cell_count, cell_count, cell_count), "test_gt.vtk", spacing)

    if debug_from == 0:
        pipe.debug = True
    use_gaussian_bvh = False
    init_rasterizer(
        gaussians,
        pipe,
        cell_count,
        use_gaussian_bvh=use_gaussian_bvh
    )
    if args.samples_per_node > 0:
        # Rule: render about this many samples per MESH NODE per iteration, never more than the
        # whole batch. The fixed 128^3 = 2.1M batch oversamples small meshes (mito 0.97M nodes,
        # rbl 0.73M) and undersamples big ones (valley 18.3M, earthquake 51M), which is why
        # mito/rbl lose nothing at 1/8 batches while valley/earthquake lose 0.8-1.6 dB at 1/2.
        n_nodes = int(gaussians.mesh.n_points)
        args.sample_frac = min(1.0, args.samples_per_node * n_nodes / size)
        print(f"samples_per_node {args.samples_per_node:g}: {n_nodes} nodes -> sample_frac {args.sample_frac:.4f}", flush=True)
    if args.samples_per_gaussian > 0:
        # Rule instead of a per-dataset --sample_frac: render about this many samples per
        # final Gaussian per iteration, never more than the whole batch. The subsampled
        # configs only lost nothing because they had samples to spare (mito 1024x 350 per
        # Gaussian at full batch, sf1 154); valley 64x has 5.5 and loses 1.6 dB at half.
        args.sample_frac = min(1.0, args.samples_per_gaussian * n_final / size)
        print(f"samples_per_gaussian {args.samples_per_gaussian:g}: sample_frac {args.sample_frac:.4f}", flush=True)
    from bvh_diff_gaussian_rasterization import _C as _rast_C
    if not hasattr(_rast_C, "build_bvh_cached"):
        # An older rasterizer build without these entry points: run without them.
        args.leaf_order = args.bvh_cache = 0
    if args.leaf_order and not stream_samples and not use_gaussian_bvh:
        # Put every batch in sample-BVH leaf order, once: cuBQL re-sorts the (Morton-sorted)
        # points by its own codes, so primIDs is a scattered permutation and every per-sample
        # load in the render kernels is a gather. After this, rebuilding on the permuted batch
        # gives near-identity primIDs. Preprocessing, like the Morton sort above (untimed).
        t0 = time.time()
        for _b in range(num_batches):
            build_bvh(big_samples_cuda[_b], False, use_gaussian_bvh)
            _perm = sample_bvh_prim_ids().long()
            big_samples_cuda[_b] = big_samples_cuda[_b][_perm]
            big_gt_cuda[_b] = big_gt_cuda[_b][_perm]
            if args.bvh_cache and not (0.0 < args.sample_frac < 1.0):
                # Prebuild the cached BVH of the permuted batch (sample preparation, like the
                # sort and the GT probe, so outside the timed loop).
                build_bvh_cached(big_samples_cuda[_b], _b)
        torch.cuda.synchronize()
        print(f"leaf order: {num_batches} batches permuted in {time.time() - t0:.1f}s", flush=True)
    build_bvh(big_samples_cuda[0], pipe.debug, use_gaussian_bvh)
    margin_n_ext = None
    if encode_surface and args.fp_mode == "margin" and not stream_samples:
        # One exterior set per batch (the FULL batch's gt == -1 samples, whatever
        # --sample_frac draws), each with its own cached BVH; preprocessing, untimed.
        margin_n_ext = []
        for _b in range(num_batches):
            _ext = big_samples_cuda[_b][big_gt_cuda[_b] == -1]
            margin_set_exterior(_ext, _b)
            margin_n_ext.append(max(1, int(_ext.shape[0])))
        print(f"margin FP: {num_batches} exterior sets, mean {np.mean(margin_n_ext):.0f} samples", flush=True)

    # loss_idx = (gt == 2)
    # loss_samples = big_samples_cuda[0][loss_idx]
    # loss_gt = gt[loss_idx]

    n = False
    # Training-time instrumentation. Wall clock over the iteration loop only:
    # excludes mesh loading, precomputed-sample loading and checkpoint writes, which
    # is the convention the baseline table uses ("training time only").
    save_seconds = 0.0
    # PROFILE_ITERS=N: profile iterations 1000..1000+N and dump the per-kernel table.
    # Diagnostic only -- unset, this costs nothing.
    # PROFILE_MODE=nsys instead opens a cudaProfilerApi capture range over the same
    # iterations (run under `nsys profile --capture-range=cudaProfilerApi`), with NVTX
    # ranges around each phase (struct d0b03fb).
    _prof_n = int(os.environ.get("PROFILE_ITERS", 0))
    _prof = None
    _nsys = _prof_n > 0 and os.environ.get("PROFILE_MODE") == "nsys"
    _nv_on = False
    def _nv(name=None):
        if _nv_on:
            torch.cuda.nvtx.range_pop() if name is None else torch.cuda.nvtx.range_push(name)
    if _prof_n and not _nsys:
        from torch.profiler import profile, ProfilerActivity
        _prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    sub_k = 0
    sub_stride = 1
    sub_base = None
    if 0.0 < args.sample_frac < 1.0:
        if args.sample_mode == "stratified":
            # stride must divide the batch evenly so every point is reachable.
            sub_stride = max(2, int(round(1.0 / args.sample_frac)))
            sub_k = size // sub_stride
            sub_base = (torch.arange(sub_k, device="cuda") * sub_stride)
        else:
            sub_k = max(1, int(round(size * args.sample_frac)))
        print(f"--sample_frac {args.sample_frac} ({args.sample_mode}): rendering "
              f"{sub_k} of {size} samples per iteration", flush=True)
    torch.cuda.synchronize()
    train_t0 = time.perf_counter()
    # --ema state: dict of buffers, (re)built lazily so densification cannot desync it.
    ema_params = None
    ema_names = ["_xyz", "_scaling", "_rotation", "_weight", "_values"]
    hinge_frozen, hinge_lv_sum, hinge_lv_n = None, 0.0, 0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        deb = False
        if iteration % 1000 == 0:
            deb = True

        # if iteration in saving_iterations or iteration in testing_iterations:
        #     gt_cells = save_gt
        #     gt = torch.tensor(gt_cells).cuda()
        #     current_samples = save_cell
        # else:
        # num_loss = loss_samples.shape[0]
        jit_idx = np.random.randint(0, num_batches)
        # gt_cells = np.concatenate([
        #     loss_gt,
        #     big_gt[jit_idx][:(size - num_loss)]
        # ])
        gt = big_gt_cuda[jit_idx]
        # current_samples = np.concatenate([
        #     loss_samples,
        #     big_samples[jit_idx][:(size - num_loss)]
        # ])
        current_samples = big_samples_cuda[jit_idx]
        if sub_k > 0:
            # Either way the indices must come out ASCENDING: the batch is Morton-sorted
            # and both the sample BVH build and its traversal rely on that locality.
            if sub_base is not None:
                # Stratified: one draw from each run of `sub_stride`. Ordered by
                # construction, so no sort -- the i.i.d. branch below pays 0.17 ms an
                # iteration for one.
                sub_idx = sub_base + torch.randint(
                    0, sub_stride, (sub_k,), device="cuda")
            else:
                sub_idx = torch.randint(0, size, (sub_k,), device="cuda").sort().values
            gt = gt[sub_idx]
            current_samples = current_samples[sub_idx]
        if _nsys:
            if iteration == 1000:
                torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStart(); _nv_on = True
            elif iteration == 1000 + _prof_n:
                torch.cuda.synchronize(); _nv_on = False; torch.cuda.cudart().cudaProfilerStop()
                print(f"NSYS window done ({_prof_n} iterations)", flush=True)
                return
        _nv("iter")
        if _prof is not None:
            if iteration == 1000:
                torch.cuda.synchronize(); _prof.__enter__()
            elif iteration == 1000 + _prof_n:
                torch.cuda.synchronize(); _prof.__exit__(None, None, None)
                print(_prof.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=35), flush=True)
                print(f"PROFILED {_prof_n} iterations", flush=True)
                return
        iter_start.record()
        _nv("bvh")
        if args.bvh_cache and sub_k == 0 and not use_gaussian_bvh and not stream_samples:
            build_bvh_cached(current_samples, jit_idx)
        else:
            build_bvh(current_samples, deb, use_gaussian_bvh)
        _nv()
        xyz_lr = gaussians.update_learning_rate(iteration)

        # Render
        _nv("render+loss")
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
        # (An unused `overlap_loss = torch.tensor(0, device="cuda")` stood here: a pageable
        # host-to-device copy that blocked the CPU on the forward render every iteration.)
        # overlap_loss = torch.mean(torch.pow(10, 100 * intersection_weights) - 1)
        # overlap_loss = torch.mean(intersection_weights)
        # Hinge FN target just above the occupancy threshold (OCC_T env, default the hard
        # cutoff's 0.01): (1 + --fn_margin) above it for surface models, 10% for interior.
        # Keep it symmetric with --fp_margin: the dead band between the two hinge targets
        # decides where the boundary settles, and centred below OCC_T it leaves FNs.
        fn_cut = ((1.0 + args.fn_margin) if encode_surface else 1.1) * float(os.environ.get("OCC_T", "0.01"))
        fn_tau = args.fn_thresh if args.fn_mode == "log" else fn_cut
        # FP threshold: log -> the per-Gaussian support threshold (any covered cell has aw >=
        # it); hinge -> a margin below the occupancy threshold OCC_T.
        if args.fp_mode == "hinge":
            fp_t = (1.0 - args.fp_margin) * float(os.environ.get("OCC_T", "0.01"))
        else:
            fp_t = float(os.environ.get("SUPPORT_T", "1e-4"))
        if _fused_loss is not None and os.environ.get("FUSED", "1") != "0":
            # Same loss as the torch branch below in 3 kernels instead of ~20 (+ ~15
            # backward); struct d0b03fb. FUSED=0 restores the torch branch.
            # --hinge_freeze F: RMSE-relative hinge weights until iteration F, then fixed at
            # the weights the mean data loss of iterations F-50..F gives (one host read).
            hw_fn, hw_fp, hw_rel = args.fn_reg, args.fp_reg, args.hinge_rel_pow
            if hinge_frozen is not None:
                hw_fn, hw_fp, hw_rel = args.fn_reg * hinge_frozen, args.fp_reg * hinge_frozen, 0.0
            _lo = _fused_loss(cells, gt, weights, l1=args.loss == "l1",
                              masked=(not encode_surface) or args.surf_mask_data,
                              fn_mode=args.fn_mode, fn_thresh=fn_tau, fn_reg=hw_fn,
                              fp_reg=hw_fp if (encode_surface and args.fp_mode != "margin") else 0.0,
                              fn_rel=args.fn_rel,
                              # only when used, so builds predating the log FP term still run
                              **({"fp_t": fp_t, "fp_mode": args.fp_mode}
                                 if args.fp_mode in ("log", "hinge") else {}),
                              **({"norm_all": True} if args.fnfp_norm == "all" else {}),
                              **({"rel_pow": hw_rel} if hw_rel > 0 else {}))
            loss, l1_lv, false_negative, false_positive = _lo[0], _lo[1], _lo[2], _lo[3]
            if args.hinge_freeze > 0 and args.hinge_rel_pow > 0 and hinge_frozen is None:
                if iteration > args.hinge_freeze - 50:
                    hinge_lv_sum = hinge_lv_sum + l1_lv.detach()
                    hinge_lv_n += 1
                if iteration >= args.hinge_freeze:
                    hinge_frozen = float(hinge_lv_sum / max(hinge_lv_n, 1)) ** args.hinge_rel_pow
                    print(f"hinge weights frozen at iteration {iteration}: scale {hinge_frozen:.3e} -> "
                          f"fn {args.fn_reg * hinge_frozen:.4g} fp {args.fp_reg * hinge_frozen:.4g}", flush=True)
            fn_vals = None  # computed only when a log line needs it
        else:
            recon_mask = torch.logical_and(gt != -1, cells != -1)
            # l1_lv = l1_loss(cells[recon_mask], gt[recon_mask])
            # PSNR is MSE-based, so L2 optimises the reported metric directly.
            if encode_surface and not args.surf_mask_data:
                if args.loss == "l2":
                    l1_lv = torch.mean((cells - gt) ** 2)
                else:
                    l1_lv = torch.abs(cells - gt).mean()
            else:
                # Masked mean written as a multiply: boolean indexing (cells[recon_mask])
                # launches nonzero(), whose data-dependent output size forces a device
                # sync every iteration. Mathematically identical.
                denom = recon_mask.sum().clamp_min(1)
                if args.loss == "l2":
                    l1_lv = (((cells - gt) ** 2) * recon_mask).sum() / denom
                else:
                    l1_lv = ((cells - gt).abs() * recon_mask).sum() / denom
            # l1_lv = ((cells[recon_mask] - gt[recon_mask]) ** 2).mean()
            # TODO: FIX FP AND FN FOR CHANGING CELL COUNTS
            k = 600  # Adjust this to control decay rate
            # fn_mask = torch.logical_and(gt != -1, weights < 0.03)
            # false_negative = torch.exp(-k * weights[fn_mask])
            # false_negative = false_negative[false_negative > 0].mean()
            # Cells at W == 0 are excluded: the rasterizer zeroes them and skips them in the
            # backward pass, so they carry no gradient and would only dilute the mean.
            # Written as a multiply, not boolean indexing -- identical maths, but indexing's
            # data-dependent output shape forces a GPU sync every iteration.
            fn_mask = torch.logical_and(gt != -1, weights > 0.0)
            if args.fn_mode == "log":
                # Log barrier from struct (f3cc97d): log(fn_thresh / aw) for covered cells below
                # fn_thresh. Only meaningful with SOFT_CUTOFF 1, where weakly covered cells keep
                # their true aw (under the hard cutoff every masked cell has aw > 0.01 > fn_thresh).
                # eps caps the penalty at log(100) where aw -> 0.
                eps = 0.01 * args.fn_thresh
                fn_vals = torch.clamp(torch.log(args.fn_thresh / (weights + eps)), min=0) * fn_mask
            else:
                fn_vals = torch.clamp(fn_cut - weights, min=0) * fn_mask
            fn_count = (fn_vals > 0).sum().float()
            # The FN penalty is an absolute quantity (at most fn_reg * fn_cut) while the data
            # loss varies by ~100x across datasets: mito sits at 1e-4..1e-3 where struct sits
            # at 3e-3..3e-2. So a single absolute fn_reg cannot be right for both. In "rel"
            # mode fn_reg is a multiple of the current data loss instead, which self-calibrates.
            fn_rel = args.fn_mode == "rel" or args.fn_rel
            fn_scale = args.fn_reg * l1_lv.detach() if fn_rel else args.fn_reg
            if args.fnfp_norm == "all":
                fn_count = (gt != -1).sum().float()
            false_negative = fn_scale * fn_vals.sum() / (fn_count + 1e-8)
            # false_negative = args.fn_reg * fn_vals.mean()

            # Multiply form for the same reason as above; and with encode_surface off the
            # fp term never enters the loss, so skip computing it entirely.
            if encode_surface and args.fp_mode != "margin":
                fp_mask = torch.logical_and(gt == -1, weights > 0.0)
                if args.fp_mode == "log":
                    fp_vals = torch.clamp(torch.log(torch.clamp_min(weights, 1e-30) / fp_t), min=0) * fp_mask
                elif args.fp_mode == "hinge":
                    fp_vals = torch.clamp(weights - fp_t, min=0) * fp_mask
                    fp_mask = fp_vals > 0   # mean over violators, like the FN hinge
                else:
                    fp_vals = weights * fp_mask
                fp_den = (gt == -1).sum().float() if args.fnfp_norm == "all" else fp_mask.sum().float()
                false_positive = args.fp_reg * fp_vals.sum() / (fp_den + 1e-8)
            else:
                false_positive = torch.zeros((), device="cuda")

            # t = 0.01
            # delta = 0.002
            # # fn_mask = (gt != -1)
            # if fn_mask.any():
            #     false_negative = torch.exp(-k * weights[fn_mask])
            #     # false_negative = torch.exp(-k * torch.clamp(weights[fn_mask] - 0.01, 0.0))
            #     # false_negative = torch.clamp(0.015 - weights[fn_mask], min=0)
            #     # false_negative = 0.01 * (torch.clamp((t + delta - weights[fn_mask]) / delta, min=0.0) ** 2).mean()
            #     # false_negative = (torch.pow(10, -1000 * (weights[fn_mask] - 0.01))).mean()
            #     mean_mask = (false_negative > 0.0)
            #     if mean_mask.any():
            #         false_negative = false_negative[mean_mask].mean()
            #     else:
            #         false_negative = torch.tensor(0., device="cuda")
            # else:
            #     false_negative = torch.tensor(0., device="cuda")
            # mask = torch.logical_and(gt == -1, weights > 0)
            # if mask.any() and not struct:
            #     false_positive = (1 * (1 - torch.exp(-k * weights[mask]))).mean()
            # else:
            #     false_positive = torch.tensor(0., device="cuda")
            if encode_surface:
                loss = l1_lv + false_negative + false_positive
            else:
                loss = l1_lv + false_negative
        if margin_n_ext is not None:
            # Support-margin FP (margin.cu), mean over the batch's exterior samples.
            false_positive = args.fp_reg * margin_fp(gaussians, jit_idx, fp_t) / margin_n_ext[jit_idx]
            loss = loss + false_positive
        if gaussians._values.shape[0] > args.cap_max:
            n = True
        if use_mcmc:
            loss = loss + args.weight_reg * torch.abs(gaussians.get_weight).mean()
            loss = loss + args.scale_reg * torch.abs(gaussians.get_scaling).mean()

        _nv(); _nv("backward")
        loss.backward()
        _nv()
        iter_end.record()

        with torch.no_grad():
            if args.fn_reg2 >= 0 and iteration > opt.densify_from_iter:
                args.fn_reg = args.fn_reg2
            # Compute the lossy samples where new Gaussians are needed
            recon_mask = torch.logical_and(cells != -1, gt != -1)
            # recon_mask = (gt != -1)
            # if iteration not in saving_iterations and iteration not in testing_iterations:
            #     med = torch.median(torch.abs(cells - gt))
            #     stdn, meann = torch.std_mean(torch.abs(cells - gt))
            #     mean = (mean * avg + meann) / (avg + 1)
            #     avg += 1
            #     loss_idx = torch.logical_and(
            #         torch.abs(cells - gt) > error_thresh,
            #         recon_mask
            #     ).cpu().numpy()
            #     lossy_frac = 0.9 * lossy_frac + 0.1 * np.count_nonzero(loss_idx) / size
                # loss_samples = current_samples[loss_idx]
                # loss_gt = gt_cells[loss_idx]
                # if loss_idx.sum() < 1000:
                #     error_thresh *= 0.9
                #     print(f"Lossy samples low, new error thresh: {error_thresh}")

            def _fn_count():
                # Cells the FN term is active on (logging only; the fused loss does not
                # materialise fn_vals).
                if fn_vals is not None:
                    return int((fn_vals > 0).sum())
                m = torch.logical_and(gt != -1, weights > 0.0)
                if args.fn_mode == "log":
                    m = torch.logical_and(m, weights + 0.01 * fn_tau < fn_tau)
                else:
                    m = torch.logical_and(m, weights < fn_tau)
                return int(m.sum())

            # Logging
            if log_to_file and iteration % 20 == 0:
                # (cells.cpu() used to be pulled here for a tensor_to_vtk dump that is
                # long commented out -- it was a 2M-element device-to-host copy, and a
                # sync, every 20 iterations for a value nothing read.)
                # float64 and a non-binding epsilon: at 1e-8 the epsilon alone caps
                # PSNR at 80 dB and compresses everything above ~72 (a true 90 dB reads
                # 79.6). float32 also runs out of digits when the residual is ~1e-5.
                mse = torch.mean((cells.double() - gt.double()) ** 2)
                psnr = -10 * torch.log10(mse + 1e-16)
                live = torch.logical_and(cells != -1, gt != -1)
                mse2 = (((cells - gt) ** 2) * live).sum() / live.sum().clamp_min(1)
                psnr2 = -10 * torch.log10(mse2 + 1e-16)
                num_gaussians = gaussians.get_values.shape[0]
                log_data.append({
                    "iteration": iteration,
                    "loss": loss.item(),
                    "l_v": l1_lv.item(),
                    "fn_cells": _fn_count(),
                    "fn_loss": float(false_negative),
                    "dead_cells": int((weights == 0).sum()),
                    "w_lt_002": int(torch.logical_and(weights > 0, weights < 0.02).sum()),
                    "w_lt_005": int(torch.logical_and(weights > 0, weights < 0.05).sum()),
                    "w_min_live": float(weights[weights > 0].min()) if (weights > 0).any() else 0.0,
                    "w_median": float(weights.median()),
                    # "false_positive": false_positive.item(),
                    "psnr": psnr.item(),
                    "psnr2": psnr2.item(),
                    "num_gaussians": num_gaussians
                })

            # Progress bar
            # if iteration in testing_iterations:
            #     print(f"Testing PSNR at iteration {iteration}: {psnr}")
            #     print(f"Testing fraction of samples that are lossy: {np.count_nonzero(loss_idx) / size}, avg: {lossy_frac}")
                # if np.count_nonzero(loss_idx) / size < 0.01 and iteration > 1:
                #     error_thresh -= 0.1
                #     lossy_frac = 0
                #     print(f"Error thresh changed to {error_thresh}")
            if iteration % 250 == 0:
                iter_end.synchronize()
                print(iter_start.elapsed_time(iter_end), "ms")
                ema_loss_for_log = 0.1 * loss.item() + 0.9 * ema_loss_for_log
                # float64 and a non-binding epsilon: at 1e-8 the epsilon alone caps
                # PSNR at 80 dB and compresses everything above ~72 (a true 90 dB reads
                # 79.6). float32 also runs out of digits when the residual is ~1e-5.
                mse = torch.mean((cells.double() - gt.double()) ** 2)
                psnr = -10 * torch.log10(mse + 1e-16)
                live250 = torch.logical_and(cells != -1, gt != -1)
                mse2 = (((cells - gt) ** 2) * live250).sum() / live250.sum().clamp_min(1)
                psnr2 = -10 * torch.log10(mse2 + 1e-16)
                ema_lv_for_log = 0.1 * l1_lv + 0.9 * ema_lv_for_log
                ema_lfp_for_log = 0.1 * false_positive + 0.9 * ema_lfp_for_log
                ema_lfn_for_log = 0.1 * false_negative + 0.9 * ema_lfn_for_log
                ema_lpsnr_for_log = 0.1 * psnr + 0.9 * ema_lpsnr_for_log
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "L_v": f"{ema_lv_for_log:.{5}f}",
                        "L_fp": f"{ema_lfp_for_log:.{5}f}",
                        "L_fn": f"{ema_lfn_for_log:.{5}f}",
                        # "PSNR": f"{ema_lpsnr_for_log:.{5}f}"
                    }
                )
                progress_bar.update(250)
                # print(f"0 cells: {torch.count_nonzero(gt == 0).cpu().numpy()}, -1: {torch.count_nonzero(gt == -1).cpu().numpy()}")
                # print(f"0 cells: {torch.count_nonzero(cells == 0).cpu().numpy()}, -1: {torch.count_nonzero(cells == -1).cpu().numpy()}")
                print(f"False negative: {torch.count_nonzero(torch.logical_and(cells== -1, gt != -1))}, false positive: {torch.count_nonzero(torch.logical_and(cells != -1, gt == -1))}")
                print(f"Num Gaussians: {gaussians.get_values.shape[0]}, psnr: {psnr}, psnr2: {psnr2}, weight: {torch.mean(weights)}")
                print(f"False negative mask: {_fn_count()}, false negative {false_negative.item()}, false positive {false_positive.item()}")
                print(f"Weight loss: {args.weight_reg * torch.abs(gaussians.get_weight).mean()}")
                # print(f"loss_samples.shape: {loss_samples.shape[0]}")
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
                torch.cuda.synchronize()
                _save_t0 = time.perf_counter()
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                if ema_params is not None:
                    # Ship the average, not the last iterate. Swap in-place so a mid-run
                    # save leaves training state untouched afterwards.
                    _bak = {nm: getattr(gaussians, nm).detach().clone() for nm in ema_names}
                    for nm in ema_names:
                        getattr(gaussians, nm).copy_(ema_params[nm])
                    # Instrumentation only: PSNR of the EMA model on the current batch,
                    # printed beside the live-model PSNR the benchmark reads.
                    ema_pkg = render(gaussians, False)
                    ema_cells = ema_pkg["cells"]
                    ema_mse = torch.mean((ema_cells.double() - gt.double()) ** 2)
                    ema_psnr = -10 * torch.log10(ema_mse + 1e-16)
                    # Masked twin, excluding cells the -1 sentinel touches on either
                    # side. Identical to ema_psnr when every sample is interior, but
                    # for sample sets that contain exterior points (encode_surface, or
                    # a precomputed set with a few out-of-mesh points) the unmasked
                    # figure is dominated by sentinel mismatch: 81 such cells out of
                    # 2.1M drag earthquake 64x from 62.3 to 43.7.
                    _m2 = torch.logical_and(ema_cells != -1, gt != -1)
                    ema_mse2 = torch.mean((ema_cells.double()[_m2] - gt.double()[_m2]) ** 2)
                    ema_psnr2 = -10 * torch.log10(ema_mse2 + 1e-16)
                    # Average FN/FP/psnr2 over EVERY jitter batch, not just whichever
                    # one the loop happened to end on -- a single batch is 1 of 100 and
                    # at FN counts in the single digits that is not a stable estimate.
                    tot_fn = tot_fp = tot_ext = 0
                    for _b in range(num_batches):
                        _gt = big_gt_cuda[_b]
                        build_bvh(big_samples_cuda[_b], False, use_gaussian_bvh)
                        _c = render(gaussians, False)["cells"]
                        tot_fn += int(torch.count_nonzero(
                            torch.logical_and(_c == -1, _gt != -1)))
                        tot_fp += int(torch.count_nonzero(
                            torch.logical_and(_c != -1, _gt == -1)))
                        tot_ext += int(torch.count_nonzero(_gt == -1))
                    # PSNR deliberately NOT reported from this loop: it is an
                    # in-sample average over the training jitter batches. Use the
                    # separate full-volume evaluation for the authoritative number.
                    print(f"ALLBATCH fn: {tot_fn} fp: {tot_fp} "
                          f"exterior: {tot_ext} cells: {num_batches * size}", flush=True)
                    build_bvh(current_samples, False, use_gaussian_bvh)
                    ema_fn = int(torch.count_nonzero(
                        torch.logical_and(ema_pkg["cells"] == -1, gt != -1)))
                    ema_fp = int(torch.count_nonzero(
                        torch.logical_and(ema_pkg["cells"] != -1, gt == -1)))
                    n_ext = int(torch.count_nonzero(gt == -1))
                    print(f"EMA psnr: {ema_psnr.item()}")
                    print(f"EMA psnr2: {ema_psnr2.item()}")
                    print(f"EMA fn: {ema_fn}, fp: {ema_fp}, exterior: {n_ext}")
                scene.save(iteration)
                if ema_params is not None:
                    for nm in ema_names:
                        getattr(gaussians, nm).copy_(_bak[nm])
                torch.cuda.synchronize()
                save_seconds += time.perf_counter() - _save_t0
            #     cpu_cells = cells.cpu().numpy()
            #     tensor_to_vtk(cpu_cells.reshape(cell_count, cell_count, cell_count), f"out_vtk/test_{iteration}.vtk", spacing)
            #     tensor_to_vtk(torch.abs((cells - gt)).cpu().numpy().reshape(cell_count, cell_count, cell_count), f"out_vtk/test_{iteration}_loss.vtk", spacing)
            #     vtk_files.append({
            #         "name": f"test_{iteration}.vtk",
            #         "time": float(saving_iterations.index(iteration))
            #     })
            #     vtk_files_loss.append({
            #         "name": f"test_{iteration}_loss.vtk",
            #         "time": float(saving_iterations.index(iteration))
            #     })

            _nv("densify")
            # Densification
            if (iteration <= opt.densify_until_iter and
                iteration >= opt.densify_from_iter and
                iteration % opt.densification_interval == 0 and
                 iteration not in saving_iterations and
                 iteration not in testing_iterations
            ):
                if use_mcmc:
                    dead_mask = (gaussians.get_weight <= 0.005).squeeze(-1)
                    gaussians.relocate_gs(dead_mask=dead_mask, cells=cells, gt=gt)
                    gaussians.add_new_gs(cap_max=args.cap_max)

                elif args.densify_3dgs:
                    # Clone/split by position-gradient magnitude. grad_threshold is 0:
                    # the stock 0.0002 is tuned for image-space 3DGS and never fires
                    # here, which would leave the arm far short of the target count and
                    # make the comparison meaningless.
                    budget = int(
                        args.cap_max - gaussians.get_values.shape[0]
                        + torch.count_nonzero(gaussians.get_weight <= 0.005)
                        + (n_final - args.cap_max)
                    )
                    if args.densify_batch > 0:
                        k3 = min(args.densify_batch, budget)
                    else:
                        remaining = max(1, args.densify_events - densifies)
                        k3 = min(-(-budget // remaining), budget)
                    k3 = max(0, k3)
                    g = gaussians._xyz.grad
                    if k3 > 0 and g is not None:
                        n_cl = gaussians.densify_and_clone(g, 0.0, 1.0, max_new_points=k3)
                        left = max(0, k3 - int(n_cl or 0))
                        if left > 0:
                            # NOT gaussians._xyz.grad: densify_and_clone replaced
                            # _xyz with a new tensor, so .grad is None by now. Reuse the
                            # gradient captured before cloning; densify_and_split already
                            # handles grads shorter than the current point count.
                            gaussians.densify_and_split(
                                g, 0.0, 1.0, max_new_points=left)
                        densifies += 1
                        gaussians.prune_points(
                            (gaussians.get_weight <= 0.005).squeeze(-1))
                else:
                    cells_flat = cells.ravel()
                    gt_flat = gt.ravel()
                    mask = gt_flat != -1

                    diff = torch.abs(cells_flat[mask] - gt_flat[mask])

                    budget = int(
                        args.cap_max - gaussians.get_values.shape[0]
                        + torch.count_nonzero(gaussians.get_weight <= 0.005)
                        + (n_final - args.cap_max)
                    )
                    if args.densify_batch > 0:
                        k = min(args.densify_batch, budget)
                    else:
                        # Auto: spread the remaining budget over a fixed number of
                        # densification events, so the batch size scales with cap_max,
                        # init size and iteration count instead of being a constant that
                        # silently becomes all-at-once at high compression.
                        remaining = max(1, args.densify_events - densifies)
                        k = min(-(-budget // remaining), budget)

                    k = max(0, min(k, diff.numel()))  # ensure k is valid

                    if k > 0:
                      if args.densify_alpha <= 0:
                          # --densify_alpha 0 restores the old pure-topk placement.
                          topk_idx_masked = torch.topk(diff, k).indices
                      else:
                          # Sample cells without replacement with probability proportional
                          # to err**alpha, via the Gumbel top-k trick (still a single
                          # topk). Plain topk aims the whole budget at a thin tail;
                          # spreading placement over the error *mass* uses the budget
                          # better. alpha=1.5 gained +0.14 to +1.79 dB in struct.
                          logits = args.densify_alpha * torch.log(diff + 1e-12)
                          u = torch.rand_like(diff).clamp_min(1e-20)
                          gumbel = -torch.log(-torch.log(u))
                          topk_idx_masked = torch.topk(logits + gumbel, k).indices
                      loss_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)[topk_idx_masked]

                      gaussians.densify_and_prune(
                          opt.densify_grad_threshold,
                          min_weight,
                          # new_scale,
                          torch.mean(gaussians.get_scaling) / 6.0,
                        # current_samples[np.logical_and(current_samples == -1, gt != -1)],
                        # gt_cells.ravel()[np.logical_and(cpu_cells.ravel() == -1, gt_cells.ravel() != -1)].reshape(-1, 1)
                          current_samples[loss_idx],
                          gt[loss_idx].reshape(-1, 1),
                          k
                      )
                      # The EMA buffers (gaussians.ema_buffers) follow every row
                      # mutation through GaussianModel's _ema_prune / _ema_extend /
                      # _ema_reset hooks (struct de47fcd), so nothing to replay here.
                densifies += 1

            _nv(); _nv("adam+ema")
            # Optimizer step
            if iteration < opt.iterations:
                # One Adam launch over all five tensors, with the Polyak-average update
                # riding along in the same pass once the EMA is live (struct d0b03fb /
                # f3cc97d). MULTI_ADAM=0 / FUSED_EMA=0 restore the torch paths.
                ema_ready = (args.ema > 0 and iteration >= args.ema_from and ema_params is not None
                             and ema_params["_xyz"].shape[0] == gaussians._xyz.shape[0])
                ema_done = gaussians.optimizer_step(ema=ema_params if ema_ready else None,
                                                    ema_w=1.0 - args.ema)
                gaussians.optimizer.zero_grad(set_to_none=True)

                if args.ema > 0 and iteration >= args.ema_from:
                    cur = {nm: getattr(gaussians, nm) for nm in ema_names}
                    if (ema_params is None
                            or ema_params["_xyz"].shape[0] != cur["_xyz"].shape[0]):
                        # First use. A shape mismatch means a row mutation bypassed the
                        # hooks; restarting is safe but shortens the window, so say so.
                        if ema_params is not None:
                            print(f"[ema] buffer desync at iter {iteration}: "
                                  f"{ema_params['_xyz'].shape[0]} -> {cur['_xyz'].shape[0]}; "
                                  f"restarting the average")
                        ema_params = {nm: cur[nm].detach().clone() for nm in ema_names}
                        gaussians.ema_buffers = ema_params
                    elif not ema_done:
                        # One fused kernel for all five tensors.
                        torch._foreach_lerp_(
                            [ema_params[nm] for nm in ema_names],
                            [cur[nm].detach() for nm in ema_names],
                            1.0 - args.ema)
                    # The hooks replace buffer tensors (prune/cat), so re-read them.
                    ema_params = gaussians.ema_buffers

                if use_mcmc:
                    L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)

                    def op_sigmoid(x, k=100, x0=0.995):
                        return 1 / (1 + torch.exp(-k * (x - x0)))

                    noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_weight)) * args.noise_lr * xyz_lr
                    noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                    gaussians._xyz.add_(noise)


            _nv(); _nv()  # adam+ema, iter
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



def _max_init(args, n_final):
    """Cap on the initial Gaussian count: --init_share_max of the final count and/or an
    absolute --init_max (both 0 = no cap)."""
    caps = []
    if args.init_share_max > 0:
        caps.append(args.init_share_max * n_final)
    if args.init_max > 0:
        caps.append(args.init_max)
    return min(caps) if caps else None


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
    parser.add_argument("--precomputed_samples", action="store_true")
    parser.add_argument("--precomputed_prefix", type=str,
                        default="/lus/eagle/projects/dist_relational_alg/ldyken53/earthquake2big",
                        help="Prefix for <prefix>_gt.npy / <prefix>_samples.npy with --precomputed_samples.")
    parser.add_argument("--stream_samples", action="store_true",
                        help="Keep the sample/GT arrays in pinned host memory and copy one "
                             "batch to the GPU per iteration (~2 ms) instead of holding the "
                             "whole set (and a sorted copy) on the GPU. Needed for sets "
                             "beyond ~200 batches at 64x.")
    parser.add_argument("--encode_surface", action="store_true")
    parser.add_argument("--loss", type=str, default="l1", choices=["l1", "l2"])
    # Each iteration renders the whole 2,097,152-point jitter batch, and the per-
    # iteration cost is dominated by that sample count, not by the model: at 1024x a
    # 4,539-Gaussian rbl run is SLOWER per iteration than an 88,659-Gaussian
    # earthquake one. That puts a ~12 ms/iteration floor under every run and is why
    # the two small datasets lose to UGINR on wall clock. --sample_frac renders a
    # fresh random subset of each batch instead; the points come from the same
    # distribution, there are just fewer of them per step.
    parser.add_argument("--sample_frac", type=float, default=1.0,
                        help="Fraction of each jitter batch rendered per iteration (1.0 = all).")
    # "stratified" takes one random point from each consecutive run of `stride` points
    # in the Morton-ordered batch. The indices come out already ordered, so it needs no
    # sort (the i.i.d. path pays 0.17 ms/iteration for one), it cannot leave a spatial
    # gap the way an i.i.d. draw can, and it is a lower-variance estimator of the same
    # quantity. "random" is the plain i.i.d. draw, kept for comparison.
    parser.add_argument("--sample_mode", type=str, default="stratified",
                        choices=["stratified", "random"])
    parser.add_argument("--fn_mode", type=str, default="abs", choices=["abs", "rel", "log"],
                        help="abs: fn_reg is an absolute weight. rel: fn_reg multiplies the data loss. "
                             "log: log barrier log(fn_thresh / aw) (use with a SOFT_CUTOFF 1 build).")
    parser.add_argument("--hinge_freeze", type=int, default=0,
                        help="with --hinge_rel_pow: freeze the relative hinge scale at this iteration (0 = never)")
    parser.add_argument("--hinge_rel_pow", type=float, default=0.0,
                        help="if > 0: FN and FP hinge weights x (detached data loss)^p; 0.5 with --loss l2 = RMSE scaling")
    parser.add_argument("--fnfp_norm", type=str, default="viol", choices=["viol", "all"],
                        help="FN/FP sums divided by their violators (viol) or by all interior/exterior cells (all)")
    parser.add_argument("--interior_skin", type=float, default=0.0,
                        help="--encode_surface: share of each batch sampled in a band just inside the surface")
    parser.add_argument("--fn_margin", type=float, default=0.05,
                        help="--encode_surface FN hinge target (1 + fn_margin) * OCC_T")
    parser.add_argument("--fp_margin", type=float, default=0.05,
                        help="--fp_mode hinge: penalise exterior aw above (1 - fp_margin) * OCC_T")
    parser.add_argument("--fp_mode", type=str, default="lin", choices=["lin", "log", "margin", "hinge"],
                        help="--encode_surface FP term per covered exterior cell: lin = its weight, "
                             "log = log(weight / SUPPORT_T) (zero at the edge of coverage, 1/w gradient), "
                             "margin = sum over Gaussians reaching it of qmax - q (margin.cu)")
    parser.add_argument("--surf_mask_data", action="store_true",
                        help="--encode_surface: data term over interior cells only (exterior handled by the FP term)")
    parser.add_argument("--fn_rel", action="store_true",
                        help="scale fn_reg by the (detached) data loss for any --fn_mode, as rel does "
                             "for the hinge: keeps the FN/data ratio fixed across PSNR regimes")
    parser.add_argument("--fn_thresh", type=float, default=0.002,
                        help="Accumulated-weight threshold of the --fn_mode log barrier.")
    # Default 0 = auto: spread the budget over --densify_events events instead of a
    # fixed count per event (struct c159f17). A fixed cap silently becomes
    # all-at-once at high compression. Pass a positive value for fixed batches.
    parser.add_argument("--densify_batch", type=int, default=0,
                        help="Gaussians added per densification event; <=0 uses --densify_events.")
    # Densification-method ablation. Default (no flag) = error-based placement with
    # probability sampling, i.e. --densify_alpha 1.5.
    #   --densify_alpha 0  -> error-based, plain top-k
    #   --use_mcmc         -> MCMC relocate/add (relocate_gs + add_new_gs + xyz noise)
    #   --densify_3dgs     -> 3DGS clone/split on position gradients
    # NOTE: the EMA buffers follow rows through prune_points / densification_postfix /
    # _write_rows, but MCMC's relocate overwrites rows without restarting their average,
    # and these ablations were run without EMA. Both flags force --ema 0 and say so.
    parser.add_argument("--use_mcmc", action="store_true",
                        help="MCMC relocate/add densification instead of error-based.")
    parser.add_argument("--densify_3dgs", action="store_true",
                        help="3DGS clone/split densification instead of error-based.")
    parser.add_argument("--densify_alpha", type=float, default=1.5,
                        help="Error-mass densification: sample cells with p ~ err**alpha; <=0 restores pure topk.")
    # Ship the Polyak average of the last ~100 iterations instead of the last
    # iterate. Late training orbits the optimum (each iteration fits a different
    # jitter batch); averaging the orbit gained +1.3 to +2.1 dB on mito and was
    # never worse on sf1/valley. 0 disables and ships the raw last iterate.
    parser.add_argument("--ema", type=float, default=0.99,
                        help="Polyak-average decay for the shipped model; 0 disables.")
    parser.add_argument("--ema_from", type=int, default=-1,
                        help="Iteration the EMA starts at; -1 = iterations - 100.")
    parser.add_argument("--densify_events", type=int, default=3,
                        help="With --densify_batch 0, spread the budget over this many events.")
    parser.add_argument("--max_scale", type=float, default=0.02)
    parser.add_argument("--max_scale_k", type=float, default=0.0,
                        help="if > 0: max_scale = max(--max_scale, k * N^(-1/3)), N the final Gaussian count")
    parser.add_argument("--cap_exact", action="store_true",
                        help="end densification at exactly cap_max Gaussians (struct) instead of cap_max + 1000")
    parser.add_argument("--init_share_max", type=float, default=0.0,
                        help="if > 0: cap the initial Gaussian count at this share of the final count (struct: 0.25)")
    parser.add_argument("--samples_per_node", type=float, default=0.0,
                        help="if > 0: sample_frac = min(1, this * mesh node count / batch size)")
    parser.add_argument("--samples_per_gaussian", type=float, default=0.0,
                        help="if > 0: sample_frac = min(1, this * final Gaussian count / batch size)")
    parser.add_argument("--init_max", type=int, default=0,
                        help="if > 0: cap the initial Gaussian count at this absolute number")
    parser.add_argument("--init_weight", type=float, default=0.1,
                        help="initial Gaussian weight (struct: 0.01)")
    parser.add_argument("--bvh_cache", type=int, default=1,
                        help="1: build each full batch's sample BVH once and reuse it (no per-iteration build)")
    parser.add_argument("--leaf_order", type=int, default=1,
                        help="1: permute each sample batch into BVH leaf order at startup (coalesced render loads)")
    parser.add_argument("--max_scale_vol", action="store_true",
                        help="--max_scale_k spacing from the mesh bounding-box volume, (V/N)^(1/3), not N^(-1/3)")
    parser.add_argument("--densify_growth", type=float, default=0.0,
                        help="if > 1: set --densify_events so each event grows the model by about this factor")
    parser.add_argument(
        "--sample_cache_dir", type=str,
        default="/home/ldyken53/gaussian-volume/sample_cache",
        help="Cache dir for the sampled ground truth; '' disables.")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    # The 0.02 default is calibrated for the interior-only task, where nothing
    # opposes the FN barrier. With --encode_surface the fp term actively pushes
    # weights down and 0.02 loses outright (25% of interior samples go dead).
    # 0.52 sits on the measured knee: FN <= 37 on all nine configs with the
    # lowest FP the knee allows. Explicit --fn_reg always wins.
    if args.encode_surface and "--fn_reg" not in sys.argv[1:]:
        args.fn_reg = 0.52
        print("encode_surface: fn_reg defaulting to 0.52 (pass --fn_reg to override)")
    if (args.use_mcmc or args.densify_3dgs) and args.ema > 0:
        print("[ablation] --use_mcmc/--densify_3dgs do not journal row mutations for "
              "the EMA; forcing --ema 0 to avoid silently averaging unrelated "
              "Gaussians. Use an --ema 0 control for a matched comparison.", flush=True)
        args.ema = 0.0
    if args.ema_from < 0:
        args.ema_from = max(1, args.iterations - 100)
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
        args.precomputed_prefix,
        args.encode_surface,
        args.sample_cache_dir,
        args.stream_samples
    )

    # All done
    print("\nTraining complete.")
