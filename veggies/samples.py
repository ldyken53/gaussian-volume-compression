"""What a model trains on. MeshSamples: fixed batches of points in (and, for surface models,
around) a mesh, sampled through a BVH. GridSamples: jittered lattices over a regular grid, sampled
by the tile rasterizer. Both give train.py the same interface: next() picks an iteration's samples
and returns their ground truth (-1 outside a mesh), render() samples the model there, and
positions() maps sample indices to points for densification."""

import math
import os
import time

import numpy as np
import pyvista as pv
import torch

from veggies import sampling
from veggies.utils import morton_codes


def sample_mesh_points(mesh, num_batches, batch_size, importance=0.0, device="cuda"):
    """Uniform random cells, then a uniform point inside each (dense where cells are small).
    With importance, that share of the cells is drawn with probability proportional to the cell's
    value range instead."""
    points = torch.tensor(mesh.points, dtype=torch.float32, device=device)
    tets = mesh.triangulate()
    cells = torch.tensor(
        tets.cells.reshape(-1, 5)[:, 1:], dtype=torch.long, device=device
    )
    total = num_batches * batch_size
    pick = torch.randint(0, cells.shape[0], (total,), device=device)
    if importance > 0:
        name = mesh.array_names[0]
        assert name in mesh.point_data, "--importance needs a point-data array"
        vals = torch.tensor(mesh.point_data[name], dtype=torch.float32, device=device)
        rr = vals[cells]
        cdf = (rr.amax(1) - rr.amin(1)).double().cumsum(0)
        del rr
        hot = torch.rand(total, device=device) < importance
        u = torch.rand(int(hot.sum()), device=device, dtype=torch.float64) * cdf[-1]
        pick[hot] = torch.searchsorted(cdf, u).clamp_(max=cells.shape[0] - 1)
        del cdf, u, hot
    corners = points[cells[pick]]
    del pick
    # The gaps of sorted uniforms are uniform barycentric coordinates in a tetrahedron.
    u = torch.rand(total, 3, device=device).sort(dim=1).values
    bary = torch.zeros(total, 4, device=device)
    bary[:, 0] = u[:, 0]
    bary[:, 1] = u[:, 1] - u[:, 0]
    bary[:, 2] = u[:, 2] - u[:, 1]
    bary[:, 3] = 1.0 - u[:, 2]
    return (bary.unsqueeze(2) * corners).sum(dim=1).reshape(num_batches, batch_size, 3)


def sample_exterior_points(
    mesh, num_batches, batch_size, offset=0.01, device="cuda", inward=False
):
    """Points up to offset outside the surface along its normals (inside if inward)."""
    surface = (
        mesh.extract_surface()
        .triangulate()
        .compute_normals(cell_normals=True, point_normals=False)
    )
    normals = torch.tensor(
        surface.cell_data["Normals"], dtype=torch.float32, device=device
    )
    points = torch.tensor(surface.points, dtype=torch.float32, device=device)
    faces = torch.tensor(
        surface.faces.reshape(-1, 4)[:, 1:], dtype=torch.long, device=device
    )
    total = num_batches * batch_size
    face = torch.randint(0, faces.shape[0], (total,), device=device)
    v = points[faces[face]]
    u = torch.rand(total, 2, device=device)
    s0 = u[:, 0].sqrt()
    bary = torch.stack([1 - s0, s0 * (1 - u[:, 1]), s0 * u[:, 1]], dim=1)
    on_surface = (bary.unsqueeze(2) * v).sum(dim=1)
    dist = torch.rand(total, 1, device=device) * offset
    return (on_surface + normals[face] * (-dist if inward else dist)).reshape(
        num_batches, batch_size, 3
    )


# Mesh models.

SAMPLE_SEED = 20260903  # fixed, so cached ground truth matches every run's points
BATCH_SIZE = 128**3
NUM_BATCHES = 100


def sample_cache_path(cache_dir, source_path, surface, interior_skin, importance=0.0):
    if not cache_dir:
        return None
    stem = os.path.splitext(os.path.basename(source_path))[0]
    tag = f"{stem}_c128_b{NUM_BATCHES}_s{int(bool(surface))}"
    if surface and interior_skin > 0:
        tag += f"_skin{interior_skin:g}"
    if importance > 0:
        tag += f"_imp{importance:g}"
    return os.path.join(cache_dir, tag + "_gt.npy")


def load_samples(args, mesh):
    """(gt, samples) as (B, S) and (B, S, 3) arrays; gt is -1 outside the mesh."""
    if args.precomputed_samples:
        gt = np.load(args.precomputed_samples + "_gt.npy")
        samples = np.load(args.precomputed_samples + "_samples.npy")
        assert samples.shape[:2] == gt.shape
        print(f"Loaded precomputed samples {gt.shape} from {args.precomputed_samples}")
        return gt, samples

    surface = args.kind == "surface"
    size = BATCH_SIZE
    cache = sample_cache_path(
        args.sample_cache_dir,
        args.source_path,
        surface,
        args.interior_skin,
        args.importance,
    )
    torch.manual_seed(SAMPLE_SEED)
    if surface:
        # 20% just outside the surface and interior_skin just inside, so both sides of the
        # boundary are sampled at the same density per unit area.
        n_out = int(math.floor(size * 0.2))
        n_skin = int(math.floor(size * args.interior_skin))
        parts = [
            sample_mesh_points(
                mesh, NUM_BATCHES, size - n_out - n_skin, args.importance
            )
            .cpu()
            .numpy(),
            sample_exterior_points(mesh, NUM_BATCHES, n_out).cpu().numpy(),
        ]
        if n_skin > 0:
            parts.append(
                sample_exterior_points(mesh, NUM_BATCHES, n_skin, inward=True)
                .cpu()
                .numpy()
            )
        samples = np.concatenate(parts, axis=1)
    else:
        samples = (
            sample_mesh_points(mesh, NUM_BATCHES, size, args.importance).cpu().numpy()
        )

    if cache is not None and os.path.exists(cache):
        print(f"Loaded cached ground truth from {cache}")
        return np.load(cache), samples.reshape(NUM_BATCHES, size, 3)

    samples = samples.reshape(NUM_BATCHES * size, 3)
    probed = pv.PolyData(samples).sample(mesh)
    gt = probed[mesh.array_names[0]]
    gt[~probed["vtkValidPointMask"].astype(bool)] = -1
    gt = gt.reshape(NUM_BATCHES, size)
    samples = samples.reshape(NUM_BATCHES, size, 3)
    if cache is not None:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + f".tmp{os.getpid()}.npy"
        np.save(tmp, np.ascontiguousarray(gt, dtype=np.float32))
        os.replace(tmp, cache)
        print(f"Wrote ground-truth cache to {cache}")
    return gt, samples


def drop_exterior_samples(gt, samples):
    """Replace samples outside the mesh with other samples of the same batch (interior models)."""
    outside = gt == -1
    if not outside.any():
        return gt, samples
    rng = np.random.default_rng(SAMPLE_SEED)
    gt, samples = np.array(gt, copy=True), np.array(samples, copy=True)
    for b in range(gt.shape[0]):
        idx = np.flatnonzero(outside[b])
        if idx.size == 0:
            continue
        repl = rng.choice(np.flatnonzero(~outside[b]), size=idx.size, replace=False)
        gt[b, idx] = gt[b, repl]
        samples[b, idx] = samples[b, repl]
    print(f"replaced {int(outside.sum())} out-of-mesh samples with in-mesh ones")
    return gt, samples


class MeshSamples:
    """Each iteration samples a random batch, or a subsample of one (samples_per_node /
    sample_frac)."""

    def __init__(self, args, mesh, rule):
        surface = args.kind == "surface"
        gt, samples = load_samples(args, mesh)
        if not surface:
            gt, samples = drop_exterior_samples(gt, samples)
        num_batches, size = gt.shape
        gt = torch.tensor(gt, dtype=torch.float, device="cuda")
        samples = torch.tensor(samples, dtype=torch.float, device="cuda")

        # Morton-sort each batch: BVH builds and subsampling rely on the locality.
        order = (
            morton_codes(torch.clamp(samples, 0.0, 1.0).reshape(-1, 3))
            .reshape(num_batches, size)
            .argsort(dim=1)
        )
        samples = samples.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))
        gt = gt.gather(1, order)

        if args.samples_per_node > 0:
            args.sample_frac = min(
                1.0, args.samples_per_node * int(mesh.n_points) / size
            )
            print(
                f"samples_per_node {args.samples_per_node:g}: sample_frac {args.sample_frac:.4f}"
            )
        full_batch = not (0.0 < args.sample_frac < 1.0)

        # Batches in their BVH leaf order make sample loads near-sequential; full-batch runs keep
        # each batch's BVH too.
        for b in range(num_batches):
            sampling.set_points(samples[b])
            perm = sampling.leaf_order().long()
            samples[b] = samples[b][perm]
            gt[b] = gt[b][perm]
            if full_batch:
                sampling.set_points_cached(samples[b], b)
        torch.cuda.synchronize()
        sampling.set_points(samples[0])

        self.margin_n_ext = None
        if surface and args.fp_mode == "margin":
            self.margin_n_ext = []
            for b in range(num_batches):
                ext = samples[b][gt[b] == -1]
                sampling.margin_set_exterior(ext, b)
                self.margin_n_ext.append(max(1, int(ext.shape[0])))

        self.sub_k, self.sub_stride, self.sub_base = 0, 1, None
        if not full_batch:
            if args.sample_mode == "stratified":
                # One draw from each run of sub_stride: ascending by construction, no gaps.
                self.sub_stride = max(2, int(round(1.0 / args.sample_frac)))
                self.sub_k = size // self.sub_stride
                self.sub_base = (
                    torch.arange(self.sub_k, device="cuda") * self.sub_stride
                )
            else:
                self.sub_k = max(1, int(round(size * args.sample_frac)))
            print(
                f"rendering {self.sub_k} of {size} samples per iteration ({args.sample_mode})"
            )
        self.rule = rule
        self.gt_all, self.samples_all = gt, samples
        self.num_batches, self.size = num_batches, size
        self.batch, self.samples = 0, None
        # While recording (set before next()), reach holds each Gaussian's largest w G at an
        # inside sample since the last exterior_gaussians() call.
        self.recording, self.reach = False, None

    @torch.no_grad()
    def exterior_gaussians(self, model, threshold):
        """Indices of the Gaussians with w >= threshold whose w G stayed below threshold at every
        inside sample rendered since the last call, then a fresh record. On surface models these
        are boundary Gaussians the FP hinge pushed past the exterior band, where no sample sees
        them. None if the record does not cover the current Gaussians."""
        reach, self.reach = self.reach, None
        if reach is None or reach.shape[0] != model.count:
            return None
        w = model.weights.reshape(-1)
        return torch.nonzero((reach < threshold) & (w >= threshold)).squeeze(1)

    def next(self, model):
        self.batch = np.random.randint(0, self.num_batches)
        gt = self.gt_all[self.batch]
        samples = self.samples_all[self.batch]
        if self.sub_k > 0:
            if self.sub_base is not None:
                idx = self.sub_base + torch.randint(
                    0, self.sub_stride, (self.sub_k,), device="cuda"
                )
            else:
                idx = (
                    torch.randint(0, self.size, (self.sub_k,), device="cuda")
                    .sort()
                    .values
                )
            gt, samples = gt[idx], samples[idx]
            sampling.set_points(samples)
        else:
            sampling.set_points_cached(samples, self.batch)
        if self.recording:
            sampling.set_point_flags((gt != -1).float())
        self.samples = samples
        return gt

    def render(self, model):
        reach = None
        if self.recording:
            if self.reach is None or self.reach.shape[0] != model.count:
                self.reach = torch.zeros(model.count, device="cuda")
            reach = self.reach
        return sampling.sample_points(model, self.rule, reach)

    def positions(self, idx):
        return self.samples[idx]


# Grid models.

CELL_COUNT = 128


class NodeSampler:
    """Ground truth from the native grid, resident on the GPU: K precomputed jittered lattices with
    the trilinear interpolant (jitter mode), or raw node values on an integer-strided subgrid at a
    fresh offset each time (node mode, --jitter_gt 0)."""

    def __init__(self, grid, cell_count):
        nx, ny, nz = grid.dimensions
        lo = np.array(grid.bounds[0::2])
        hi = np.array(grid.bounds[1::2])
        self.n = np.array([nx, ny, nz])
        self.lo = lo
        self.d = (hi - lo) / (self.n - 1)
        self.cc = cell_count
        self.stride = np.maximum((self.n - 1) // (cell_count - 1), 1)
        self.span = (cell_count - 1) * self.stride
        self.noff = self.n - self.span
        self.jittered = False
        # Grids are stored [z, y, x], the rasterizer's cell order. fp16 keeps the largest volumes
        # beside the model; its step is far below the models' error.
        values = grid.point_data[grid.point_data.keys()[0]]
        self.vol = torch.tensor(
            np.ascontiguousarray(values.reshape(nz, ny, nx), dtype=np.float16),
            device="cuda",
        )
        self.off = np.zeros(3, dtype=np.int64)
        print(
            f"Node sampling: stride {self.stride.tolist()}, {int(np.prod(self.noff))} offsets, "
            f"{self.vol.numel() * 2 / 1e9:.1f} GB resident"
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
        """Precompute k (jitter, trilinear ground truth) sets, then release the resident volume."""
        cc = self.cc
        ar = torch.arange(cc, device="cuda", dtype=torch.float32)
        zi, yi, xi = torch.meshgrid(ar, ar, ar, indexing="ij")
        base = [xi, yi, zi]
        n = [int(v) for v in self.n]
        st = [(n[a] - 1) / (cc - 1) for a in range(3)]
        self.jittered = True
        self.jit_step = st
        self.cur_jit = 0
        vol = self.vol.reshape(-1)
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
                return vol[((z0 + dz) * n[1] + (y0 + dy)) * n[0] + (x0 + dx)].float()

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
            f"Jittered GT: {k} sets, {(self.jit_gt.numel() + self.jit_jitter.numel()) * 2 / 1e9:.2f} GB "
            f"resident; volume ({vol_gb:.1f} GB) released"
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
        """Positions of flat cell indices ([z, y, x] order), for densification."""
        a = torch.div(idx, self.cc * self.cc, rounding_mode="floor")
        b = torch.div(idx % (self.cc * self.cc), self.cc, rounding_mode="floor")
        c = idx % self.cc
        node = torch.stack([c, b, a], dim=1).float()

        def t(v):
            return torch.tensor(v, dtype=torch.float, device=idx.device)

        if self.jittered:
            # The jittered position the error was measured at, not the bare lattice node.
            jit = self.jit_jitter[self.cur_jit].float().view(-1, 3)[idx]
            return t(self.lo) + node * t(self.jit_step) * t(self.d) + jit
        return t(self.lo) + (t(self.off) + node * t(self.stride)) * t(self.d)


def probe_gt(model, cell_count):
    """(gt, jitter, samples) for --probe_gt: the volume probed at 100 fixed jitters."""
    t0 = time.time()
    spacing = [
        (model.bounds_max[i] - model.bounds_min[i]) / (cell_count - 1) for i in range(3)
    ]
    axes = [
        np.linspace(model.bounds_min[i], model.bounds_max[i], cell_count)
        for i in range(3)
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    samples = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T.reshape(
        cell_count, cell_count, cell_count, 3
    )
    samples = np.flip(np.rot90(samples, k=1, axes=(2, 0)), axis=2).reshape(
        -1, 3
    )  # to [z, y, x]
    num_jitters = 100
    samples = np.tile(samples, (num_jitters, 1))
    jitter = np.random.uniform(-0.5, 0.5, samples.shape)
    jitter *= np.array(spacing)[None, :]
    jitter[: cell_count**3, :] = 0
    samples = np.clip(
        samples + jitter, np.array(model.bounds_min), np.array(model.bounds_max)
    )
    gt = (
        pv.PolyData(samples)
        .sample(model.volume)
        .point_data[model.volume.array_names[0]]
    )
    gt = gt.reshape(num_jitters, cell_count**3)
    samples = samples.reshape(num_jitters, cell_count**3, 3)
    jitter = jitter.reshape(num_jitters, cell_count**3, 3)
    print(f"Time to sample gt: {time.time() - t0}")

    def cuda(a):
        return torch.tensor(a, dtype=torch.float, device="cuda")

    return cuda(gt), cuda(jitter), cuda(samples)


class GridSamples:
    """Each iteration samples a random lattice: one of --jitter_gt K jittered lattices (default),
    the raw nodes of a strided subgrid at a random offset (--jitter_gt 0), or one of 100 probed
    jitters (--probe_gt)."""

    def __init__(self, args, model, rule):
        self.jitter_gt = args.jitter_gt
        self.rule = rule
        self.node = None
        if args.probe_gt:
            self.big_gt, self.big_jitter, self.big_samples = probe_gt(model, CELL_COUNT)
            self.num_jitters = self.big_gt.shape[0]
        else:
            self.node = NodeSampler(model.volume, CELL_COUNT)
            if args.jitter_gt > 0:
                self.node.build_jitter_sets(args.jitter_gt)
            else:
                self.jitter = torch.zeros(
                    3 * CELL_COUNT**3, dtype=torch.float, device="cuda"
                )
        self.points = self.node

    def next(self, model):
        if self.node is not None:
            if self.jitter_gt > 0:
                gt, self.jitter = self.node.draw_jit(np.random.randint(self.jitter_gt))
            else:
                gt = self.node.draw(True)
            model.bounds_min, model.bounds_max = self.node.bounds()
        else:
            i = np.random.randint(0, self.num_jitters)
            self.jitter = self.big_jitter[i].ravel()
            gt = self.big_gt[i].reshape(CELL_COUNT, CELL_COUNT, CELL_COUNT)
            self.points = self.big_samples[i]
        return gt

    def render(self, model):
        return sampling.sample_grid(model, self.jitter, CELL_COUNT, self.rule)

    def positions(self, idx):
        return self.points[idx]
