"""Where training and evaluation samples come from.

sample_mesh_points / sample_exterior_points draw points inside / just outside a volume mesh.
MeshSamples and GridSamples are train.py's two sample sources: batches of points in (and, for
surface models, around) a mesh, rendered through the BVH; and jittered lattices over a regular
grid, rendered by the tile rasterizer. Both give train.py one interface: next() draws an
iteration's samples and returns their ground truth (-1 outside a mesh), render() renders the
model at them, and positions() maps sample indices to points for densification.
"""

import math
import os
import time

import numpy as np
import pyvista as pv
import torch

from gaussian_renderer import (
    build_bvh,
    build_bvh_cached,
    margin_set_exterior,
    morton3d_unit,
    render_grid,
    render_points,
    sample_bvh_prim_ids,
)

# ---- Points in and around a mesh ----


def sample_mesh_points(mesh, num_batches, batch_size, device="cuda"):
    """Uniform random cells, then a uniform point inside each (dense where cells are small)."""
    points = torch.tensor(mesh.points, dtype=torch.float32, device=device)
    tet_mesh = mesh.triangulate()
    cells = torch.tensor(
        tet_mesh.cells.reshape(-1, 5)[:, 1:], dtype=torch.long, device=device
    )

    total = num_batches * batch_size
    cell_idx = torch.randint(0, cells.shape[0], (total,), device=device)
    verts = points[cells[cell_idx]]

    # Sorted uniforms; their gaps are uniform barycentric coordinates in a tetrahedron.
    u = torch.rand(total, 3, device=device).sort(dim=1).values
    bary = torch.zeros(total, 4, device=device)
    bary[:, 0] = u[:, 0]
    bary[:, 1] = u[:, 1] - u[:, 0]
    bary[:, 2] = u[:, 2] - u[:, 1]
    bary[:, 3] = 1.0 - u[:, 2]

    samples = (bary.unsqueeze(2) * verts).sum(dim=1)
    return samples.reshape(num_batches, batch_size, 3)


def sample_exterior_points(
    mesh, num_batches, batch_size, offset=0.01, device="cuda", inward=False
):
    """Points up to `offset` outside the surface along its outward normals (inside if inward)."""
    surf = (
        mesh.extract_surface()
        .triangulate()
        .compute_normals(cell_normals=True, point_normals=False)
    )
    face_normals = torch.tensor(
        surf.cell_data["Normals"], dtype=torch.float32, device=device
    )
    points = torch.tensor(surf.points, dtype=torch.float32, device=device)
    faces = torch.tensor(
        surf.faces.reshape(-1, 4)[:, 1:], dtype=torch.long, device=device
    )

    total = num_batches * batch_size
    face_idx = torch.randint(0, faces.shape[0], (total,), device=device)
    v = points[faces[face_idx]]

    u = torch.rand(total, 2, device=device)
    sqrt_u0 = u[:, 0].sqrt()
    bary = torch.stack([1 - sqrt_u0, sqrt_u0 * (1 - u[:, 1]), sqrt_u0 * u[:, 1]], dim=1)
    surf_pts = (bary.unsqueeze(2) * v).sum(dim=1)

    normals = face_normals[face_idx]
    dist = torch.rand(total, 1, device=device) * offset
    samples = surf_pts + normals * (-dist if inward else dist)
    return samples.reshape(num_batches, batch_size, 3)


# ---- Mesh models: batches of samples rendered through the BVH ----

# Fixed seed for the sample points, so the cached ground truth matches every run's points.
SAMPLE_SEED = 20260903
BATCH_SIZE = 128**3
NUM_BATCHES = 100


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


class MeshSamples:
    """NUM_BATCHES fixed batches of points in (and, for surface models, around) a mesh, with
    their ground truth. Each iteration renders a random batch, or a subsample of one
    (--samples_per_node / --sample_frac), at the points of a BVH built over them."""

    def __init__(self, args, mesh, surface):
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
        big_samples_cuda = big_samples_cuda.gather(
            1, order.unsqueeze(-1).expand(-1, -1, 3)
        )
        big_gt_cuda = big_gt_cuda.gather(1, order)

        if args.samples_per_node > 0:
            # About this many samples per mesh node per iteration, at most the whole batch.
            args.sample_frac = min(
                1.0, args.samples_per_node * int(mesh.n_points) / size
            )
            print(
                f"samples_per_node {args.samples_per_node:g}: "
                f"sample_frac {args.sample_frac:.4f}"
            )
        full_batch = not (0.0 < args.sample_frac < 1.0)

        # Permute each batch into its sample BVH's leaf order, so render loads are
        # near-sequential; full-batch runs also keep each batch's BVH.
        for b in range(num_batches):
            build_bvh(big_samples_cuda[b])
            perm = sample_bvh_prim_ids().long()
            big_samples_cuda[b] = big_samples_cuda[b][perm]
            big_gt_cuda[b] = big_gt_cuda[b][perm]
            if full_batch:
                build_bvh_cached(big_samples_cuda[b], b)
        torch.cuda.synchronize()
        build_bvh(big_samples_cuda[0])

        # --fp_mode margin: each batch's exterior samples, registered for margin_fp.
        self.margin_n_ext = None
        if surface and args.fp_mode == "margin":
            self.margin_n_ext = []
            for b in range(num_batches):
                ext = big_samples_cuda[b][big_gt_cuda[b] == -1]
                margin_set_exterior(ext, b)
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
        self.gt_all, self.samples_all = big_gt_cuda, big_samples_cuda
        self.num_batches, self.size = num_batches, size
        self.batch, self.samples = 0, None

    def next(self, gaussians):
        """A random batch (or a subsample of one) as the render's samples; returns its gt."""
        self.batch = np.random.randint(0, self.num_batches)
        gt = self.gt_all[self.batch]
        samples = self.samples_all[self.batch]
        if self.sub_k > 0:
            # Indices stay ascending, as the BVH build expects.
            if self.sub_base is not None:
                sub_idx = self.sub_base + torch.randint(
                    0, self.sub_stride, (self.sub_k,), device="cuda"
                )
            else:
                sub_idx = (
                    torch.randint(0, self.size, (self.sub_k,), device="cuda")
                    .sort()
                    .values
                )
            gt = gt[sub_idx]
            samples = samples[sub_idx]
            build_bvh(samples)
        else:
            build_bvh_cached(samples, self.batch)
        self.samples = samples
        return gt

    def render(self, gaussians):
        return render_points(gaussians)

    def positions(self, idx):
        return self.samples[idx]


# ---- Grid models: jittered lattices rendered by the tile rasterizer ----

CELL_COUNT = 128


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
        (gaussians.maxes[i] - gaussians.mins[i]) / (cell_count - 1) for i in range(3)
    ]
    axes = [
        np.linspace(gaussians.mins[i], gaussians.maxes[i], cell_count) for i in range(3)
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


class GridSamples:
    """A regular grid's ground truth on a CELL_COUNT^3 lattice over its domain: --jitter_gt K
    precomputed jittered lattices with the trilinear interpolant (default), raw node values on
    a strided subgrid at a random offset (--jitter_gt 0), or the grid probed at 100 fixed
    jitters (--probe_gt). Each iteration renders a random one."""

    def __init__(self, args, gaussians):
        self.jitter_gt = args.jitter_gt
        self.node = None
        if args.probe_gt:
            self.big_gt, self.big_jitter, self.big_samples = probe_gt(
                gaussians, CELL_COUNT
            )
            self.num_jitters = self.big_gt.shape[0]
        else:
            self.node = NodeSampler(gaussians.mesh, CELL_COUNT)
            if args.jitter_gt > 0:
                self.node.build_jitter_sets(args.jitter_gt)
            else:
                self.jitter = torch.zeros(
                    3 * CELL_COUNT**3, dtype=torch.float, device="cuda"
                )
        self.points = self.node

    def next(self, gaussians):
        """A random lattice as the render's cells (the model's bounds follow it); returns
        its gt."""
        if self.node is not None:
            if self.jitter_gt > 0:
                gt, self.jitter = self.node.draw_jit(np.random.randint(self.jitter_gt))
            else:
                gt = self.node.draw(True)
            gaussians.mins, gaussians.maxes = self.node.bounds()
        else:
            jit_idx = np.random.randint(0, self.num_jitters)
            self.jitter = self.big_jitter[jit_idx].ravel()
            gt = self.big_gt[jit_idx].reshape(CELL_COUNT, CELL_COUNT, CELL_COUNT)
            self.points = self.big_samples[jit_idx]
        return gt

    def render(self, gaussians):
        return render_grid(gaussians, self.jitter, CELL_COUNT)

    def positions(self, idx):
        return self.points[idx]
