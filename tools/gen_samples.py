"""Training samples for a mesh too large to triangulate at once (train.py --precomputed_samples).

    python tools/gen_samples.py mesh.vtu out/mesh --batches 400 --size 2097152 [--surface]

Writes <out>_samples.npy (batches, size, 3) and <out>_gt.npy (batches, size), drawn like
train.py's own samples (uniform points in tets, --importance of the tets picked by value range)
and scaled the same way. The batches are built a chunk at a time from the tets of a subset of the
cells, so no triangulation holds the whole mesh. On meshes with non-tet cells the in-mesh values
are probed from the cells (--probe_interior), as eval.py scores them. With --surface, each batch's interior points are followed by --ext_share of the
batch up to --band_width outside the mesh and --interior_skin of it up to --band_width inside.
Finished chunks are recorded in <out>_progress.txt, and a rerun continues from there.
"""

import argparse
import os
import time
from typing import NamedTuple

import numpy as np
import pyvista as pv

from tets import (
    has_non_tets,
    points_in_tets,
    probe,
    read_mesh,
    scale_for_probe,
    triangulate,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("src", help="the mesh")
    p.add_argument("out", help="output prefix")
    p.add_argument("--array", help="the scalar array (default: the first)")
    p.add_argument("--batches", type=int, default=400)
    p.add_argument("--size", type=int, default=2097152, help="points per batch")
    p.add_argument(
        "--chunk_batches",
        type=int,
        default=25,
        help="batches per triangulated subset of cells",
    )
    p.add_argument(
        "--subset",
        type=int,
        default=40_000_000,
        help="cells triangulated per chunk; must be >> the chunk's interior points / 4",
    )
    p.add_argument(
        "--stratified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="split the cells disjointly across the chunks and draw each chunk's tets without "
        "replacement (default), instead of drawing tets with replacement from a random subset",
    )
    p.add_argument(
        "--importance",
        type=float,
        default=0.5,
        help="with --stratified: the share of each chunk's points drawn with probability "
        "proportional to their tet's value range (sharp features), the rest uniformly (0.5, "
        "train.py's default)",
    )
    p.add_argument(
        "--importance_pow",
        type=float,
        default=1.0,
        help="raise the value ranges to this power first; > 1 concentrates on the sharpest tets",
    )
    p.add_argument(
        "--stop_after", type=int, default=0, help="stop after this many chunks (0: all)"
    )
    p.add_argument(
        "--inner", type=int, default=4_000_000, help="points evaluated at once"
    )
    p.add_argument(
        "--surface",
        action="store_true",
        help="train.py's surface batches: --ext_share 0.2 --interior_skin 0.1",
    )
    p.add_argument(
        "--ext_share", type=float, default=0.0, help="share just outside the mesh"
    )
    p.add_argument(
        "--interior_skin", type=float, default=0.0, help="share just inside it"
    )
    p.add_argument(
        "--probe_interior",
        action=argparse.BooleanOptionalAction,
        help="take the in-mesh points' values from the mesh's own cells, as eval.py scores them, "
        "instead of linearly within the tets they were drawn from (default: when the mesh has "
        "non-tet cells, where the two differ)",
    )
    p.add_argument("--band_width", type=float, default=0.01)
    p.add_argument(
        "--band_pow", type=float, default=1.0, help="distance = u^p * --band_width"
    )
    a = p.parse_args()
    if a.surface:
        a.ext_share = a.ext_share or 0.2
        a.interior_skin = a.interior_skin or 0.1
    if a.batches % a.chunk_batches:
        p.error("--batches must be a multiple of --chunk_batches")
    if a.importance > 0 and not a.stratified:
        p.error(
            "--importance needs --stratified (use --importance 0 with --no-stratified)"
        )
    return a


class Band(NamedTuple):
    triangles: np.ndarray
    points: np.ndarray
    normals: np.ndarray  # outward, per triangle
    prob: np.ndarray  # area share, per triangle


def boundary(m, inner):
    """The mesh's boundary triangles with outward normals. A face's outward side is the one away
    from its own volume cell, so meshes whose surface is not consistently orientable still get
    their band on the right side. A non-conforming mesh's extract_surface also returns internal
    faces; a face is kept only if the point 1e-4 outside it is outside the mesh."""
    t = time.time()
    surf = m.extract_surface(pass_cellid=True).triangulate()
    tri = surf.faces.reshape(-1, 4)[:, 1:]
    sp = np.asarray(surf.points, dtype=np.float64)
    nrm = np.cross(sp[tri[:, 1]] - sp[tri[:, 0]], sp[tri[:, 2]] - sp[tri[:, 0]])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm /= np.maximum(2.0 * area, 1e-30)[:, None]
    owner = np.asarray(surf.cell_data["vtkOriginalCellIds"])
    uniq, inv = np.unique(owner, return_inverse=True)
    centers = np.asarray(m.extract_cells(uniq).cell_centers().points, dtype=np.float64)
    fc = sp[tri].mean(1)
    flip = ((fc - centers[inv]) * nrm).sum(1) < 0
    nrm[flip] *= -1
    outer = np.empty(tri.shape[0], dtype=bool)
    for lo in range(0, tri.shape[0], inner):
        out = pv.PolyData(
            (fc[lo : lo + inner] + 1e-4 * nrm[lo : lo + inner]).astype(np.float32)
        )
        outer[lo : lo + inner] = ~np.asarray(out.sample(m)["vtkValidPointMask"]).astype(
            bool
        )
    print(
        f"surface {tri.shape[0]} triangles in {time.time() - t:.0f}s, "
        f"{100 * flip.mean():.1f}% of face normals flipped outward, "
        f"{100 * outer.mean():.1f}% of faces ({100 * area[outer].sum() / area.sum():.1f}% of the area) "
        "on the true boundary",
        flush=True,
    )
    tri, nrm, area = tri[outer], nrm[outer], area[outer]
    return Band(tri, sp, nrm, area / area.sum())


def band_points(band, rng, n, inward, width, power):
    """n points up to width from the surface (area-uniform faces), outside or inside."""
    f = rng.choice(band.triangles.shape[0], size=n, p=band.prob)
    u = rng.random((n, 2))
    s0 = np.sqrt(u[:, 0])
    b = np.stack([1 - s0, s0 * (1 - u[:, 1]), s0 * u[:, 1]], 1)
    p = (b[:, :, None] * band.points[band.triangles[f]]).sum(1)
    d = rng.random((n, 1)) ** power * width
    return (p + band.normals[f] * (-d if inward else d)).astype(np.float32)


def importance_pick(tets, vals, n, share, power, rng):
    """n tets with replacement: share of them with probability proportional to the tet's value
    range^power, the rest uniformly."""
    ntet = tets.shape[0]
    rr = vals[tets]
    rr = rr.max(1) - rr.min(1)
    rp = rr**power if power != 1.0 else rr
    pw = (1.0 - share) / ntet + share * rp / max(rp.sum(), 1e-30)
    del rp
    pw /= pw.sum()
    pick = rng.choice(ntet, size=n, replace=True, p=pw)
    hi = rr > 0.05
    print(
        f"  importance {share}: tets with range > 0.05 are {hi.mean() * 100:.2f}% of tets "
        f"and get {hi[pick].mean() * 100:.1f}% of samples "
        f"(enrichment {hi[pick].mean() / max(hi.mean(), 1e-12):.1f}x); "
        f"distinct tets {len(np.unique(pick)) / n * 100:.1f}%",
        flush=True,
    )
    return pick


def stratified_pick(tets, source, n_cells, n, a, vals, rng):
    """The tet of each of a chunk's n interior points, in random order."""
    counts = np.bincount(source, minlength=n_cells)
    assert counts.min() >= 1, "a cell produced no tets"
    ntet = tets.shape[0]
    if a.importance > 0:
        pick = importance_pick(tets, vals, n, a.importance, a.importance_pow, rng)
    else:
        pick = rng.choice(ntet, size=min(n, ntet), replace=False)
        if pick.shape[0] < n:
            pad = rng.choice(ntet, size=n - pick.shape[0], replace=False)
            pick = np.concatenate([pick, pad])
    pick = pick[rng.permutation(pick.shape[0])]
    share = min(1.0, n / ntet)
    print(
        f"  stratified: {n_cells} cells, {ntet} tets ({counts.mean():.2f}/cell), "
        f"{100.0 * share:.1f}% of tets, "
        f"~{100.0 * (1.0 - (1.0 - share) ** counts.mean()):.1f}% of cells covered",
        flush=True,
    )
    return pick


def interior_points(tets, pick, pts, vals, n, inner, rng):
    """n points: one in each tets[pick], or (pick None) in tets drawn with replacement."""
    samples = np.empty((n, 3), dtype=np.float32)
    gt = np.empty((n,), dtype=np.float32)
    for lo in range(0, n, inner):
        hi = min(lo + inner, n)
        v = (
            tets[pick[lo:hi]]
            if pick is not None
            else tets[rng.integers(0, tets.shape[0], size=hi - lo)]
        )
        s, g = points_in_tets(v, pts, vals, rng)
        samples[lo:hi] = s.astype(np.float32)
        gt[lo:hi] = g.astype(np.float32)
    return samples, gt


def main():
    a = parse_args()
    size_ext = int(a.size * a.ext_share)
    size_skin = int(a.size * a.interior_skin)
    size_int = a.size - size_ext - size_skin
    n_chunks = a.batches // a.chunk_batches
    per_chunk = a.chunk_batches * size_int
    progress = a.out + "_progress.txt"
    done = 0
    if os.path.exists(progress):
        with open(progress) as f:
            done = int(f.read().strip() or 0)
        print(f"resuming: {done}/{n_chunks} chunks already written", flush=True)

    m, name, vals, pts = read_mesh(a.src, a.array, np.float32)
    print(
        f"normalised: vals[{vals.min():.4f},{vals.max():.4f}] pts[{pts.min():.4f},{pts.max():.4f}]",
        flush=True,
    )
    if a.probe_interior is None:
        a.probe_interior = has_non_tets(m)
    print(
        f"in-mesh values: {'probed from the cells' if a.probe_interior else 'linear in the tets'}",
        flush=True,
    )
    band = None
    if size_ext or size_skin or a.probe_interior:
        # The probe reads the mesh itself, so it gets the same scaling.
        scale_for_probe(m, name, vals, pts)
    if size_ext or size_skin:
        band = boundary(m, a.inner)

    mode = "r+" if done else "w+"
    S = np.lib.format.open_memmap(
        a.out + "_samples.npy",
        mode=mode,
        dtype=np.float32,
        shape=(a.batches, a.size, 3),
    )
    G = np.lib.format.open_memmap(
        a.out + "_gt.npy", mode=mode, dtype=np.float32, shape=(a.batches, a.size)
    )
    for c in range(done, n_chunks):
        t = time.time()
        rng = np.random.default_rng(1000 + c)
        if a.stratified:
            # Every n_chunks-th cell: the chunks partition the cells, each spread over the domain.
            cells = np.arange(c, m.n_cells, n_chunks, dtype=np.int64)
        else:
            cells = rng.choice(m.n_cells, size=min(a.subset, m.n_cells), replace=False)
        tets, source = triangulate(m, cells, tag_source=a.stratified)
        pick = (
            stratified_pick(tets, source, cells.shape[0], per_chunk, a, vals, rng)
            if a.stratified
            else None
        )
        samples, gt = interior_points(tets, pick, pts, vals, per_chunk, a.inner, rng)
        if a.probe_interior:
            tp = time.time()
            linear, gt = gt, probe(m, name, samples, a.inner)
            ok = gt != -1
            mse = float(np.mean((gt[ok] - linear[ok]).astype(np.float64) ** 2))
            print(
                f"  probed in {time.time() - tp:.0f}s: {100 * (~ok).mean():.4f}% not found; "
                f"tet-linear values differ by {-10 * np.log10(max(mse, 1e-30)):.2f} dB PSNR",
                flush=True,
            )
            del linear, ok
        b0, cb = c * a.chunk_batches, a.chunk_batches
        S[b0 : b0 + cb, :size_int] = samples.reshape(cb, size_int, 3)
        G[b0 : b0 + cb, :size_int] = gt.reshape(cb, size_int)
        if band is not None:
            tb = time.time()
            empty = []
            for lo, n, inward in (
                (size_int, size_ext, False),
                (size_int + size_ext, size_skin, True),
            ):
                if n == 0:
                    continue
                bp = band_points(band, rng, cb * n, inward, a.band_width, a.band_pow)
                bg = probe(m, name, bp, a.inner)
                S[b0 : b0 + cb, lo : lo + n] = bp.reshape(cb, n, 3)
                G[b0 : b0 + cb, lo : lo + n] = bg.reshape(cb, n)
                empty.append(
                    f"{'skin' if inward else 'exterior'} {100 * (bg == -1).mean():.1f}% empty"
                )
            print(f"  band in {time.time() - tb:.0f}s: " + ", ".join(empty), flush=True)
        S.flush()
        G.flush()
        with open(progress, "w") as f:
            f.write(str(c + 1))
        print(
            f"chunk {c + 1}/{n_chunks}: {tets.shape[0]} tets in {time.time() - t:.0f}s, "
            f"gt[{gt.min():.4f},{gt.max():.4f}] mean {gt.mean():.4f}",
            flush=True,
        )
        del tets, source, pick, samples, gt
        if a.stop_after and c + 1 >= a.stop_after:
            print(f"stopping after {c + 1} chunks as requested", flush=True)
            break
    print(f"DONE {a.batches}x{a.size} -> {a.out}_{{samples,gt}}.npy", flush=True)


if __name__ == "__main__":
    main()
