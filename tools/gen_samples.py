"""Build a large training-sample set for a mesh whose full triangulation is impractical.

Same distribution as sample_mesh_points / precompute_tet.py -- cells uniform, then a
uniform point inside the cell -- but produced in chunks so the triangulation stays
bounded. Each chunk draws its OWN uniform random subset of cells, so the union over
chunks is still uniform over all cells while no single triangulation exceeds RAM.

Output matches what train.py --precomputed_samples expects:
  <out>_samples.npy  (batches, size, 3) float32
  <out>_gt.npy       (batches, size)    float32

Surface sets (--surface, or --ext_share / --interior_skin): like train.py's surface batches, each
batch is interior points, then --ext_share points up to --band_width outside the surface, then
--interior_skin points up to --band_width inside it, offset along the surface normals. Each face's
outward side is taken from its own volume cell (away from the cell's centroid), so meshes whose
surface is not consistently orientable (earthquake) still get their band on the right side. Only
true boundary faces are used: a non-conforming mesh's extract_surface also returns internal faces
(earthquake: 2/3 of them, 83% of the area), so a face is kept only if the point 1e-4 outside it is
outside the mesh. Band points are probed against the mesh: -1 outside, the interpolated value inside.

Resumable: writes via memmap and records finished chunks in <out>_progress.txt, so a
job that expires mid-run continues where it stopped.
"""

import os, sys, time, argparse, numpy as np, pyvista as pv

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--batches", type=int, default=400)
ap.add_argument("--size", type=int, default=2097152)
ap.add_argument(
    "--chunk_batches",
    type=int,
    default=25,
    help="batches produced per triangulated subset",
)
ap.add_argument(
    "--subset",
    type=int,
    default=40_000_000,
    help="cells triangulated per chunk; must be >> points_per_chunk/4",
)
ap.add_argument(
    "--stratified",
    action="store_true",
    help="partition cells disjointly across chunks and give EVERY cell at "
    "least one sample, instead of drawing tets with replacement",
)
ap.add_argument(
    "--importance",
    type=float,
    default=0.0,
    help="with --stratified: fraction LAM of each chunk's budget drawn with "
    "probability proportional to the tet's vertex-value range (shock "
    "tets), the rest uniform. 0 = plain uniform without replacement.",
)
ap.add_argument(
    "--importance_pow",
    type=float,
    default=1.0,
    help="raise the tet value-range to this power before normalising; >1 "
    "concentrates the importance budget on the sharpest tets",
)
ap.add_argument(
    "--stop_after",
    type=int,
    default=0,
    help="stop once this many chunks are written (0 = all)",
)
ap.add_argument(
    "--inner",
    type=int,
    default=4_000_000,
    help="points evaluated at once, to bound temporary allocations",
)
ap.add_argument(
    "--surface",
    action="store_true",
    help="train.py's surface batch mix: --ext_share 0.2 --interior_skin 0.1",
)
ap.add_argument("--ext_share", type=float, default=0.0, help="exterior band share")
ap.add_argument("--interior_skin", type=float, default=0.0, help="inner skin share")
ap.add_argument("--band_width", type=float, default=0.01)
ap.add_argument("--band_pow", type=float, default=1.0, help="band distance = u^p * width")
a = ap.parse_args()
if a.surface:
    a.ext_share = a.ext_share or 0.2
    a.interior_skin = a.interior_skin or 0.1
# Per batch, as train.py's load_samples: interior, then exterior band, then inner skin.
size_ext = int(a.size * a.ext_share)
size_skin = int(a.size * a.interior_skin)
size_int = a.size - size_ext - size_skin

assert a.batches % a.chunk_batches == 0, "batches must be a multiple of chunk_batches"
n_chunks = a.batches // a.chunk_batches
per_chunk = a.chunk_batches * size_int
prog_path = a.out + "_progress.txt"
done = 0
if os.path.exists(prog_path):
    done = int(open(prog_path).read().strip() or 0)
    print(f"resuming: {done}/{n_chunks} chunks already written", flush=True)

t = time.time()
m = pv.read(a.src)
print(f"read {time.time()-t:.1f}s  {m.n_points} pts  {m.n_cells} cells", flush=True)
m = m.cell_data_to_point_data()
vals = m.get_array(m.array_names[0]).reshape(-1).astype(np.float64)
vals = (vals - vals.min()) / (vals.max() - vals.min())
pts = np.asarray(m.points, dtype=np.float64)
gmin, gmax = pts.min(), pts.max()
pts = ((pts - gmin) / (gmax - gmin)).astype(np.float32)
print(
    f"normalised: vals[{vals.min():.4f},{vals.max():.4f}] "
    f"pts[{pts.min():.4f},{pts.max():.4f}]",
    flush=True,
)

band = None
if size_ext or size_skin:
    # The probe needs the mesh itself in the unit cube with values in [0, 1].
    name = m.array_names[0]
    m.points = pts
    m.point_data[name] = vals.astype(np.float32)
    t = time.time()
    surf = m.extract_surface(pass_cellid=True).triangulate()
    tri = surf.faces.reshape(-1, 4)[:, 1:]
    sp = np.asarray(surf.points, dtype=np.float64)
    nrm = np.cross(sp[tri[:, 1]] - sp[tri[:, 0]], sp[tri[:, 2]] - sp[tri[:, 0]])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm /= np.maximum(2.0 * area, 1e-30)[:, None]
    # Outward = away from the face's own volume cell, whatever the surface's winding says.
    owner = np.asarray(surf.cell_data["vtkOriginalCellIds"])
    uniq, inv = np.unique(owner, return_inverse=True)
    centers = np.asarray(m.extract_cells(uniq).cell_centers().points, dtype=np.float64)
    fc = sp[tri].mean(1)
    flip = ((fc - centers[inv]) * nrm).sum(1) < 0
    nrm[flip] *= -1
    # Internal faces of a non-conforming mesh have the mesh on their outer side too: drop them.
    outer = np.empty(tri.shape[0], dtype=bool)
    for lo in range(0, tri.shape[0], a.inner):
        pr = pv.PolyData((fc[lo : lo + a.inner] + 1e-4 * nrm[lo : lo + a.inner]).astype(np.float32))
        outer[lo : lo + a.inner] = ~np.asarray(pr.sample(m)["vtkValidPointMask"]).astype(bool)
    print(
        f"surface {tri.shape[0]} triangles in {time.time() - t:.0f}s, "
        f"{100 * flip.mean():.1f}% of face normals flipped outward, "
        f"{100 * outer.mean():.1f}% of faces ({100 * area[outer].sum() / area.sum():.1f}% of the area) "
        "on the true boundary",
        flush=True,
    )
    tri, nrm, area = tri[outer], nrm[outer], area[outer]
    band = (tri, sp, nrm, area / area.sum(), name)


def band_points(rng, n, inward):
    """n points up to band_width from the surface (area-uniform faces), outside or inside."""
    tri, sp, nrm, prob, _ = band
    f = rng.choice(tri.shape[0], size=n, p=prob)
    u = rng.random((n, 2))
    s0 = np.sqrt(u[:, 0])
    b = np.stack([1 - s0, s0 * (1 - u[:, 1]), s0 * u[:, 1]], 1)
    p = (b[:, :, None] * sp[tri[f]]).sum(1)
    d = rng.random((n, 1)) ** a.band_pow * a.band_width
    return (p + nrm[f] * (-d if inward else d)).astype(np.float32)


def probe(points):
    """Mesh value at each point, -1 outside the mesh."""
    out = np.empty(points.shape[0], dtype=np.float32)
    for lo in range(0, points.shape[0], a.inner):
        pr = pv.PolyData(points[lo : lo + a.inner]).sample(m)
        g = np.asarray(pr[band[4]], dtype=np.float32)
        g[~np.asarray(pr["vtkValidPointMask"]).astype(bool)] = -1.0
        out[lo : lo + a.inner] = g
    return out


mode = "r+" if done else "w+"
S = np.lib.format.open_memmap(
    a.out + "_samples.npy", mode=mode, dtype=np.float32, shape=(a.batches, a.size, 3)
)
G = np.lib.format.open_memmap(
    a.out + "_gt.npy", mode=mode, dtype=np.float32, shape=(a.batches, a.size)
)

for c in range(done, n_chunks):
    tc = time.time()
    rng = np.random.default_rng(1000 + c)  # per-chunk seed => resumable
    if a.stratified:
        # Disjoint stride partition: chunk c owns cells c, c+n_chunks, c+2*n_chunks...
        # so the union over chunks is every cell exactly once. Stride keeps each chunk
        # spread over the whole domain rather than taking a contiguous slab.
        sel = np.arange(c, m.n_cells, n_chunks, dtype=np.int64)
    else:
        sel = rng.choice(m.n_cells, size=min(a.subset, m.n_cells), replace=False)
    t = time.time()
    sub = m.extract_cells(sel)
    te = time.time() - t
    orig = np.asarray(sub.point_data["vtkOriginalPointIds"])
    if a.stratified:
        # Tag each source cell so its tets can be found after triangulation.
        sub.cell_data["cid"] = np.arange(sub.n_cells, dtype=np.int64)
    t = time.time()
    tm = sub.triangulate()
    tt = time.time() - t
    cells = orig[tm.cells.reshape(-1, 5)[:, 1:]]
    if a.stratified:
        cid = np.asarray(tm.cell_data["cid"])
        assert np.all(np.diff(cid) >= 0), "tets not grouped by source cell"
        ncell = sel.shape[0]
        starts = np.searchsorted(cid, np.arange(ncell, dtype=np.int64), side="left")
        counts = (
            np.searchsorted(cid, np.arange(ncell, dtype=np.int64), side="right")
            - starts
        )
        assert counts.min() >= 1, "a cell produced no tets"
        # One sample in every cell, then distribute the remaining budget over a random
        # subset of cells -- so coverage is 100% by construction, not by luck.
        # Draw tets WITHOUT replacement. Uniform over tets keeps the sample
        # distribution identical to sample_mesh_points and to the eval set (both are
        # tet-uniform); without replacement spends every point on a distinct tet
        # instead of re-hitting ones already covered.
        ntet = cells.shape[0]
        if a.importance > 0:
            # Shock-targeted draw: the held-out error on impact sits almost entirely on
            # one high-gradient surface where Gaussians are ~10x smaller than elsewhere
            # but sample density is the same, so each sees ~0.1 samples/sigma^3/batch.
            # Weight tets by their value range so those regions get the budget.
            rr = vals[cells]
            rr = rr.max(1) - rr.min(1)
            lam = a.importance
            rp = rr**a.importance_pow if a.importance_pow != 1.0 else rr
            pw = (1.0 - lam) / ntet + lam * rp / max(rp.sum(), 1e-30)
            del rp
            pw /= pw.sum()
            pick = rng.choice(ntet, size=per_chunk, replace=True, p=pw)
            hi = rr > 0.05
            print(
                f"  importance lam={lam}: tets with range>0.05 are {hi.mean()*100:.2f}% of tets "
                f"and get {hi[pick].mean()*100:.1f}% of samples "
                f"(enrichment {hi[pick].mean()/max(hi.mean(),1e-12):.1f}x); "
                f"distinct tets {len(np.unique(pick))/per_chunk*100:.1f}%",
                flush=True,
            )
            del rr, pw, hi
        else:
            pick = rng.choice(ntet, size=min(per_chunk, ntet), replace=False)
            if (
                pick.shape[0] < per_chunk
            ):  # budget exceeds tets: top up with a reshuffle
                pad = rng.choice(ntet, size=per_chunk - pick.shape[0], replace=False)
                pick = np.concatenate([pick, pad])
        pick = pick[rng.permutation(pick.shape[0])]
        cov_t = 100.0 * min(1.0, per_chunk / ntet)
        tpc = counts.mean()
        cov_c = 100.0 * (1.0 - (1.0 - min(1.0, per_chunk / ntet)) ** tpc)
        print(
            f"  stratified: {ncell} cells, {ntet} tets ({tpc:.2f}/cell), "
            f"{cov_t:.1f}% of tets, ~{cov_c:.1f}% of cells covered",
            flush=True,
        )
    del sub, tm
    out_s = np.empty((per_chunk, 3), dtype=np.float32)
    out_g = np.empty((per_chunk,), dtype=np.float32)
    for lo in range(0, per_chunk, a.inner):
        hi = min(lo + a.inner, per_chunk)
        k = hi - lo
        v = (
            cells[pick[lo:hi]]
            if a.stratified
            else cells[rng.integers(0, cells.shape[0], size=k)]
        )
        u = np.sort(rng.random((k, 3)), axis=1)
        b = np.empty((k, 4))
        b[:, 0] = u[:, 0]
        b[:, 1] = u[:, 1] - u[:, 0]
        b[:, 2] = u[:, 2] - u[:, 1]
        b[:, 3] = 1 - u[:, 2]
        out_s[lo:hi] = (b[:, :, None] * pts[v]).sum(1).astype(np.float32)
        out_g[lo:hi] = (b * vals[v]).sum(1).astype(np.float32)
    b0 = c * a.chunk_batches
    cb = a.chunk_batches
    S[b0 : b0 + cb, :size_int] = out_s.reshape(cb, size_int, 3)
    G[b0 : b0 + cb, :size_int] = out_g.reshape(cb, size_int)
    if band is not None:
        t = time.time()
        parts = [(size_int, size_ext, False), (size_int + size_ext, size_skin, True)]
        frac = []
        for lo, n, inward in parts:
            if n == 0:
                continue
            bp = band_points(rng, cb * n, inward)
            bg = probe(bp)
            S[b0 : b0 + cb, lo : lo + n] = bp.reshape(cb, n, 3)
            G[b0 : b0 + cb, lo : lo + n] = bg.reshape(cb, n)
            frac.append(f"{'skin' if inward else 'exterior'} {100 * (bg == -1).mean():.1f}% empty")
        print(f"  band in {time.time() - t:.0f}s: " + ", ".join(frac), flush=True)
    S.flush()
    G.flush()
    open(prog_path, "w").write(str(c + 1))
    print(
        f"chunk {c+1}/{n_chunks}: {cells.shape[0]} tets "
        f"(extract {te:.0f}s, triangulate {tt:.0f}s, total {time.time()-tc:.0f}s) "
        f"gt[{out_g.min():.4f},{out_g.max():.4f}] mean {out_g.mean():.4f}",
        flush=True,
    )
    del cells, out_s, out_g
    if a.stop_after and c + 1 >= a.stop_after:
        print(f"stopping after {c+1} chunks as requested", flush=True)
        break

print(f"DONE {a.batches}x{a.size} -> {a.out}_{{samples,gt}}.npy", flush=True)
