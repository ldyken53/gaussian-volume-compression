"""Precompute a tet-interior evaluation set for a mesh whose full triangulation is
impractical (impactnorm: 247M mixed cells -> ~740M tets, ~24 GB of connectivity).

sample_mesh_points picks cells UNIFORMLY and then a uniform point inside each, so
drawing a uniform random SUBSET of cells first and sampling within that subset is
statistically the same thing -- and it lets us triangulate a few million cells
instead of all of them.

Ground truth needs no probe: a point at barycentric coords b inside a tet is worth
b . (values at that tet's four vertices), which is exactly the linear interpolation
vtkProbeFilter performs. Values and coordinates are normalised the same way
read_unstructured (scene/dataset_readers.py) does, so the set matches what the models
were trained against.
"""

import sys, time, argparse, numpy as np, pyvista as pv

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--n", type=int, default=1_000_000)
ap.add_argument(
    "--subset",
    type=int,
    default=4_000_000,
    help="cells to triangulate; 0 = all. Must be >> n/4 to keep tets varied.",
)
a = ap.parse_args()

t = time.time()
m = pv.read(a.src)
print(f"read {time.time()-t:.1f}s  {m.n_points} pts  {m.n_cells} cells", flush=True)
m = m.cell_data_to_point_data()

vals = m.get_array(m.array_names[0]).reshape(-1).astype(np.float64)
vals = (vals - vals.min()) / (vals.max() - vals.min())
pts = np.asarray(m.points, dtype=np.float64)
gmin, gmax = pts.min(), pts.max()
pts = (pts - gmin) / (gmax - gmin)

rng = np.random.default_rng(0)
if a.subset and a.subset < m.n_cells:
    sel = rng.choice(m.n_cells, size=a.subset, replace=False)
    t = time.time()
    sub = m.extract_cells(sel)
    print(f"extract_cells {time.time()-t:.1f}s -> {sub.n_cells} cells", flush=True)
    # extract_cells renumbers points; map back via the original ids it carries
    orig = np.asarray(sub.point_data["vtkOriginalPointIds"])
    t = time.time()
    tm = sub.triangulate()
    print(f"triangulate {time.time()-t:.1f}s -> {tm.n_cells} tets", flush=True)
    cells = orig[tm.cells.reshape(-1, 5)[:, 1:]]  # back to original point ids
else:
    t = time.time()
    tm = m.triangulate()
    print(f"triangulate {time.time()-t:.1f}s -> {tm.n_cells} tets", flush=True)
    cells = tm.cells.reshape(-1, 5)[:, 1:]

ci = rng.integers(0, cells.shape[0], size=a.n)
v = cells[ci]
u = np.sort(rng.random((a.n, 3)), axis=1)
b = np.empty((a.n, 4))
b[:, 0] = u[:, 0]
b[:, 1] = u[:, 1] - u[:, 0]
b[:, 2] = u[:, 2] - u[:, 1]
b[:, 3] = 1 - u[:, 2]
samples = (b[:, :, None] * pts[v]).sum(1)
gt = (b * vals[v]).sum(1)
np.save(a.out + "_samples.npy", samples.astype(np.float32))
np.save(a.out + "_gt.npy", gt.astype(np.float32))
print(
    f"wrote {a.out}_{{samples,gt}}.npy  n={a.n}  "
    f"gt[{gt.min():.5f},{gt.max():.5f}] mean {gt.mean():.5f}  "
    f"xyz bounds {np.round(samples.min(0),4)}..{np.round(samples.max(0),4)}",
    flush=True,
)
