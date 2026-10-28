"""What gen_samples.py and precompute_tet.py share: a mesh scaled the way train.py scales it, and
uniform points in its tets."""

import time

import numpy as np
import pyvista as pv


def read_mesh(path, array=None, points_dtype=np.float64):
    """(mesh with point data, array name, values in [0, 1], points in the unit cube)."""
    t = time.time()
    m = pv.read(path)
    print(
        f"read {time.time() - t:.1f}s  {m.n_points} pts  {m.n_cells} cells", flush=True
    )
    m = m.cell_data_to_point_data()
    name = array or m.array_names[0]
    if name not in m.array_names:
        raise SystemExit(f"no array {name!r}; the mesh has {m.array_names}")
    vals = m.get_array(name).reshape(-1).astype(np.float64)
    vals = (vals - vals.min()) / (vals.max() - vals.min())
    pts = np.asarray(m.points, dtype=np.float64)
    lo, hi = pts.min(), pts.max()
    pts = ((pts - lo) / (hi - lo)).astype(points_dtype)
    return m, name, vals, pts


def triangulate(m, cells=None, tag_source=False):
    """The tets of the given cells (all if None) as (T, 4) point ids of m, and with tag_source
    each tet's index into cells."""
    t = time.time()
    if cells is None:
        tm = m.triangulate()
        tets = tm.cells.reshape(-1, 5)[:, 1:]
    else:
        sub = m.extract_cells(cells)
        orig = np.asarray(sub.point_data["vtkOriginalPointIds"])
        if tag_source:
            sub.cell_data["source"] = np.arange(sub.n_cells, dtype=np.int64)
        tm = sub.triangulate()
        tets = orig[tm.cells.reshape(-1, 5)[:, 1:]]
    print(
        f"  triangulated {m.n_cells if cells is None else len(cells)} cells -> {tets.shape[0]} tets "
        f"in {time.time() - t:.0f}s",
        flush=True,
    )
    return tets, (np.asarray(tm.cell_data["source"]) if tag_source else None)


def points_in_tets(tets, pts, vals, rng):
    """One uniform point in each of the (k, 4) tets, and its linearly interpolated value. That is
    the mesh's own value only where the tet is a cell of it; elsewhere use probe()."""
    k = tets.shape[0]
    u = np.sort(rng.random((k, 3)), axis=1)
    b = np.empty((k, 4))
    b[:, 0] = u[:, 0]
    b[:, 1] = u[:, 1] - u[:, 0]
    b[:, 2] = u[:, 2] - u[:, 1]
    b[:, 3] = 1 - u[:, 2]
    return (b[:, :, None] * pts[tets]).sum(1), (b * vals[tets]).sum(1)


def has_non_tets(m):
    """True if some cell is not a tetrahedron. Inside one, linear interpolation within its split
    tets differs from the cell's own (trilinear on a hexahedron), which is what eval.py scores.
    """
    return bool(np.any(np.asarray(m.celltypes) != pv.CellType.TETRA))


def scale_for_probe(m, name, vals, pts):
    """Give m read_mesh's scaled points and values, in place, so probe() returns train.py's
    units."""
    m.points = pts
    m.point_data[name] = vals.astype(np.float32)


def probe(m, name, points, inner=4_000_000):
    """The mesh's own value at each point (vtkProbeFilter), -1 outside it."""
    out = np.empty(points.shape[0], dtype=np.float32)
    for lo in range(0, points.shape[0], inner):
        pr = pv.PolyData(points[lo : lo + inner]).sample(m)
        g = np.asarray(pr[name], dtype=np.float32)
        g[~np.asarray(pr["vtkValidPointMask"]).astype(bool)] = -1.0
        out[lo : lo + inner] = g
    return out
