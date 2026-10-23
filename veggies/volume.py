"""Reading a volume: a regular grid (VTK image data) or a mesh (any other VTK dataset with 3D
cells), scaled into the unit cube with values in [0, 1], and the nodes its Gaussians start from.
"""

import os
import re
from typing import NamedTuple

import numpy as np
import pyvista as pv
import torch

GRID_SUFFIXES = (".vti",)
BYTES_PER_GAUSSIAN = 48


class Seeds(NamedTuple):
    points: np.ndarray
    values: np.ndarray


class Normalization(NamedTuple):
    """unit-cube point = (point - offset) / scale; [0, 1] value = (value - value_min) / range."""

    offset: float
    scale: float
    value_min: float
    value_max: float


IDENTITY = Normalization(0.0, 1.0, 0.0, 1.0)


class Volume(NamedTuple):
    dataset: object
    is_grid: bool
    normalization: Normalization
    nbytes: int  # points + cells + scalar as stored (meshes); 4 per node (grids)


def _legacy_header(path, lines=12):
    with open(path, "rb") as f:
        return [f.readline().decode("ascii", "replace").split() for _ in range(lines)]


def is_grid_file(path):
    """Whether path holds a regular grid, judged from its extension or legacy VTK header."""
    p = path.lower()
    if p.endswith(GRID_SUFFIXES):
        return True
    if p.endswith(".vtk") and os.path.exists(path):
        return any(
            t[:2] == ["DATASET", "STRUCTURED_POINTS"] for t in _legacy_header(path) if t
        )
    return False


def grid_node_count(path):
    p = path.lower()
    if p.endswith(".vtk"):
        for t in _legacy_header(path, 32):
            if t and t[0].upper() == "DIMENSIONS":
                return int(t[1]) * int(t[2]) * int(t[3])
    elif p.endswith(".vti"):
        with open(path, "rb") as f:
            head = f.read(4096).decode("ascii", "replace")
        m = re.search(
            r'WholeExtent="\s*(-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (-?\d+)', head
        )
        if m:
            e = [int(x) for x in m.groups()]
            return (e[1] - e[0] + 1) * (e[3] - e[2] + 1) * (e[5] - e[4] + 1)
    raise SystemExit(
        f"{path}: cannot read the grid size from its header; pass --cap_max"
    )


def grid_gaussian_count(path, ratio):
    """Gaussians at a compression ratio of a grid stored as float32: nodes / (12 ratio)."""
    return grid_node_count(path) // int(round(BYTES_PER_GAUSSIAN / 4 * ratio))


def mesh_gaussian_count(volume, ratio):
    """Gaussians at a compression ratio of a mesh: its bytes / (48 ratio)."""
    return int(round(volume.nbytes / (BYTES_PER_GAUSSIAN * ratio)))


def mesh_nbytes(mesh):
    scalar = mesh[mesh.array_names[0]]
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    arrays = (mesh.points, mesh.cell_connectivity, mesh.offset, mesh.celltypes, scalar)
    return sum(np.asarray(a).nbytes for a in arrays)


def select_array(dataset, name):
    """Make name the dataset's only data array, so every later step uses it."""
    if name is None or name == dataset.array_names[0]:
        return dataset
    if name not in dataset.array_names:
        raise SystemExit(f"no array {name!r}; the volume has {dataset.array_names}")
    is_point = name in dataset.point_data
    data = np.asarray(
        dataset.point_data[name] if is_point else dataset.cell_data[name]
    ).copy()
    dataset.clear_data()
    (dataset.point_data if is_point else dataset.cell_data)[name] = data
    return dataset


def _as_float(dataset, name):
    data = dataset.point_data if name in dataset.point_data else dataset.cell_data
    if not np.issubdtype(data[name].dtype, np.floating):
        data[name] = np.asarray(data[name]).astype(np.float32)


def normalize_grid(grid, name):
    _as_float(grid, name)
    values = grid.get_array(name).reshape(-1, 1)
    vmin, vmax = values.min(), values.max()
    grid.get_array(name)[:] = ((values - vmin) / (vmax - vmin)).ravel()
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    lo, hi = min(xmin, ymin, zmin), max(xmax, ymax, zmax)
    grid.translate(np.array([-lo, -lo, -lo]), inplace=True)
    grid.scale(1.0 / (hi - lo), inplace=True)
    print("Grid scaled into the unit cube")
    return Normalization(float(lo), float(hi - lo), float(vmin), float(vmax))


def normalize_mesh(mesh):
    """The mesh with cell data moved to points, values in [0, 1] and its largest extent [0, 1]."""
    _as_float(mesh, mesh.array_names[0])
    mesh = mesh.cell_data_to_point_data()
    name = mesh.array_names[0]
    values = mesh.get_array(name).reshape(-1, 1)
    vmin, vmax = values.min(), values.max()
    mesh.get_array(name)[:] = ((values - vmin) / (vmax - vmin)).ravel()
    lo, hi = mesh.points.min(), mesh.points.max()
    mesh.translate(np.array([-lo, -lo, -lo]), inplace=True)
    mesh.scale(1 / (hi - lo), inplace=True)
    return mesh, Normalization(float(lo), float(hi - lo), float(vmin), float(vmax))


def load(path, array=None, normalized=False):
    """The volume at path, normalized unless it already is (grids only)."""
    if not os.path.exists(path):
        raise SystemExit(f"no such volume: {path}")
    dataset = select_array(pv.read(path), array)
    if isinstance(dataset, pv.ImageData):
        nbytes = 4 * dataset.n_points
        if normalized:
            return Volume(dataset, True, IDENTITY, nbytes)
        return Volume(
            dataset, True, normalize_grid(dataset, dataset.point_data.keys()[0]), nbytes
        )
    if dataset.n_cells == 0:
        raise SystemExit(f"{path}: neither a grid nor a mesh with cells")
    nbytes = mesh_nbytes(dataset)
    mesh, norm = normalize_mesh(dataset)
    print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")
    return Volume(mesh, False, norm, nbytes)


def capped_fraction(fraction, n, max_points):
    if max_points is not None and fraction * n > max_points:
        print(
            f"init capped: fraction {fraction:g} -> {max_points / n:g} ({int(max_points)} of {n} nodes)"
        )
        return max_points / n
    return fraction


def seed_points(volume, fraction, max_points=None):
    """A random fraction of the nodes, at most max_points of them."""
    ds = volume.dataset
    if volume.is_grid:
        name = ds.point_data.keys()[0]
        nx, ny, nz = ds.dimensions
        ox, oy, oz = ds.origin
        sx, sy, sz = ds.spacing
        n = ds.n_points
        mask = torch.rand(n) < capped_fraction(fraction, n, max_points)
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        k = idx // (nx * ny)
        r = idx % (nx * ny)
        j, i = r // nx, r % nx
        points = torch.stack(
            [i.float() * sx + ox, j.float() * sy + oy, k.float() * sz + oz], dim=1
        )
        return Seeds(points, ds.point_data[name][mask].reshape(-1, 1))
    n = ds.points.shape[0]
    values = ds.get_array(ds.array_names[0]).reshape(-1, 1)
    idx = np.random.choice(
        n, size=int(n * capped_fraction(fraction, n, max_points)), replace=False
    )
    return Seeds(ds.points[idx], values[idx])
