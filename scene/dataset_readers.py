from typing import NamedTuple

import numpy as np
import pyvista as pv
import torch


class BasicPointCloud(NamedTuple):
    points: np.array
    values: np.array


def _cap_fraction(fraction, n_pts, max_points):
    if max_points is not None and fraction * n_pts > max_points:
        print(
            f"init capped: fraction {fraction:g} -> {max_points / n_pts:g} "
            f"({int(max_points)} of {n_pts} nodes)"
        )
        return max_points / n_pts
    return fraction


def readDatau(path, fraction, max_points=None):
    """Mesh with values in [0, 1] and largest extent the unit cube, plus a random node subset."""
    mesh = pv.read(path)
    mesh = mesh.cell_data_to_point_data()
    values = mesh.get_array(mesh.array_names[0]).reshape(-1, 1)
    values = (values - values.min()) / (values.max() - values.min())
    mesh.get_array(mesh.array_names[0])[:] = values.ravel()

    global_min = mesh.points.min()
    global_max = mesh.points.max()
    mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
    mesh.scale(1 / (global_max - global_min), inplace=True)

    num_points = mesh.points.shape[0]
    print(num_points)
    fraction = _cap_fraction(fraction, num_points, max_points)
    indices = np.random.choice(
        num_points, size=int(num_points * fraction), replace=False
    )
    return mesh, BasicPointCloud(points=mesh.points[indices], values=values[indices])


def readData(path, fraction, max_points=None):
    """readDatau for a structured .vtk volume with a "value" array."""
    mesh = pv.read(path)
    values = mesh.get_array("value").reshape(-1, 1)
    values = (values - values.min()) / (values.max() - values.min())
    mesh.get_array("value")[:] = values.ravel()

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    global_min = min(xmin, ymin, zmin)
    global_max = max(xmax, ymax, zmax)
    mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
    mesh.scale(1.0 / (global_max - global_min), inplace=True)

    nx, ny, nz = mesh.dimensions
    ox, oy, oz = mesh.origin
    sx, sy, sz = mesh.spacing
    n_pts = mesh.n_points
    fraction = _cap_fraction(fraction, n_pts, max_points)
    mask = torch.rand(n_pts) < fraction
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    k = idx // (nx * ny)
    r = idx % (nx * ny)
    j = r // nx
    i = r % nx
    pts = torch.stack(
        [i.float() * sx + ox, j.float() * sy + oy, k.float() * sz + oz], dim=1
    )
    vals = mesh.point_data["value"][mask].reshape(-1, 1)
    return mesh, BasicPointCloud(points=pts, values=vals)


def read_mesh(path, fraction, max_points=None):
    """readDatau or readData by extension."""
    if path.lower().endswith(".vtu"):
        return readDatau(path, fraction, max_points)
    if path.lower().endswith(".vtk"):
        return readData(path, fraction, max_points)
    raise ValueError(f"not a .vtu or .vtk mesh: {path}")
