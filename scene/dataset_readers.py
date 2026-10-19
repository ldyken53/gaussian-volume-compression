import numpy as np
import pyvista as pv
import torch

from utils.graphics_utils import BasicPointCloud


def readData(path, fraction, normalized, max_points=None):
    """Load the volume and sample a random fraction of its nodes as initial points.

    Unless normalized, values are min-max scaled to [0, 1] and the mesh to the unit cube.
    """
    mesh = pv.read(path)
    scalar_name = mesh.point_data.keys()[0]

    if not normalized:
        values = mesh.get_array(scalar_name).reshape(-1, 1)
        values = (values - values.min()) / (values.max() - values.min())
        mesh.get_array(scalar_name)[:] = values.ravel()
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        global_min = min(xmin, ymin, zmin)
        global_max = max(xmax, ymax, zmax)
        mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
        mesh.scale(1.0 / (global_max - global_min), inplace=True)
        print("Mesh scaled")

    nx, ny, nz = mesh.dimensions
    ox, oy, oz = mesh.origin
    sx, sy, sz = mesh.spacing
    n_pts = mesh.n_points
    if max_points is not None and fraction * n_pts > max_points:
        print(
            f"init capped: fraction {fraction:g} -> {max_points / n_pts:g} "
            f"({int(max_points)} of {n_pts} nodes)"
        )
        fraction = max_points / n_pts

    mask = torch.rand(n_pts) < fraction
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    k = idx // (nx * ny)
    r = idx % (nx * ny)
    j = r // nx
    i = r % nx
    pts = torch.stack(
        [i.float() * sx + ox, j.float() * sy + oy, k.float() * sz + oz], dim=1
    )
    vals = mesh.point_data[scalar_name][mask].reshape(-1, 1)
    return mesh, BasicPointCloud(points=pts, values=vals)
