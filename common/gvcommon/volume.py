"""Seed points: the node subset initial Gaussians start from, and the structured-points (.vtk)
reader both pipelines use."""

from typing import NamedTuple

import numpy as np
import pyvista as pv
import torch


class BasicPointCloud(NamedTuple):
    points: np.array
    values: np.array


def cap_fraction(fraction, n_pts, max_points):
    """The seeded share of n_pts nodes, lowered so it seeds at most max_points."""
    if max_points is not None and fraction * n_pts > max_points:
        print(
            f"init capped: fraction {fraction:g} -> {max_points / n_pts:g} "
            f"({int(max_points)} of {n_pts} nodes)"
        )
        return max_points / n_pts
    return fraction


def read_structured(path, fraction, normalized=False, max_points=None, scalar=None):
    """A structured-points volume and a random `fraction` of its nodes as seed points.

    Unless normalized, values are min-max scaled to [0, 1] and the volume to the unit cube.
    scalar names the point array (default: the first one).
    """
    mesh = pv.read(path)
    name = scalar if scalar is not None else mesh.point_data.keys()[0]
    if not normalized:
        values = mesh.get_array(name).reshape(-1, 1)
        values = (values - values.min()) / (values.max() - values.min())
        mesh.get_array(name)[:] = values.ravel()
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
    fraction = cap_fraction(fraction, n_pts, max_points)
    mask = torch.rand(n_pts) < fraction
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
    k = idx // (nx * ny)
    r = idx % (nx * ny)
    j = r // nx
    i = r % nx
    pts = torch.stack(
        [i.float() * sx + ox, j.float() * sy + oy, k.float() * sz + oz], dim=1
    )
    vals = mesh.point_data[name][mask].reshape(-1, 1)
    return mesh, BasicPointCloud(points=pts, values=vals)
