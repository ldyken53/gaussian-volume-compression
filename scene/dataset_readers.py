import numpy as np
import pyvista as pv

from gvcommon.volume import BasicPointCloud, cap_fraction, read_structured


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
    fraction = cap_fraction(fraction, num_points, max_points)
    indices = np.random.choice(
        num_points, size=int(num_points * fraction), replace=False
    )
    return mesh, BasicPointCloud(points=mesh.points[indices], values=values[indices])


def read_mesh(path, fraction, max_points=None):
    """readDatau, or the shared structured-points reader for a .vtk with a "value" array."""
    if path.lower().endswith(".vtu"):
        return readDatau(path, fraction, max_points)
    if path.lower().endswith(".vtk"):
        return read_structured(path, fraction, max_points=max_points, scalar="value")
    raise ValueError(f"not a .vtu or .vtk mesh: {path}")
