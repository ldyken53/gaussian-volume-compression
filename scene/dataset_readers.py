import numpy as np
import pyvista as pv

from scene.gaussian_model import BasicPointCloud

def readData(path, fraction):
    mesh = pv.read(path)

    print("before values scaling")
    # Rescale the values to the range [0, 1]
    values = mesh.get_array("value").reshape(-1, 1)
    values_min = values.min()
    values_max = values.max()
    values = (values - values_min) / (values_max - values_min)
    mesh.get_array("value")[:] = values.ravel()
    print("after values scaling")

    print("before points scaling")
    # Scale mesh to the unit cube
    global_min = mesh.points.min()
    global_max = mesh.points.max()
    mesh.points[:] = (mesh.points - global_min) / (global_max - global_min)
    print("after points scaling")

    num_points = mesh.points.shape[0]
    indices = np.random.choice(num_points, size=int(num_points * fraction), replace=False)
    points_sampled = mesh.points[indices]
    values_sampled = values[indices]

    return mesh, BasicPointCloud(points=points_sampled, values=values_sampled)
