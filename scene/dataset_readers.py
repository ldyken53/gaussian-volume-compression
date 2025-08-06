import numpy as np
import pyvista as pv
import torch 

from scene.gaussian_model import BasicPointCloud

def readData(path, fraction, normalized=False):
    mesh = pv.read(path)
    print("Mesh read")

    if not normalized:
        # Rescale the values to the range [0, 1]
        values = mesh.get_array("value").reshape(-1, 1)
        values_min = values.min()
        values_max = values.max()
        values = (values - values_min) / (values_max - values_min)
        mesh.get_array("value")[:] = values.ravel()

        # Scale mesh to the unit cube
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        global_min = min(xmin, ymin, zmin)
        global_max = max(xmax, ymax, zmax)
        mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
        mesh.scale(1.0/(global_max - global_min), inplace=True)
        # mesh.translate(np.array([0.01,0.01,0.01]), inplace=True)
        print("Mesh scaled")

    if fraction != -1:
        num_points = mesh.n_points
        pts = mesh.points
        print("Points gathered")
        mask = torch.rand(num_points) < fraction
        print("After rand")
        pts_sampled = pts[mask]
        vals = mesh.point_data["value"][mask].reshape(-1, 1)
        print("Mesh dropout")
        return mesh, BasicPointCloud(points=pts_sampled, values=vals), pts
    else:
        return mesh, BasicPointCloud(points=None, values=None), mesh.points
