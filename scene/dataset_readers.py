import numpy as np
import pyvista as pv

from scene.gaussian_model import BasicPointCloud
from gpu_mesh_sampling import gpu_sample

def readData(path, fraction):
    mesh = pv.read(path)

    # Rescale the values to the range [0, 1]
    values = mesh.get_array("value").reshape(-1, 1)
    values_min = values.min()
    values_max = values.max()
    values = (values - values_min) / (values_max - values_min)
    mesh.get_array("value")[:] = values.ravel()

    # Scale mesh to the unit cube
    global_min = mesh.points.min()
    global_max = mesh.points.max()
    mesh.translate(np.array([-global_min, -global_min, -global_min]), inplace=True)
    mesh.scale(1.0/(global_max - global_min), inplace=True)
    # mesh.translate(np.array([0.01,0.01,0.01]), inplace=True)

    num_points = mesh.points.shape[0]
    indices = np.random.choice(num_points, size=int(num_points * fraction), replace=False)
    points_sampled = mesh.points[indices]
    values_sampled = values[indices]

    # cell_count = 25
    # xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    # x = np.linspace(xmin, xmax, cell_count)
    # y = np.linspace(ymin, ymax, cell_count)
    # z = np.linspace(zmin, zmax, cell_count)
    # x, y, z = np.meshgrid(x, y, z, indexing='ij')
    # samples = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    # samples_3d = samples.reshape(cell_count, cell_count, cell_count, 3)
    # rot = np.rot90(samples_3d, k=1, axes=(2,0))
    # samples_tf = np.flip(rot, axis=2)
    # save_cell = samples_tf.reshape(-1, 3)
    # save_gt = gpu_sample(
    #     mesh.points, 
    #     mesh.cell_connectivity.astype(np.int64),
    #     mesh.point_data['value'],
    #     save_cell
    # )

    return mesh, BasicPointCloud(points=points_sampled, values=values_sampled)

