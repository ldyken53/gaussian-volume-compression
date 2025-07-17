from ._gpu_mesh_sampling import sample_mesh
import numpy as np

def gpu_sample(
    pts: np.ndarray,
    conn: np.ndarray,
    values: np.ndarray,
    samples: np.ndarray
):
    return sample_mesh(pts, conn, values, samples)