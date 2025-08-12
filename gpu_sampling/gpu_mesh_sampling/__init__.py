from ._gpu_mesh_sampling import sample_mesh
import numpy as np

def gpu_sample(
    dims: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    values: np.ndarray,
    samples: np.ndarray
):
    return sample_mesh(dims, origin, spacing, values, samples)