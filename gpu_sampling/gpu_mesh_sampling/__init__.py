from ._gpu_mesh_sampling import sample_mesh
import numpy as np

def test_sample(
    pts: np.ndarray,
    conn: np.ndarray,
    values: np.ndarray,
    samples: np.ndarray
):
    print("TEST WORKED")
    return sample_mesh(pts, conn, values, samples)