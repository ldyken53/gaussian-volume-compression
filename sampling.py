"""Training and evaluation sample points for an unstructured mesh in the unit cube."""

import torch


def sample_mesh_points(mesh, num_batches, batch_size, device="cuda"):
    """Uniform random cells, then a uniform point inside each (dense where cells are small)."""
    points = torch.tensor(mesh.points, dtype=torch.float32, device=device)
    tet_mesh = mesh.triangulate()
    cells = torch.tensor(
        tet_mesh.cells.reshape(-1, 5)[:, 1:], dtype=torch.long, device=device
    )

    total = num_batches * batch_size
    cell_idx = torch.randint(0, cells.shape[0], (total,), device=device)
    verts = points[cells[cell_idx]]

    # Sorted uniforms; their gaps are uniform barycentric coordinates in a tetrahedron.
    u = torch.rand(total, 3, device=device).sort(dim=1).values
    bary = torch.zeros(total, 4, device=device)
    bary[:, 0] = u[:, 0]
    bary[:, 1] = u[:, 1] - u[:, 0]
    bary[:, 2] = u[:, 2] - u[:, 1]
    bary[:, 3] = 1.0 - u[:, 2]

    samples = (bary.unsqueeze(2) * verts).sum(dim=1)
    return samples.reshape(num_batches, batch_size, 3)


def sample_exterior_points(
    mesh, num_batches, batch_size, offset=0.01, device="cuda", inward=False
):
    """Points up to `offset` outside the surface along its outward normals (inside if inward)."""
    surf = (
        mesh.extract_surface()
        .triangulate()
        .compute_normals(cell_normals=True, point_normals=False)
    )
    face_normals = torch.tensor(
        surf.cell_data["Normals"], dtype=torch.float32, device=device
    )
    points = torch.tensor(surf.points, dtype=torch.float32, device=device)
    faces = torch.tensor(
        surf.faces.reshape(-1, 4)[:, 1:], dtype=torch.long, device=device
    )

    total = num_batches * batch_size
    face_idx = torch.randint(0, faces.shape[0], (total,), device=device)
    v = points[faces[face_idx]]

    u = torch.rand(total, 2, device=device)
    sqrt_u0 = u[:, 0].sqrt()
    bary = torch.stack([1 - sqrt_u0, sqrt_u0 * (1 - u[:, 1]), sqrt_u0 * u[:, 1]], dim=1)
    surf_pts = (bary.unsqueeze(2) * v).sum(dim=1)

    normals = face_normals[face_idx]
    dist = torch.rand(total, 1, device=device) * offset
    samples = surf_pts + normals * (-dist if inward else dist)
    return samples.reshape(num_batches, batch_size, 3)
