"""CUDA kernels for Gaussian volume models.

gvraster.bvh   Gaussians rendered at arbitrary sample points through a BVH (unstructured meshes)
gvraster.tile  Gaussians rendered onto a cell lattice by a tile rasterizer (regular grids)
gvraster.ops   the training kernels both use: one-launch Adam (+ EMA) and the fused volume loss

Each submodule loads only its own extension, so a build missing one renderer still runs the
other.
"""
