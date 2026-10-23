"""The CUDA kernels of VEGGIES models.

gvraster.bvh   sampling at arbitrary points through a BVH (meshes), and the activations
gvraster.tile  sampling on a jittered lattice with a tile rasterizer (grids)
gvraster.ops   Adam with the parameter averages, and the training loss

Each submodule loads only its own extension.
"""
