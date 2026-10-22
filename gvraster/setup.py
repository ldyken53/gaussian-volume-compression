"""gvraster: the CUDA side of Gaussian volume models, as three extension modules.

  gvraster._bvh   renders Gaussians at arbitrary sample points through a BVH (meshes), plus the
                  fused activations and the surface models' margin FP term
  gvraster._tile  renders Gaussians onto a cell lattice with a tile rasterizer (regular grids)
  gvraster._ops   the training kernels both renderers share: one-launch Adam (+ EMA) and the
                  fused volume loss

Build in place with `python setup.py build_ext --inplace` and put this directory on PYTHONPATH.
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))
glm = os.path.join(here, "third_party", "glm")
cubql = os.path.join(here, "third_party", "cuBQL")


def sources(sub, names):
    return [os.path.join("csrc", sub, n) for n in names]


setup(
    name="gvraster",
    packages=["gvraster"],
    ext_modules=[
        CUDAExtension(
            name="gvraster._ops",
            sources=sources("ops", ["multi_adam.cu", "volume_loss.cu", "ops.cpp"]),
            depends=sources("ops", ["ops.h"]),
            extra_compile_args={"nvcc": ["-DNDEBUG"]},
        ),
        CUDAExtension(
            name="gvraster._bvh",
            sources=sources(
                "bvh",
                [
                    "cuda_rasterizer/forward.cu",
                    "cuda_rasterizer/backward.cu",
                    "rasterize_points.cu",
                    "fused_ops.cu",
                    "margin.cu",
                    "ext.cu",
                ],
            ),
            depends=sources(
                "bvh", ["cuda_rasterizer/config.h", "cuda_rasterizer/auxiliary.h"]
            ),
            extra_compile_args={"nvcc": ["-I" + glm, "-I" + cubql, "-DNDEBUG"]},
        ),
        CUDAExtension(
            name="gvraster._tile",
            sources=sources(
                "tile",
                [
                    "cuda_rasterizer/rasterizer_impl.cu",
                    "cuda_rasterizer/forward.cu",
                    "cuda_rasterizer/backward.cu",
                    "rasterize_points.cu",
                    "ext.cpp",
                ],
            ),
            depends=sources(
                "tile", ["cuda_rasterizer/config.h", "cuda_rasterizer/auxiliary.h"]
            ),
            extra_compile_args={"nvcc": ["-I" + glm]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
