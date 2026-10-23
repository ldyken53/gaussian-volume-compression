"""gvraster: the CUDA kernels of VEGGIES models, as three extension modules.

  gvraster._bvh   sampling at arbitrary points through a BVH (meshes), the activations, and the
                  margin FP term
  gvraster._tile  sampling on a jittered lattice with a tile rasterizer (grids)
  gvraster._ops   Adam with the parameter averages, and the training loss

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
            sources=sources("ops", ["multi_adam.cu", "volume_loss.cu", "bindings.cpp"]),
            depends=sources("ops", ["ops.h"]),
            extra_compile_args={"nvcc": ["-DNDEBUG"]},
        ),
        CUDAExtension(
            name="gvraster._bvh",
            sources=sources(
                "bvh",
                [
                    "forward.cu",
                    "backward.cu",
                    "sample.cu",
                    "activations.cu",
                    "margin.cu",
                    "points.cu",
                    "bindings.cpp",
                ],
            ),
            depends=sources(
                "bvh",
                [
                    "bvh.h",
                    "points.h",
                    "forward.h",
                    "backward.h",
                    "gaussian.h",
                    "scratch.h",
                ],
            ),
            extra_compile_args={"nvcc": ["-I" + glm, "-I" + cubql, "-DNDEBUG"]},
        ),
        CUDAExtension(
            name="gvraster._tile",
            sources=sources(
                "tile",
                ["tiles.cu", "forward.cu", "backward.cu", "sample.cu", "bindings.cpp"],
            ),
            depends=sources(
                "tile",
                [
                    "tile.h",
                    "tiles.h",
                    "forward.h",
                    "backward.h",
                    "config.h",
                    "activation.h",
                ],
            ),
            extra_compile_args={"nvcc": ["-I" + glm]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
