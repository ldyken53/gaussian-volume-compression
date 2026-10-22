import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))
# Kernel sources both pipelines' extensions compile (see multi_adam.cu).
common = os.path.normpath(os.path.join(here, "..", "common", "cuda"))

setup(
    name="bvh-diff-gaussian-rasterization",
    packages=["bvh_diff_gaussian_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="bvh_diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "multi_adam.cu",
                "fused_ops.cu",
                "margin.cu",
                "ext.cu",
            ],
            depends=[
                "cuda_rasterizer/config.h",
                "cuda_rasterizer/auxiliary.h",
                os.path.join(common, "multi_adam.cuh"),
            ],
            extra_compile_args={
                "nvcc": [
                    "-I" + os.path.join(here, "third_party/glm/"),
                    "-I" + os.path.join(here, "third_party/cuBQL/"),
                    "-I" + common,
                    "-DNDEBUG",
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
