import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))
# One glm for the repository: the unstructured extension's submodule.
glm = os.path.normpath(
    os.path.join(here, "..", "..", "diff-gaussian-rasterization", "third_party", "glm")
)
# Kernel sources both pipelines' extensions compile (see multi_adam.cu, fused_ops.cu).
common = os.path.normpath(os.path.join(here, "..", "..", "common", "cuda"))

setup(
    name="diff_gaussian_rasterization",
    packages=["diff_gaussian_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "multi_adam.cu",
                "fused_ops.cu",
                "ext.cpp",
            ],
            depends=[
                "cuda_rasterizer/config.h",
                "cuda_rasterizer/auxiliary.h",
                os.path.join(common, "multi_adam.cuh"),
                os.path.join(common, "volume_loss.cuh"),
                os.path.join(common, "gv_ops.h"),
            ],
            include_dirs=[common],
            extra_compile_args={"nvcc": ["-I" + glm]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
