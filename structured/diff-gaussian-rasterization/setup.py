import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = os.path.dirname(os.path.abspath(__file__))

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
            depends=["cuda_rasterizer/config.h", "cuda_rasterizer/auxiliary.h"],
            extra_compile_args={
                "nvcc": ["-I" + os.path.join(here, "third_party/glm/")]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
