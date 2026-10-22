#include "rasterize_points.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
	m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
}
