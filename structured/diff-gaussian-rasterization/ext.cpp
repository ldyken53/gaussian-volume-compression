#include "rasterize_points.h"
#include "gv_ops.h"
#include <torch/extension.h>
#include <vector>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
	m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
	m.def("multi_adam_", &multi_adam_);
	m.def("fused_loss_forward", &fused_loss_forward);
	m.def("fused_loss_backward", &fused_loss_backward);
}
