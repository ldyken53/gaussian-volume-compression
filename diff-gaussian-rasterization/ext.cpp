#include "rasterize_points.h"
#include <torch/extension.h>
#include <vector>

void multi_adam_(std::vector<torch::Tensor> params, std::vector<torch::Tensor> grads,
	std::vector<torch::Tensor> exp_avgs, std::vector<torch::Tensor> exp_avg_sqs,
	std::vector<double> lrs, int64_t step, double beta1, double beta2, double eps,
	std::vector<torch::Tensor> emas, double ema_w);
std::vector<torch::Tensor> fused_loss_forward(
	torch::Tensor cells, torch::Tensor gt, torch::Tensor weights, double fn_reg, double tau);
std::vector<torch::Tensor> fused_loss_backward(torch::Tensor cells, torch::Tensor gt,
	torch::Tensor weights, torch::Tensor acc, double fn_reg, double tau, torch::Tensor gout);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
	m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
	m.def("multi_adam_", &multi_adam_);
	m.def("fused_loss_forward", &fused_loss_forward);
	m.def("fused_loss_backward", &fused_loss_backward);
}
