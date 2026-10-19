#pragma once
#include <torch/extension.h>
#include <tuple>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(const torch::Tensor& means3D, const torch::Tensor& scales,
	const torch::Tensor& rotations, const torch::Tensor& values, const torch::Tensor& weights,
	const torch::Tensor& jitter, float max_scale, float min_x, float min_y, float min_z,
	float max_x, float max_y, float max_z, uint cell_count);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(const torch::Tensor& scales, const torch::Tensor& rotations,
	const torch::Tensor& values, const torch::Tensor& weights, const torch::Tensor& jitter,
	const torch::Tensor& out_cells, const torch::Tensor& out_weights, float max_scale, float min_x,
	float min_y, float min_z, float max_x, float max_y, float max_z, uint cell_count,
	const torch::Tensor& dL_dout_cells, const torch::Tensor& dL_dout_cell_weights,
	const torch::Tensor& geomBuffer, int R, const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer);
