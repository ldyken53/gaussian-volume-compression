#pragma once
#include <cuBQL/bvh.h>
#include <torch/extension.h>
#include <tuple>

// Renders the activated Gaussians at the samples of `bvh`; returns (cells, weights, conics).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizeGaussiansCUDA(
	const torch::Tensor& means3D, const torch::Tensor& scales, const torch::Tensor& rotations,
	const torch::Tensor& values, const torch::Tensor& weights, bool hard,
	const torch::Tensor& samples, const cuBQL::bvh3f& bvh);

// Gradients w.r.t. (means3D, scales, rotations, values, weights).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(const torch::Tensor& means3D, const torch::Tensor& scales,
	const torch::Tensor& rotations, const torch::Tensor& conics, const torch::Tensor& values,
	const torch::Tensor& weights, const torch::Tensor& out_cells, const torch::Tensor& out_weights,
	const torch::Tensor& dL_dout_cells, const torch::Tensor& dL_dout_cell_weights, bool hard,
	const torch::Tensor& samples, const cuBQL::bvh3f& bvh);
