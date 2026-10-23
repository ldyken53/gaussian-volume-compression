// Host entry points of gvraster._tile.
#pragma once
#include <torch/extension.h>
#include <tuple>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sampleForward(const torch::Tensor& means, const torch::Tensor& raw_scales,
	const torch::Tensor& raw_rotations, const torch::Tensor& raw_values,
	const torch::Tensor& raw_weights, const torch::Tensor& jitter, float min_x, float min_y,
	float min_z, float max_x, float max_y, float max_z, uint cell_count, float max_scale,
	float threshold, float floor);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sampleBackward(const torch::Tensor& raw_scales, const torch::Tensor& raw_rotations,
	const torch::Tensor& raw_values, const torch::Tensor& raw_weights, const torch::Tensor& jitter,
	float min_x, float min_y, float min_z, float max_x, float max_y, float max_z, uint cell_count,
	float max_scale, float threshold, float floor, const torch::Tensor& out_values,
	const torch::Tensor& out_influence, const torch::Tensor& dL_dout_values,
	const torch::Tensor& dL_dout_influence, int pairs, const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer, const torch::Tensor& range_buffer);
