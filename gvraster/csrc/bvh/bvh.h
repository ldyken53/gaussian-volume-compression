// Host entry points of gvraster._bvh.
#pragma once
#include <torch/extension.h>
#include <tuple>
#include <vector>

void setPoints(const torch::Tensor& points);
void setPointsCached(const torch::Tensor& points, int64_t key);
torch::Tensor leafOrder();
// One float per current point, kept in its packed 4th float: > 0 flags it for the reach record.
void setPointFlags(const torch::Tensor& flags);

// reach, if given (P): set to each Gaussian's largest w G at a flagged point.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sampleForward(const torch::Tensor& means,
	const torch::Tensor& scales, const torch::Tensor& rotations, const torch::Tensor& values,
	const torch::Tensor& weights, double threshold, double floor,
	const std::optional<torch::Tensor>& reach);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sampleBackward(const torch::Tensor& means, const torch::Tensor& scales,
	const torch::Tensor& rotations, const torch::Tensor& conics, const torch::Tensor& values,
	const torch::Tensor& weights, const torch::Tensor& out_values,
	const torch::Tensor& out_influence, const torch::Tensor& dL_dout_values,
	const torch::Tensor& dL_dout_influence, double threshold, double floor);

std::vector<torch::Tensor> activateForward(torch::Tensor raw_scales, torch::Tensor raw_rotations,
	torch::Tensor raw_values, torch::Tensor raw_weights, double max_scale);
std::vector<torch::Tensor> activateBackward(torch::Tensor raw_scales, torch::Tensor raw_rotations,
	torch::Tensor values, torch::Tensor weights, double max_scale, torch::Tensor dL_dscales,
	torch::Tensor dL_drotations, torch::Tensor dL_dvalues, torch::Tensor dL_dweights);

void marginSetExterior(torch::Tensor points, int64_t key);
std::vector<torch::Tensor> marginFP(
	torch::Tensor means, torch::Tensor conics, torch::Tensor weights, double T, int64_t key);
