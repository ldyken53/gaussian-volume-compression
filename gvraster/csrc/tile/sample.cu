#include "tile.h"
#include "tiles.h"
#include <cuda_runtime_api.h>
#include <functional>
#include <tuple>

// Storage from a byte tensor that the autograd context keeps until the backward pass.
static tiles::Allocator resizer(torch::Tensor& t)
{
	return [&t](size_t n) {
		t.resize_({(long long)n});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sampleForward(const torch::Tensor& means, const torch::Tensor& raw_scales,
	const torch::Tensor& raw_rotations, const torch::Tensor& raw_values,
	const torch::Tensor& raw_weights, const torch::Tensor& jitter, float min_x, float min_y,
	float min_z, float max_x, float max_y, float max_z, uint cell_count, float max_scale,
	float threshold, float floor)
{
	TORCH_CHECK(means.ndimension() == 2 && means.size(1) == 3, "means must be (P, 3)");
	const int P = means.size(0);
	const uint3 cells = make_uint3(cell_count, cell_count, cell_count);
	auto opts = means.options().dtype(torch::kFloat32);
	// Every lattice point is written.
	torch::Tensor out_values = torch::empty({cells.x, cells.y, cells.z}, opts);
	torch::Tensor out_influence = torch::empty({cells.x, cells.y, cells.z}, opts);
	auto bytes = torch::TensorOptions(torch::kByte).device(torch::kCUDA);
	torch::Tensor geom_buffer = torch::empty({0}, bytes);
	torch::Tensor binning_buffer = torch::empty({0}, bytes);
	torch::Tensor range_buffer = torch::empty({0}, bytes);

	int pairs = 0;
	if (P != 0)
		pairs = tiles::forward(resizer(geom_buffer), resizer(binning_buffer), resizer(range_buffer),
			P, means.contiguous().data_ptr<float>(), raw_scales.contiguous().data_ptr<float>(),
			raw_rotations.contiguous().data_ptr<float>(), raw_values.contiguous().data_ptr<float>(),
			raw_weights.contiguous().data_ptr<float>(), max_scale,
			jitter.contiguous().data_ptr<float>(), make_float3(min_x, min_y, min_z),
			make_float3(max_x, max_y, max_z), cells, out_values.data_ptr<float>(),
			out_influence.data_ptr<float>(), threshold, floor);
	else {
		out_values.fill_(-1.0f);
		out_influence.zero_();
	}
	return std::make_tuple(
		pairs, out_values, out_influence, geom_buffer, binning_buffer, range_buffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sampleBackward(const torch::Tensor& raw_scales, const torch::Tensor& raw_rotations,
	const torch::Tensor& raw_values, const torch::Tensor& raw_weights, const torch::Tensor& jitter,
	float min_x, float min_y, float min_z, float max_x, float max_y, float max_z, uint cell_count,
	float max_scale, float threshold, float floor, const torch::Tensor& out_values,
	const torch::Tensor& out_influence, const torch::Tensor& dL_dout_values,
	const torch::Tensor& dL_dout_influence, int pairs, const torch::Tensor& geom_buffer,
	const torch::Tensor& binning_buffer, const torch::Tensor& range_buffer)
{
	const int P = raw_scales.size(0);
	const uint3 cells = make_uint3(cell_count, cell_count, cell_count);
	auto opts = raw_scales.options();
	// Per Gaussian: conic[6], mean[3], value, weight, pad.
	torch::Tensor grad_rows = torch::zeros({P, 12}, opts);
	torch::Tensor dL_dmeans = torch::empty({P, 3}, opts);
	torch::Tensor dL_dscales = torch::empty({P, 3}, opts);
	torch::Tensor dL_drotations = torch::empty({P, 4}, opts);
	torch::Tensor dL_dvalues = torch::empty({P, 1}, opts);
	torch::Tensor dL_dweights = torch::empty({P, 1}, opts);

	if (P != 0)
		tiles::backward(P, pairs, raw_scales.contiguous().data_ptr<float>(),
			raw_rotations.contiguous().data_ptr<float>(), raw_values.contiguous().data_ptr<float>(),
			raw_weights.contiguous().data_ptr<float>(), max_scale,
			jitter.contiguous().data_ptr<float>(), make_float3(min_x, min_y, min_z),
			make_float3(max_x, max_y, max_z), cells, out_values.contiguous().data_ptr<float>(),
			out_influence.contiguous().data_ptr<float>(),
			reinterpret_cast<char*>(geom_buffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binning_buffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(range_buffer.contiguous().data_ptr()),
			dL_dout_values.contiguous().data_ptr<float>(),
			dL_dout_influence.contiguous().data_ptr<float>(), grad_rows.data_ptr<float>(),
			dL_dmeans.data_ptr<float>(), dL_dscales.data_ptr<float>(),
			dL_drotations.data_ptr<float>(), dL_dvalues.data_ptr<float>(),
			dL_dweights.data_ptr<float>(), threshold, floor);
	return std::make_tuple(dL_dmeans, dL_dscales, dL_drotations, dL_dvalues, dL_dweights);
}
