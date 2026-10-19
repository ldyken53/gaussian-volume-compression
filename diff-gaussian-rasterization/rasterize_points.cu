#include "cuda_rasterizer/rasterizer.h"
#include <cuda_runtime_api.h>
#include <functional>
#include <torch/extension.h>
#include <tuple>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t)
{
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(const torch::Tensor& means3D, const torch::Tensor& scales,
	const torch::Tensor& rotations, const torch::Tensor& values, const torch::Tensor& weights,
	const torch::Tensor& jitter, float max_scale, float min_x, float min_y, float min_z,
	float max_x, float max_y, float max_z, uint cell_count)
{
	TORCH_CHECK(means3D.ndimension() == 2 && means3D.size(1) == 3,
		"means3D must have dimensions (num_points, 3)");
	const int P = means3D.size(0);
	const uint3 num_cells = make_uint3(cell_count, cell_count, cell_count);
	auto float_opts = means3D.options().dtype(torch::kFloat32);
	// The render writes every cell.
	torch::Tensor out_cells = torch::empty({num_cells.x, num_cells.y, num_cells.z}, float_opts);
	torch::Tensor out_weights = torch::empty({num_cells.x, num_cells.y, num_cells.z}, float_opts);
	auto byte_opts = torch::TensorOptions(torch::kByte).device(torch::kCUDA);
	torch::Tensor geomBuffer = torch::empty({0}, byte_opts);
	torch::Tensor binningBuffer = torch::empty({0}, byte_opts);
	torch::Tensor imgBuffer = torch::empty({0}, byte_opts);

	int rendered = 0;
	if (P != 0)
		rendered = CudaRasterizer::Rasterizer::forward(resizeFunctional(geomBuffer),
			resizeFunctional(binningBuffer), resizeFunctional(imgBuffer), P,
			means3D.contiguous().data_ptr<float>(), scales.contiguous().data_ptr<float>(),
			rotations.contiguous().data_ptr<float>(), values.contiguous().data_ptr<float>(),
			weights.contiguous().data_ptr<float>(), max_scale,
			jitter.contiguous().data_ptr<float>(), make_float3(min_x, min_y, min_z),
			make_float3(max_x, max_y, max_z), num_cells, out_cells.data_ptr<float>(),
			out_weights.data_ptr<float>());
	else {
		out_cells.fill_(-1.0f);
		out_weights.zero_();
	}
	return std::make_tuple(rendered, out_cells, out_weights, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(const torch::Tensor& scales, const torch::Tensor& rotations,
	const torch::Tensor& values, const torch::Tensor& weights, const torch::Tensor& jitter,
	const torch::Tensor& out_cells, const torch::Tensor& out_weights, float max_scale, float min_x,
	float min_y, float min_z, float max_x, float max_y, float max_z, uint cell_count,
	const torch::Tensor& dL_dout_cells, const torch::Tensor& dL_dout_cell_weights,
	const torch::Tensor& geomBuffer, int R, const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer)
{
	const int P = scales.size(0);
	const uint3 num_cells = make_uint3(cell_count, cell_count, cell_count);
	auto opts = scales.options();
	// Per Gaussian: conic[6], mean[3], value, weight, pad.
	torch::Tensor grad_rows = torch::zeros({P, 12}, opts);
	torch::Tensor dL_dmeans = torch::empty({P, 3}, opts);
	torch::Tensor dL_dscales = torch::empty({P, 3}, opts);
	torch::Tensor dL_drotations = torch::empty({P, 4}, opts);
	torch::Tensor dL_dvalues = torch::empty({P, 1}, opts);
	torch::Tensor dL_dweights = torch::empty({P, 1}, opts);

	if (P != 0)
		CudaRasterizer::Rasterizer::backward(P, R, scales.contiguous().data_ptr<float>(),
			rotations.contiguous().data_ptr<float>(), values.contiguous().data_ptr<float>(),
			weights.contiguous().data_ptr<float>(), max_scale,
			jitter.contiguous().data_ptr<float>(), make_float3(min_x, min_y, min_z),
			make_float3(max_x, max_y, max_z), num_cells, out_cells.contiguous().data_ptr<float>(),
			out_weights.contiguous().data_ptr<float>(),
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
			dL_dout_cells.contiguous().data_ptr<float>(),
			dL_dout_cell_weights.contiguous().data_ptr<float>(), grad_rows.data_ptr<float>(),
			dL_dmeans.data_ptr<float>(), dL_dscales.data_ptr<float>(),
			dL_drotations.data_ptr<float>(), dL_dvalues.data_ptr<float>(),
			dL_dweights.data_ptr<float>());
	return std::make_tuple(dL_dmeans, dL_dscales, dL_drotations, dL_dvalues, dL_dweights);
}
