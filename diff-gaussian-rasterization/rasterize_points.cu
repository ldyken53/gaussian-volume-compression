#include "rasterize_points.h"
#include "cuda_rasterizer/backward.h"
#include "cuda_rasterizer/forward.h"
#include <cuda_runtime_api.h>

// Grow-only scratch: cudaMalloc and cudaFree synchronize the device.
static void* ensureScratch(void*& ptr, size_t& cap, size_t bytes)
{
	if (bytes > cap) {
		if (ptr)
			cudaFree(ptr);
		cudaMalloc(&ptr, bytes);
		cap = bytes;
	}
	return ptr;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizeGaussiansCUDA(
	const torch::Tensor& means3D, const torch::Tensor& scales, const torch::Tensor& rotations,
	const torch::Tensor& values, const torch::Tensor& weights, float occ_t, float floor,
	const torch::Tensor& samples, const cuBQL::bvh3f& bvh)
{
	TORCH_CHECK(means3D.ndimension() == 2 && means3D.size(1) == 3,
		"means3D must have dimensions (num_points, 3)");
	const int P = means3D.size(0);
	const int S = samples.size(0);
	auto float_opts = means3D.options().dtype(torch::kFloat32);
	torch::Tensor out_test = torch::zeros({S}, float_opts);
	torch::Tensor out_testw = torch::zeros({S}, float_opts);
	torch::Tensor conics = torch::zeros({P * 6}, float_opts);
	if (P != 0) {
		static void* aabbs_buf = nullptr;
		static size_t aabbs_cap = 0;
		auto* aabbs = (cuBQL::box3f*)ensureScratch(aabbs_buf, aabbs_cap, sizeof(cuBQL::box3f) * P);
		auto means = means3D.contiguous(), w = weights.contiguous();
		float* cn = conics.data_ptr<float>();
		FORWARD::preprocess(P, means.data_ptr<float>(),
			(glm::vec3*)scales.contiguous().data_ptr<float>(),
			(glm::vec4*)rotations.contiguous().data_ptr<float>(), w.data_ptr<float>(), cn, aabbs);
		FORWARD::render(P, S, means.data_ptr<float>(), values.contiguous().data_ptr<float>(),
			w.data_ptr<float>(), cn, aabbs, bvh, occ_t, floor, out_test.data_ptr<float>(),
			out_testw.data_ptr<float>());
	}
	return std::make_tuple(out_test, out_testw, conics);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(const torch::Tensor& means3D, const torch::Tensor& scales,
	const torch::Tensor& rotations, const torch::Tensor& conics, const torch::Tensor& values,
	const torch::Tensor& weights, const torch::Tensor& out_cells, const torch::Tensor& out_weights,
	const torch::Tensor& dL_dout_cells, const torch::Tensor& dL_dout_cell_weights, float occ_t,
	float floor, const torch::Tensor& samples, const cuBQL::bvh3f& bvh)
{
	const int P = means3D.size(0);
	const int S = samples.size(0);
	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dweights = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
	torch::Tensor dL_dvalues = torch::zeros({P, 1}, means3D.options());
	if (P != 0) {
		static void* dconics_buf = nullptr;
		static size_t dconics_cap = 0;
		float* dL_dconics = (float*)ensureScratch(dconics_buf, dconics_cap, sizeof(float) * P * 6);
		BACKWARD::render(P, S, means3D.contiguous().data_ptr<float>(),
			(glm::vec3*)scales.data_ptr<float>(), (glm::vec4*)rotations.data_ptr<float>(),
			conics.contiguous().data_ptr<float>(), values.contiguous().data_ptr<float>(),
			weights.contiguous().data_ptr<float>(), samples.contiguous().data_ptr<float>(), bvh,
			occ_t, floor, out_cells.contiguous().data_ptr<float>(),
			out_weights.contiguous().data_ptr<float>(),
			dL_dout_cells.contiguous().data_ptr<float>(),
			dL_dout_cell_weights.contiguous().data_ptr<float>(), dL_dmeans3D.data_ptr<float>(),
			dL_dvalues.data_ptr<float>(), dL_dweights.data_ptr<float>(),
			(glm::vec3*)dL_dscales.data_ptr<float>(), (glm::vec4*)dL_drotations.data_ptr<float>(),
			dL_dconics);
	}
	return std::make_tuple(dL_dmeans3D, dL_dscales, dL_drotations, dL_dvalues, dL_dweights);
}
