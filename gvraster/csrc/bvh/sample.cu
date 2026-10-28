#include "backward.h"
#include "bvh.h"
#include "forward.h"
#include "points.h"
#include "scratch.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sampleForward(const torch::Tensor& means,
	const torch::Tensor& scales, const torch::Tensor& rotations, const torch::Tensor& values,
	const torch::Tensor& weights, double threshold, double floor,
	const std::optional<torch::Tensor>& reach)
{
	TORCH_CHECK(means.ndimension() == 2 && means.size(1) == 3, "means must be (P, 3)");
	const Points& points = currentPoints();
	const int P = means.size(0);
	const int S = points.xyz.size(0);
	TORCH_CHECK(!reach || (reach->is_cuda() && reach->scalar_type() == torch::kFloat32 &&
							  reach->is_contiguous() && reach->numel() == P),
		"reach: contiguous float32 CUDA, one per Gaussian");
	auto opts = means.options().dtype(torch::kFloat32);
	torch::Tensor out_values = torch::zeros({S}, opts);
	torch::Tensor out_influence = torch::zeros({S}, opts);
	torch::Tensor conics = torch::zeros({P * 6}, opts);
	if (P != 0) {
		static cuBQL::box3f* boxes = nullptr;
		static size_t bytes = 0;
		growScratch(boxes, bytes, P);
		auto m = means.contiguous(), w = weights.contiguous();
		FORWARD::prepare(P, m.data_ptr<float>(), (glm::vec3*)scales.contiguous().data_ptr<float>(),
			(glm::vec4*)rotations.contiguous().data_ptr<float>(), w.data_ptr<float>(),
			conics.data_ptr<float>(), boxes);
		FORWARD::sample(P, S, m.data_ptr<float>(), values.contiguous().data_ptr<float>(),
			w.data_ptr<float>(), conics.data_ptr<float>(), boxes, points.packed, points.bvh,
			(float)threshold, (float)floor, out_values.data_ptr<float>(),
			out_influence.data_ptr<float>(), reach ? reach->data_ptr<float>() : nullptr);
	}
	return std::make_tuple(out_values, out_influence, conics);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sampleBackward(const torch::Tensor& means, const torch::Tensor& scales,
	const torch::Tensor& rotations, const torch::Tensor& conics, const torch::Tensor& values,
	const torch::Tensor& weights, const torch::Tensor& out_values,
	const torch::Tensor& out_influence, const torch::Tensor& dL_dout_values,
	const torch::Tensor& dL_dout_influence, double threshold, double floor)
{
	const Points& points = currentPoints();
	const int P = means.size(0);
	const int S = points.xyz.size(0);
	torch::Tensor dL_dmeans = torch::zeros({P, 3}, means.options());
	torch::Tensor dL_dscales = torch::zeros({P, 3}, means.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means.options());
	torch::Tensor dL_dvalues = torch::zeros({P, 1}, means.options());
	torch::Tensor dL_dweights = torch::zeros({P, 1}, means.options());
	if (P != 0) {
		static float* dL_dconics = nullptr;
		static size_t bytes = 0;
		growScratch(dL_dconics, bytes, (size_t)P * 6);
		BACKWARD::sample(P, S, means.contiguous().data_ptr<float>(),
			(glm::vec3*)scales.contiguous().data_ptr<float>(),
			(glm::vec4*)rotations.contiguous().data_ptr<float>(),
			conics.contiguous().data_ptr<float>(), values.contiguous().data_ptr<float>(),
			weights.contiguous().data_ptr<float>(), points.xyz.data_ptr<float>(), points.bvh,
			(float)threshold, (float)floor, out_values.contiguous().data_ptr<float>(),
			out_influence.contiguous().data_ptr<float>(),
			dL_dout_values.contiguous().data_ptr<float>(),
			dL_dout_influence.contiguous().data_ptr<float>(), dL_dmeans.data_ptr<float>(),
			dL_dvalues.data_ptr<float>(), dL_dweights.data_ptr<float>(),
			(glm::vec3*)dL_dscales.data_ptr<float>(), (glm::vec4*)dL_drotations.data_ptr<float>(),
			dL_dconics);
	}
	return std::make_tuple(dL_dmeans, dL_dscales, dL_drotations, dL_dvalues, dL_dweights);
}
