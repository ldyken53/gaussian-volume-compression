#include <torch/extension.h>
#include <cuda_runtime.h>
#include <unordered_map>
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include <cuBQL/bvh.h>
#include "cuBQL/builder/cuda.h"
#include "rasterize_points.h"

void multi_adam_(std::vector<torch::Tensor> params, std::vector<torch::Tensor> grads,
	std::vector<torch::Tensor> exp_avgs, std::vector<torch::Tensor> exp_avg_sqs,
	std::vector<double> lrs, int64_t step, double beta1, double beta2, double eps,
	std::vector<torch::Tensor> emas, double ema_w);
std::vector<torch::Tensor> fused_act_forward(
	torch::Tensor rs, torch::Tensor rq, torch::Tensor rv, torch::Tensor rw, double max_scale);
std::vector<torch::Tensor> fused_act_backward(torch::Tensor rs, torch::Tensor rq, torch::Tensor ov,
	torch::Tensor ow, double max_scale, torch::Tensor gs, torch::Tensor gq, torch::Tensor gv,
	torch::Tensor gw);
std::vector<torch::Tensor> fused_loss_forward(torch::Tensor c, torch::Tensor g, torch::Tensor w,
	int64_t l1, int64_t fn_log, int64_t rel, double tau, double fn_reg, double fp_reg, double fp_t,
	int64_t fp_kind, int64_t norm_all, double rel_pow);
std::vector<torch::Tensor> fused_loss_backward(torch::Tensor c, torch::Tensor g, torch::Tensor w,
	torch::Tensor acc, int64_t l1, int64_t fn_log, int64_t rel, double tau, double fn_reg,
	double fp_reg, double fp_t, int64_t fp_kind, int64_t norm_all, double rel_pow,
	torch::Tensor gout);
void margin_set_exterior(torch::Tensor samples, int64_t key);
std::vector<torch::Tensor> margin_fp(
	torch::Tensor means, torch::Tensor conic, torch::Tensor weights, double T, int64_t key);

// The current sample batch and its BVH, used by every render until the next build.
static cuBQL::bvh3f samples_bvh;
static torch::Tensor stored_samples;
static bool samples_bvh_cached = false;

__global__ void buildBoxes(cuBQL::box3f* aabbs, const float* samples, int num_samples)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_samples)
		return;
	aabbs[idx] =
		cuBQL::box3f(cuBQL::vec3f(samples[3 * idx], samples[3 * idx + 1], samples[3 * idx + 2]));
}

// Standalone point BVH, owned by the caller (margin.cu's exterior sets).
void buildPointBVH(const float* samples, int n, cuBQL::bvh3f& out)
{
	cuBQL::box3f* boxes = nullptr;
	cudaMalloc(&boxes, (size_t)n * sizeof(cuBQL::box3f));
	buildBoxes<<<(n + 255) / 256, 256>>>(boxes, samples, n);
	cuBQL::BuildConfig cfg;
	cfg.makeLeafThreshold = 257;
	cuBQL::cuda::radixBuilder(out, boxes, n, cfg);
	cudaFree(boxes);
}

// Sample positions as float4s: one aligned load per lane in the forward render.
__global__ void packSamplePos(float4* out, const float* samples, int N)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
		out[i] = make_float4(samples[3 * i], samples[3 * i + 1], samples[3 * i + 2], 0.0f);
}
static float4* g_sample_pos = nullptr;
static size_t g_sample_pos_cap = 0;
const float4* gvSamplePos() { return g_sample_pos; }

static void packSamples()
{
	const int N = stored_samples.size(0);
	if ((size_t)N > g_sample_pos_cap) {
		if (g_sample_pos)
			cudaFree(g_sample_pos);
		cudaMalloc(&g_sample_pos, (size_t)N * sizeof(float4));
		g_sample_pos_cap = N;
	}
	if (N > 0)
		packSamplePos<<<(N + 255) / 256, 256>>>(g_sample_pos, stored_samples.data_ptr<float>(), N);
}

void BuildBVH(const torch::Tensor& samples)
{
	stored_samples = samples.contiguous();
	packSamples();
	const int N = stored_samples.size(0);
	// A cached BVH belongs to the cache.
	if (!samples_bvh_cached)
		cuBQL::cuda::free(samples_bvh);
	samples_bvh_cached = false;
	samples_bvh = cuBQL::bvh3f();
	static cuBQL::box3f* d_boxes = nullptr;
	static size_t d_boxes_cap = 0;
	if ((size_t)N * sizeof(cuBQL::box3f) > d_boxes_cap) {
		if (d_boxes)
			cudaFree(d_boxes);
		cudaMalloc(&d_boxes, N * sizeof(cuBQL::box3f));
		d_boxes_cap = (size_t)N * sizeof(cuBQL::box3f);
	}
	buildBoxes<<<(N + 255) / 256, 256>>>(d_boxes, stored_samples.data_ptr<float>(), N);
	cuBQL::BuildConfig cfg;
	cfg.makeLeafThreshold = 257;
	cuBQL::cuda::radixBuilder(samples_bvh, d_boxes, N, cfg);
}

// Full-batch runs reuse one BVH per batch: no per-iteration build or its host readbacks.
struct CachedBVH {
	cuBQL::bvh3f bvh;
	torch::Tensor samples;
};
static std::unordered_map<int64_t, CachedBVH> bvh_cache;

void BuildBVHCached(const torch::Tensor& samples, const int64_t key)
{
	auto it = bvh_cache.find(key);
	if (it == bvh_cache.end()) {
		BuildBVH(samples);
		bvh_cache[key] = CachedBVH{samples_bvh, stored_samples};
		samples_bvh_cached = true;
		return;
	}
	stored_samples = it->second.samples;
	samples_bvh = it->second.bvh;
	samples_bvh_cached = true;
	packSamples();
}

// primIDs of the current BVH in leaf order (train.py permutes each batch into this order).
torch::Tensor SampleBVHPrimIDs()
{
	const int N = samples_bvh.numPrims;
	auto t = torch::empty({N}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
	if (N > 0)
		cudaMemcpyAsync(
			t.data_ptr<int>(), samples_bvh.primIDs, sizeof(uint32_t) * N, cudaMemcpyDeviceToDevice);
	return t;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Rasterize(const torch::Tensor& means3D,
	const torch::Tensor& scales, const torch::Tensor& rotations, const torch::Tensor& values,
	const torch::Tensor& weights, bool hard)
{
	return RasterizeGaussiansCUDA(
		means3D, scales, rotations, values, weights, hard, stored_samples, samples_bvh);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeBackward(const torch::Tensor& means3D, const torch::Tensor& scales,
	const torch::Tensor& rotations, const torch::Tensor& conics, const torch::Tensor& values,
	const torch::Tensor& weights, const torch::Tensor& out_cells, const torch::Tensor& out_weights,
	const torch::Tensor& dL_dout_cells, const torch::Tensor& dL_dout_cell_weights, bool hard)
{
	return RasterizeGaussiansBackwardCUDA(means3D, scales, rotations, conics, values, weights,
		out_cells, out_weights, dL_dout_cells, dL_dout_cell_weights, hard, stored_samples,
		samples_bvh);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("build_bvh", &BuildBVH);
	m.def("build_bvh_cached", &BuildBVHCached);
	m.def("sample_bvh_prim_ids", &SampleBVHPrimIDs);
	m.def("rasterize_gaussians", &Rasterize);
	m.def("rasterize_gaussians_backward", &RasterizeBackward);
	m.def("multi_adam_", &multi_adam_);
	m.def("fused_act_forward", &fused_act_forward);
	m.def("fused_act_backward", &fused_act_backward);
	m.def("fused_loss_forward", &fused_loss_forward);
	m.def("fused_loss_backward", &fused_loss_backward);
	m.def("margin_set_exterior", &margin_set_exterior);
	m.def("margin_fp", &margin_fp);
}
