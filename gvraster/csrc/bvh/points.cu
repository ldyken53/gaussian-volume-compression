#include <torch/extension.h>
#include <cuda_runtime.h>
#include <unordered_map>
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include <cuBQL/bvh.h>
#include <cuBQL/builder/cuda.h>
#include "bvh.h"
#include "points.h"
#include "scratch.h"

__global__ void buildBoxes(cuBQL::box3f* boxes, const float* xyz, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;
	boxes[idx] = cuBQL::box3f(cuBQL::vec3f(xyz[3 * idx], xyz[3 * idx + 1], xyz[3 * idx + 2]));
}

__global__ void packPoints(float4* out, const float* xyz, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = make_float4(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2], 0.0f);
}

__global__ void flagPoints(float4* out, const float* flags, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		out[i].w = flags[i];
}

namespace
{

struct Cached {
	torch::Tensor xyz;
	cuBQL::bvh3f bvh;
};

Points current;
// A cached BVH belongs to the cache, not to current.
bool current_is_cached = false;
std::unordered_map<int64_t, Cached> cache;

void build(const float* xyz, int n, cuBQL::box3f* boxes, cuBQL::bvh3f& out)
{
	buildBoxes<<<(n + 255) / 256, 256>>>(boxes, xyz, n);
	cuBQL::BuildConfig cfg;
	cfg.makeLeafThreshold = 257;
	cuBQL::cuda::radixBuilder(out, boxes, n, cfg);
}

void pack()
{
	static float4* packed = nullptr;
	static size_t bytes = 0;
	const int n = current.xyz.size(0);
	current.packed = growScratch(packed, bytes, n);
	if (n > 0)
		packPoints<<<(n + 255) / 256, 256>>>(packed, current.xyz.data_ptr<float>(), n);
}

} // namespace

const Points& currentPoints() { return current; }

void buildPointBVH(const float* xyz, int n, cuBQL::bvh3f& out)
{
	cuBQL::box3f* boxes = nullptr;
	cudaMalloc(&boxes, (size_t)n * sizeof(cuBQL::box3f));
	build(xyz, n, boxes, out);
	cudaFree(boxes);
}

void setPoints(const torch::Tensor& points)
{
	static cuBQL::box3f* boxes = nullptr;
	static size_t bytes = 0;
	current.xyz = points.contiguous();
	pack();
	if (!current_is_cached)
		cuBQL::cuda::free(current.bvh);
	current_is_cached = false;
	current.bvh = cuBQL::bvh3f();
	const int n = current.xyz.size(0);
	build(current.xyz.data_ptr<float>(), n, growScratch(boxes, bytes, n), current.bvh);
}

// Full-batch training reuses one BVH per batch: no per-iteration build or its host readbacks.
void setPointsCached(const torch::Tensor& points, int64_t key)
{
	auto it = cache.find(key);
	if (it == cache.end()) {
		setPoints(points);
		cache[key] = Cached{current.xyz, current.bvh};
		current_is_cached = true;
		return;
	}
	if (!current_is_cached && current.bvh.nodes)
		cuBQL::cuda::free(current.bvh);
	current.xyz = it->second.xyz;
	current.bvh = it->second.bvh;
	current_is_cached = true;
	pack();
}

void setPointFlags(const torch::Tensor& flags)
{
	const int n = current.xyz.size(0);
	TORCH_CHECK(flags.is_cuda() && flags.scalar_type() == torch::kFloat32 && flags.numel() == n,
		"flags: float32 CUDA, one per current point");
	auto f = flags.contiguous();
	if (n > 0)
		flagPoints<<<(n + 255) / 256, 256>>>((float4*)current.packed, f.data_ptr<float>(), n);
}

torch::Tensor leafOrder()
{
	const int n = current.bvh.numPrims;
	auto order =
		torch::empty({n}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
	if (n > 0)
		cudaMemcpyAsync(order.data_ptr<int>(), current.bvh.primIDs, sizeof(uint32_t) * n,
			cudaMemcpyDeviceToDevice);
	return order;
}
