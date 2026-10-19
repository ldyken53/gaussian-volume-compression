#include <torch/extension.h>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include <cuBQL/bvh.h>
#include "cuBQL/builder/cuda.h"
#include "rasterize_points.h"

// fused_ops.cu / multi_adam.cu (ported from gaussian-volume-struct)
void multi_adam_(std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
                 std::vector<torch::Tensor>, std::vector<double>, int64_t, double, double, double);
void multi_adam_ema_(std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
                     std::vector<torch::Tensor>, std::vector<double>, int64_t, double, double, double,
                     std::vector<torch::Tensor>, double);
void margin_set_exterior(torch::Tensor, int64_t);
std::vector<torch::Tensor> margin_fp(torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t);
void multi_adam_ex_(std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
                    std::vector<torch::Tensor>, std::vector<double>, int64_t, double, double,
                    std::vector<double>, std::vector<double>, std::vector<torch::Tensor>, double);
std::vector<torch::Tensor> fused_act_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
std::vector<torch::Tensor> fused_act_backward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double,
                                              torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> fused_loss_forward(torch::Tensor, torch::Tensor, torch::Tensor,
                                              int64_t, int64_t, int64_t, int64_t, double, double, double, double, int64_t,
                                              int64_t, double);
std::vector<torch::Tensor> fused_loss_backward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                               int64_t, int64_t, int64_t, int64_t, double, double, double, double,
                                               int64_t, int64_t, double, torch::Tensor);

static cuBQL::bvh3f samples_bvh;
static cuBQL::bvh3f gaussian_bvh;
static torch::Tensor stored_samples;

// buildBoxes: one thread per sample; packs (x,y,z) into a box3f at that point
__global__ void buildBoxes(
    cuBQL::box3f*  aabbs,
    const float*   samples,
    int            num_samples
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;

    float x = samples[3*idx + 0];
    float y = samples[3*idx + 1];
    float z = samples[3*idx + 2];
    aabbs[idx] = cuBQL::box3f(cuBQL::vec3f(x, y, z));
}

// ---------------------------------------------------------------------------------------
// Sync-free sample BVH. The training batch is Morton-sorted (train.py sorts every batch and
// subsamples in ascending order), so consecutive runs of L samples are spatially compact and
// can be the leaves directly: no sort, and none of the per-level host readbacks cuBQL's
// radix builder does (16-19 cudaEventSynchronize per build, which kept the CPU from queueing
// ahead -- nsys showed the GPU idle ~45% of a mito 1024x iteration). The tree is a complete
// binary heap over the leaves (children of node i at 2i+1, 2i+2, adjacent as cuBQL's
// traversal requires); leaf slots past the last real leaf get an empty box and are never
// entered. primIDs is the identity. Opt-in with GV_BVH=morton (BVH_LEAF sets L): measured
// slower overall, see useCubqlBuilder().
// ---------------------------------------------------------------------------------------
using bvh_node_t = cuBQL::bvh3f::node_t;

__global__ void mortonLeafBoxes(bvh_node_t* nodes, const float* samples, int N, int L,
                                int nLeaves, int nPow2)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= nPow2) return;
    float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    const int begin = warp * L;
    const int end = min(N, begin + L);
    if (warp < nLeaves)
        for (int i = begin + lane; i < end; i += 32)
            for (int d = 0; d < 3; d++) {
                const float v = samples[3 * i + d];
                lo[d] = fminf(lo[d], v); hi[d] = fmaxf(hi[d], v);
            }
    for (int o = 16; o > 0; o >>= 1)
        for (int d = 0; d < 3; d++) {
            lo[d] = fminf(lo[d], __shfl_down_sync(0xffffffff, lo[d], o));
            hi[d] = fmaxf(hi[d], __shfl_down_sync(0xffffffff, hi[d], o));
        }
    if (lane == 0) {
        bvh_node_t& n = nodes[nPow2 - 1 + warp];
        n.bounds = cuBQL::box3f(cuBQL::vec3f(lo[0], lo[1], lo[2]), cuBQL::vec3f(hi[0], hi[1], hi[2]));
        n.admin.offset = warp < nLeaves ? begin : 0;
        n.admin.count = warp < nLeaves ? end - begin : 1;  // padding: empty box, never entered
    }
}

__global__ void mortonInnerLevel(bvh_node_t* nodes, int first, int count)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const int i = first + k;
    const cuBQL::box3f a = nodes[2 * i + 1].bounds, b = nodes[2 * i + 2].bounds;
    bvh_node_t& n = nodes[i];
    n.bounds = cuBQL::box3f(
        cuBQL::vec3f(fminf(a.lower.x, b.lower.x), fminf(a.lower.y, b.lower.y), fminf(a.lower.z, b.lower.z)),
        cuBQL::vec3f(fmaxf(a.upper.x, b.upper.x), fmaxf(a.upper.y, b.upper.y), fmaxf(a.upper.z, b.upper.z)));
    n.admin.offset = 2 * i + 1;
    n.admin.count = 0;
}

__global__ void iotaU32(uint32_t* p, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = i;
}

static void buildMortonBVH(const float* samples, int N)
{
    static int L = -1;
    if (L < 0) {
        const char* e = getenv("BVH_LEAF");
        L = e ? atoi(e) : 128;
        if (L < 1 || L > 65535) L = 128;
    }
    static bvh_node_t* nodes = nullptr;
    static uint32_t* prims = nullptr;
    static size_t nodeCap = 0, primCap = 0;
    const int nLeaves = std::max(1, (N + L - 1) / L);
    int nPow2 = 1;
    while (nPow2 < nLeaves) nPow2 <<= 1;
    const size_t numNodes = 2 * (size_t)nPow2 - 1;
    // Grow-only: N is fixed for a whole run, so this allocates (and syncs) once.
    if (numNodes > nodeCap) {
        if (nodes) cudaFree(nodes);
        cudaMalloc(&nodes, numNodes * sizeof(bvh_node_t));
        nodeCap = numNodes;
    }
    if ((size_t)N > primCap) {
        if (prims) cudaFree(prims);
        cudaMalloc(&prims, (size_t)N * sizeof(uint32_t));
        primCap = N;
        iotaU32<<<(N + 255) / 256, 256>>>(prims, N);
    }
    mortonLeafBoxes<<<(nPow2 * 32 + 255) / 256, 256>>>(nodes, samples, N, L, nLeaves, nPow2);
    for (int level = nPow2 / 2; level >= 1; level >>= 1)
        mortonInnerLevel<<<(level + 255) / 256, 256>>>(nodes, level - 1, level);
    samples_bvh.nodes = nodes;
    samples_bvh.primIDs = prims;
    samples_bvh.numNodes = (uint32_t)numNodes;
    samples_bvh.numPrims = N;
}

// Standalone point BVH (margin.cu's exterior sets); caller owns it.
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

__global__ void packSamplePos(float4* out, const float* samples, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = make_float4(samples[3 * i], samples[3 * i + 1], samples[3 * i + 2], 0.0f);
}
static float4* g_sample_pos = nullptr;
static size_t g_sample_pos_cap = 0;
// Read by forward.cu's render launch.
const float4* gvSamplePos() { return g_sample_pos; }

static bool useCubqlBuilder()
{
    static int v = -1;
    // Default cuBQL: the fixed-size Morton leaves straddle Z-order jumps, and on valley 64x
    // their loose boxes cost the render kernels +37..+100% for the 0.9 ms the build saves.
    if (v < 0) { const char* e = getenv("GV_BVH"); v = (e && std::string(e) == "morton") ? 0 : 1; }
    return v == 1;
}

// Per-batch BVH cache. A full-batch run trains on a fixed set of batches (100 by default),
// so each batch's sample BVH can be built once and reused: no per-iteration build, and none
// of the radix builder's per-level host readbacks (16-19 syncs per build, which also stop
// the CPU from queueing ahead). A cached BVH is owned by the cache and never freed by the
// rebuild path below. Cached entries keep a reference to their sample tensor.
#include <unordered_map>
struct CachedBVH { cuBQL::bvh3f bvh; torch::Tensor samples; };
static std::unordered_map<int64_t, CachedBVH> bvh_cache;
static bool samples_bvh_cached = false;

void BuildBVH(const torch::Tensor& samples, const bool debug, const bool use_gaussian_bvh);

void BuildBVHCached(const torch::Tensor& samples, const int64_t key) {
    auto it = bvh_cache.find(key);
    if (it == bvh_cache.end()) {
        BuildBVH(samples, false, false);                 // builds into samples_bvh
        bvh_cache[key] = CachedBVH{samples_bvh, stored_samples};
        samples_bvh_cached = true;                       // now owned by the cache
        return;
    }
    stored_samples = it->second.samples;
    samples_bvh = it->second.bvh;
    samples_bvh_cached = true;
    const int N = stored_samples.size(0);
    if ((size_t)N > g_sample_pos_cap) {
        if (g_sample_pos) cudaFree(g_sample_pos);
        cudaMalloc(&g_sample_pos, (size_t)N * sizeof(float4));
        g_sample_pos_cap = N;
    }
    if (N > 0) packSamplePos<<<(N + 255) / 256, 256>>>(g_sample_pos, stored_samples.data_ptr<float>(), N);
}

// Host entrypoint: alloc → kernel → build BVH → free temp buffer
void BuildBVH(const torch::Tensor& samples, const bool debug, const bool use_gaussian_bvh) {
    stored_samples = samples.contiguous();
    {
        const int N = stored_samples.size(0);
        if ((size_t)N > g_sample_pos_cap) {   // grow-only: N is fixed for a run
            if (g_sample_pos) cudaFree(g_sample_pos);
            cudaMalloc(&g_sample_pos, (size_t)N * sizeof(float4));
            g_sample_pos_cap = N;
        }
        if (N > 0) packSamplePos<<<(N + 255) / 256, 256>>>(g_sample_pos, stored_samples.data_ptr<float>(), N);
    }
    if (!use_gaussian_bvh && !useCubqlBuilder()) {
        buildMortonBVH(stored_samples.data_ptr<float>(), stored_samples.size(0));
        samples_bvh_cached = false;
        return;
    }
    if (!use_gaussian_bvh) {
        // Event timing only when asked for: the unconditional cudaEventSynchronize
        // here was a full device sync on every training iteration (and the events
        // were never destroyed).
        cudaEvent_t gpuStart, gpuStop;
        if (debug) {
            cudaEventCreate(&gpuStart);
            cudaEventCreate(&gpuStop);
            cudaEventRecord(gpuStart, 0);
        }
        int  N = stored_samples.size(0);
        auto ptr = stored_samples.data_ptr<float>();

        // (A Morton-built BVH's buffers are persistent statics, never cuBQL's to free; the
        // two builders are not mixed within a run.)
        if (!samples_bvh_cached) cuBQL::cuda::free(samples_bvh);
        samples_bvh_cached = false;
        samples_bvh.nodes    = nullptr;
        samples_bvh.primIDs  = nullptr;
        samples_bvh.numNodes = 0;
        samples_bvh.numPrims = 0;
        samples_bvh = cuBQL::bvh3f();
        // Grow-only box scratch: N is fixed for a whole training run.
        static cuBQL::box3f* d_boxes = nullptr;
        static size_t d_boxes_cap = 0;
        if ((size_t)N * sizeof(cuBQL::box3f) > d_boxes_cap) {
            if (d_boxes) cudaFree(d_boxes);
            cudaMalloc(&d_boxes, N * sizeof(cuBQL::box3f));
            d_boxes_cap = (size_t)N * sizeof(cuBQL::box3f);
        }

        const int threads = 256;
        const int blocks  = (N + threads - 1) / threads;
        buildBoxes<<<blocks, threads>>>(d_boxes, ptr, N);
        cuBQL::BuildConfig cfg;
        // BVH_LEAF_THRESH: leaf-size knob (default 257). Bigger leaves trade traversal depth --
        // ~45% of the render kernels' stalls (ncu) -- for more per-sample tests.
        static int leafThresh = -1;
        if (leafThresh < 0) { const char* e = getenv("BVH_LEAF_THRESH"); leafThresh = e ? atoi(e) : 257; }
        cfg.makeLeafThreshold = leafThresh;
        // cfg.maxAllowedLeafSize = 256;
        cuBQL::cuda::radixBuilder(samples_bvh, d_boxes, N, cfg);
        // cuBQL::gpuBuilder(samples_bvh, d_boxes, N, cfg);

        if (debug) {
            cudaEventRecord(gpuStop, 0);
            cudaEventSynchronize(gpuStop);
            float msBoxes = 0.f;
            cudaEventElapsedTime(&msBoxes, gpuStart, gpuStop);
            std::cout << "Sample BVH time: " << msBoxes << " ms\n";
            cudaEventDestroy(gpuStart);
            cudaEventDestroy(gpuStop);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDAWrapper(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const torch::Tensor& values,
	const torch::Tensor& weights,
	const float scale_modifier,
	const float min_x, const float min_y, const float min_z, 
	const float max_x, const float max_y, const float max_z,
	const float background,
    const bool use_gaussian_bvh,
	const bool debug
) {
    // Only when one was built: cuBQL's free costs a stream sync even on an empty BVH.
    if (gaussian_bvh.nodes) cuBQL::cuda::free(gaussian_bvh);
    gaussian_bvh.nodes    = nullptr;
    gaussian_bvh.primIDs  = nullptr;
    gaussian_bvh.numNodes = 0;
    gaussian_bvh.numPrims = 0;
    gaussian_bvh = cuBQL::bvh3f();

    return RasterizeGaussiansCUDA(
        means3D,
        scales,
        rotations,
        values,
        weights,
        scale_modifier,
        min_x, min_y, min_z,
        max_x, max_y, max_z,
        background,
        use_gaussian_bvh,
        debug,
        stored_samples,
        samples_bvh,
        gaussian_bvh
    );

}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDAWrapper(
	const torch::Tensor& means3D,
	const torch::Tensor& scales,
    const torch::Tensor& rotations,
	const torch::Tensor& conics,
	const torch::Tensor& values,
	const torch::Tensor& weights,
	const torch::Tensor& out_cells,
	const torch::Tensor& out_weights,
	const float scale_modifier,
	const float min_x, const float min_y, const float min_z, 
	const float max_x, const float max_y, const float max_z,
	const float background,
	const torch::Tensor& dL_dout_cells,
	const torch::Tensor& dL_dout_cell_weights,
    const bool use_gaussian_bvh,
	const bool debug
) {
    return RasterizeGaussiansBackwardCUDA(
        means3D,
        scales,
        rotations,
        conics,
        values,
        weights,
        out_cells,
        out_weights,
        scale_modifier,
        min_x, min_y, min_z, 
        max_x, max_y, max_z,
        background,
        dL_dout_cells,
        dL_dout_cell_weights,
        use_gaussian_bvh,
        debug,
        stored_samples,
        samples_bvh,
        gaussian_bvh
    );
}


// The current sample BVH's primIDs (leaf order: leaf k covers primIDs[offset_k ...]), as an
// int32 CUDA tensor. train.py permutes each batch into this order once at startup, so the
// render kernels' per-sample loads become nearly sequential (ncu: the sample gather was
// ~30% of both render kernels' stalls, with 1.5e8 excess sectors).
torch::Tensor SampleBVHPrimIDs() {
    const int N = samples_bvh.numPrims;
    auto t = torch::empty({N}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    if (N > 0) cudaMemcpyAsync(t.data_ptr<int>(), samples_bvh.primIDs, sizeof(uint32_t) * N, cudaMemcpyDeviceToDevice);
    return t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_bvh", &BuildBVH, "Build BVH from samples");
    m.def("sample_bvh_prim_ids", &SampleBVHPrimIDs, "primIDs of the current sample BVH");
    m.def("build_bvh_cached", &BuildBVHCached, "Build (once) and select the sample BVH for a batch key");
    m.def("rasterize_gaussians", &RasterizeGaussiansCUDAWrapper, "Forward pass");
    m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDAWrapper, "Backward pass");
    m.def("multi_adam_", &multi_adam_);
    m.def("multi_adam_ema_", &multi_adam_ema_);
    m.def("multi_adam_ex_", &multi_adam_ex_);
    m.def("margin_set_exterior", &margin_set_exterior);
    m.def("margin_fp", &margin_fp);
    m.def("fused_act_forward", &fused_act_forward);
    m.def("fused_act_backward", &fused_act_backward);
    m.def("fused_loss_forward", &fused_loss_forward);
    m.def("fused_loss_backward", &fused_loss_backward);
}
