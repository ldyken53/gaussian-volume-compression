// The support-margin FP term (--fp_mode margin): F = the sum over exterior points s and the
// Gaussians i whose support reaches s of qmax_i - q_i(s), with qmax = 2 ln(w / T). It is zero
// exactly when no support reaches the exterior, and its gradient does not shrink with the
// Gaussian's weight. Per Gaussian: dF/dw = 2 n / w, dF/dmu = 2 C sum d, dF/dc_jk = -sum d_j d_k
// (doubled off the diagonal, the parameter convention of the 6-vector conic).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <unordered_map>
#include <cuBQL/bvh.h>
#include <cuBQL/traversal/fixedBoxQuery.h>
#include "bvh.h"
#include "points.h"
namespace cg = cooperative_groups;

namespace
{

struct ExteriorSet {
	cuBQL::bvh3f bvh;
	torch::Tensor points;
};
std::unordered_map<int64_t, ExteriorSet> exterior_sets;

__global__ void marginKernel(int P, const float* __restrict__ means,
	const float* __restrict__ conic, const float* __restrict__ weights, float T,
	const float* __restrict__ samples, const cuBQL::bvh3f bvh, float* __restrict__ val,
	float* __restrict__ gmu, float* __restrict__ gc, float* __restrict__ gw)
{
	auto warp = cg::tiled_partition<32>(cg::this_thread_block());
	const int idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
	const int lane = warp.thread_rank();
	if (idx >= P)
		return;
	const float w = weights[idx];
	const float qmax = 2.0f * logf(w / T);
	if (!(qmax > 0.0f)) { // no support at all (w < T): contributes nothing
		if (lane == 0) {
			val[idx] = 0.0f;
			gw[idx] = 0.0f;
			for (int k = 0; k < 3; k++)
				gmu[3 * idx + k] = 0.0f;
			for (int k = 0; k < 6; k++)
				gc[6 * idx + k] = 0.0f;
		}
		return;
	}
	const float px = means[3 * idx], py = means[3 * idx + 1], pz = means[3 * idx + 2];
	const float a = conic[6 * idx], b = conic[6 * idx + 1], c = conic[6 * idx + 2];
	const float d = conic[6 * idx + 3], e = conic[6 * idx + 4], f = conic[6 * idx + 5];
	// Support box: half-width m sqrt(Sigma_kk), the diagonal of the conic's inverse.
	const float det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
	const float m = sqrtf(qmax);
	const float hx = m * sqrtf(fmaxf((d * f - e * e) / det, 0.0f));
	const float hy = m * sqrtf(fmaxf((a * f - c * c) / det, 0.0f));
	const float hz = m * sqrtf(fmaxf((a * d - b * b) / det, 0.0f));
	const cuBQL::box3f box(
		cuBQL::vec3f(px - hx, py - hy, pz - hz), cuBQL::vec3f(px + hx, py + hy, pz + hz));

	float s_m = 0.0f, n = 0.0f, Gx = 0.0f, Gy = 0.0f, Gz = 0.0f;
	float xx = 0.0f, xy = 0.0f, xz = 0.0f, yy = 0.0f, yz = 0.0f, zz = 0.0f;
	cuBQL::fixedBoxQuery::forEachLeaf<float, 3>(
		[&](uint32_t* primIDs, int count) {
			for (int i = lane; i < count; i += 32) {
				const uint32_t s = primIDs[i];
				const float dx = samples[3 * s] - px, dy = samples[3 * s + 1] - py,
							dz = samples[3 * s + 2] - pz;
				const float q = a * dx * dx + d * dy * dy + f * dz * dz +
								2.0f * (b * dx * dy + c * dx * dz + e * dy * dz);
				if (!(q >= 0.0f && q <= qmax))
					continue;
				s_m += qmax - q;
				n += 1.0f;
				Gx += dx;
				Gy += dy;
				Gz += dz;
				xx += dx * dx;
				xy += dx * dy;
				xz += dx * dz;
				yy += dy * dy;
				yz += dy * dz;
				zz += dz * dz;
			}
			return 0;
		},
		bvh, box);
	s_m = cg::reduce(warp, s_m, cg::plus<float>());
	n = cg::reduce(warp, n, cg::plus<float>());
	Gx = cg::reduce(warp, Gx, cg::plus<float>());
	Gy = cg::reduce(warp, Gy, cg::plus<float>());
	Gz = cg::reduce(warp, Gz, cg::plus<float>());
	xx = cg::reduce(warp, xx, cg::plus<float>());
	xy = cg::reduce(warp, xy, cg::plus<float>());
	xz = cg::reduce(warp, xz, cg::plus<float>());
	yy = cg::reduce(warp, yy, cg::plus<float>());
	yz = cg::reduce(warp, yz, cg::plus<float>());
	zz = cg::reduce(warp, zz, cg::plus<float>());
	if (lane == 0) {
		val[idx] = s_m;
		gw[idx] = 2.0f * n / w;
		// d(-q)/dmu = 2 C (s - mu), summed over the pairs.
		gmu[3 * idx] = 2.0f * (a * Gx + b * Gy + c * Gz);
		gmu[3 * idx + 1] = 2.0f * (b * Gx + d * Gy + e * Gz);
		gmu[3 * idx + 2] = 2.0f * (c * Gx + e * Gy + f * Gz);
		gc[6 * idx] = -xx;
		gc[6 * idx + 1] = -2.0f * xy;
		gc[6 * idx + 2] = -2.0f * xz;
		gc[6 * idx + 3] = -yy;
		gc[6 * idx + 4] = -2.0f * yz;
		gc[6 * idx + 5] = -zz;
	}
}

} // namespace

// Registers exterior set key, building its BVH the first time.
void marginSetExterior(torch::Tensor points, int64_t key)
{
	if (exterior_sets.count(key))
		return;
	TORCH_CHECK(points.is_cuda() && points.scalar_type() == torch::kFloat32,
		"exterior points: float32 CUDA");
	auto xyz = points.contiguous();
	const int n = xyz.size(0);
	ExteriorSet set;
	set.points = xyz;
	if (n > 0)
		buildPointBVH(xyz.data_ptr<float>(), n, set.bvh);
	exterior_sets[key] = set;
}

// (F per Gaussian, dF/dmeans, dF/dconics, dF/dweights) against exterior set key.
std::vector<torch::Tensor> marginFP(
	torch::Tensor means, torch::Tensor conics, torch::Tensor weights, double T, int64_t key)
{
	auto it = exterior_sets.find(key);
	TORCH_CHECK(it != exterior_sets.end(), "margin_fp: exterior set not registered");
	const int P = means.size(0);
	auto o = means.options();
	auto val = torch::zeros({P}, o), gmu = torch::zeros({P, 3}, o), gc = torch::zeros({P, 6}, o),
		 gw = torch::zeros({P}, o);
	if (P > 0 && it->second.points.size(0) > 0) {
		auto mc = means.contiguous(), cc = conics.contiguous(), wc = weights.contiguous();
		marginKernel<<<(P * 32 + 255) / 256, 256>>>(P, mc.data_ptr<float>(), cc.data_ptr<float>(),
			wc.data_ptr<float>(), (float)T, it->second.points.data_ptr<float>(), it->second.bvh,
			val.data_ptr<float>(), gmu.data_ptr<float>(), gc.data_ptr<float>(),
			gw.data_ptr<float>());
		C10_CUDA_KERNEL_LAUNCH_CHECK();
	}
	return {val, gmu, gc, gw};
}
