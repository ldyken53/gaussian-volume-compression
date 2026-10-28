#include "forward.h"
#include "gaussian.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuBQL/traversal/fixedBoxQuery.h>
namespace cg = cooperative_groups;

__global__ void prepareCUDA(int P, const float* means, const glm::vec3* scales,
	const glm::vec4* rotations, const float* weights, float* conics, cuBQL::box3f* boxes)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float cov[6];
	computeCov3D(scales[idx], rotations[idx], cov);

	const float a = cov[0];
	const float b = cov[1];
	const float c = cov[2];
	const float d = cov[3];
	const float e = cov[4];
	const float f = cov[5];
	const float det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
	const float det_inv = 1.0 / det;
	conics[idx * 6 + 0] = (d * f - e * e) * det_inv;
	conics[idx * 6 + 1] = (c * e - b * f) * det_inv;
	conics[idx * 6 + 2] = (b * e - c * d) * det_inv;
	conics[idx * 6 + 3] = (a * f - c * c) * det_inv;
	conics[idx * 6 + 4] = (b * c - a * e) * det_inv;
	conics[idx * 6 + 5] = (a * d - b * b) * det_inv;

	const float m = supportRadius(weights[idx]);
	const float3 position = {means[3 * idx], means[3 * idx + 1], means[3 * idx + 2]};
	boxes[idx] = supportBox(position, cov, m);
}

// One warp per Gaussian; the lanes split each overlapped leaf's points.
__global__ void sampleCUDA(int P, const float* means, const float* values, const float* weights,
	const float4* __restrict__ points, const float* conics, const cuBQL::box3f* boxes,
	const cuBQL::bvh3f bvh, float* V, float* I, float* __restrict__ reach)
{
	auto warp = cg::tiled_partition<32>(cg::this_thread_block());
	int idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
	int lane = warp.thread_rank();
	if (idx >= P)
		return;
	const float3 position = {means[3 * idx], means[3 * idx + 1], means[3 * idx + 2]};
	const float conic[6] = {conics[idx * 6 + 0], conics[idx * 6 + 1], conics[idx * 6 + 2],
		conics[idx * 6 + 3], conics[idx * 6 + 4], conics[idx * 6 + 5]};

	// Held in registers (the atomics may alias the inputs); off-diagonals doubled.
	const float w = weights[idx], v = values[idx];
	const float qmax = supportQmax(w);
	const float cxx = conic[0], c2xy = 2.0f * conic[1], c2xz = 2.0f * conic[2];
	const float cyy = conic[3], c2yz = 2.0f * conic[4], czz = conic[5];
	float wmax = 0.0f; // the largest w G at a flagged (inside) point, when reach is recorded

	cuBQL::fixedBoxQuery::forEachLeaf<float, 3>(
		[&](uint32_t* primIDs, int count) {
			for (int i = lane; i < count; i += 32) {
				uint primID = primIDs[i];
				const float4 p = points[primID];
				const float dx = p.x - position.x;
				const float dy = p.y - position.y;
				const float dz = p.z - position.z;
				const float t0 = fmaf(c2xz, dz, fmaf(c2xy, dy, cxx * dx));
				const float t1 = fmaf(c2yz, dz, cyy * dy);
				const float q = fmaf(dx, t0, fmaf(dy, t1, czz * dz * dz));
				// Also rejects NaN.
				if (!(q >= 0.0f && q <= qmax))
					continue;
				const float weight = w * expf(-0.5f * q);
				atomicAdd(&I[primID], weight);
				atomicAdd(&V[primID], weight * v);
				if (reach != nullptr && p.w > 0.0f)
					wmax = fmaxf(wmax, weight);
			}
			return 0;
		},
		bvh, boxes[idx]);
	if (reach != nullptr) {
		wmax = cg::reduce(warp, wmax, cg::greater<float>());
		if (lane == 0)
			reach[idx] = wmax;
	}
}

// Undefined points keep their influence for the FN and FP terms.
__global__ void applyRuleCUDA(int S, float threshold, float floor, float* V, const float* I)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= S)
		return;
	const float influence = I[idx];
	if (influence <= threshold)
		V[idx] = -1.0;
	else
		V[idx] = V[idx] / max(influence, floor);
}

void FORWARD::prepare(int P, const float* means, const glm::vec3* scales,
	const glm::vec4* rotations, const float* weights, float* conics, cuBQL::box3f* boxes)
{
	prepareCUDA<<<(P + 255) / 256, 256>>>(P, means, scales, rotations, weights, conics, boxes);
}

void FORWARD::sample(int P, int S, const float* means, const float* values, const float* weights,
	const float* conics, const cuBQL::box3f* boxes, const float4* points, const cuBQL::bvh3f& bvh,
	float threshold, float floor, float* out_values, float* out_influence, float* reach)
{
	sampleCUDA<<<(P * 32 + 255) / 256, 256>>>(
		P, means, values, weights, points, conics, boxes, bvh, out_values, out_influence, reach);
	applyRuleCUDA<<<(S + 255) / 256, 256>>>(S, threshold, floor, out_values, out_influence);
}
