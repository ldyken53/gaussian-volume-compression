#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cuBQL/traversal/fixedBoxQuery.h>
namespace cg = cooperative_groups;

// (x, y, z, 0) sample positions of the current batch, written by the BVH build (ext.cu).
const float4* gvSamplePos();

__global__ void preprocessCUDA(int P, const float* means3D, const glm::vec3* scales,
	const glm::vec4* rotations, const float* weights, float* conics, cuBQL::box3f* aabbs)
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

	const float m = truncationRadius(weights[idx]);
	const float3 position = {means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2]};
	aabbs[idx] = tightGaussianBox(position, cov, m);
}

// One warp per Gaussian; lanes split each overlapped leaf's samples.
__global__ void renderCUDA(int P, const float* means3D, const float* values, const float* weights,
	const float4* __restrict__ sample_pos, const float* conics, const cuBQL::box3f* aabbs,
	const cuBQL::bvh3f bvh, float* out_test, float* out_testw)
{
	auto warp = cg::tiled_partition<32>(cg::this_thread_block());
	int idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
	int lane = warp.thread_rank();
	if (idx >= P)
		return;
	const float3 position = {means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2]};
	const float conic[6] = {conics[idx * 6 + 0], conics[idx * 6 + 1], conics[idx * 6 + 2],
		conics[idx * 6 + 3], conics[idx * 6 + 4], conics[idx * 6 + 5]};

	// Held in registers (the atomics may alias the inputs); off-diagonals pre-doubled.
	const float w = weights[idx], v = values[idx];
	const float qmax = supportQmax(w);
	const float cxx = conic[0], c2xy = 2.0f * conic[1], c2xz = 2.0f * conic[2];
	const float cyy = conic[3], c2yz = 2.0f * conic[4], czz = conic[5];

	cuBQL::fixedBoxQuery::forEachLeaf<float, 3>(
		[&](uint32_t* primIDs, int count) {
			for (int i = lane; i < count; i += 32) {
				uint primID = primIDs[i];
				const float4 sp = sample_pos[primID];
				const float dx = sp.x - position.x;
				const float dy = sp.y - position.y;
				const float dz = sp.z - position.z;
				const float t0 = fmaf(c2xz, dz, fmaf(c2xy, dy, cxx * dx));
				const float t1 = fmaf(c2yz, dz, cyy * dy);
				const float q = fmaf(dx, t0, fmaf(dy, t1, czz * dz * dz));
				// Also rejects NaN.
				if (!(q >= 0.0f && q <= qmax))
					continue;
				const float weight = w * expf(-0.5f * q);
				atomicAdd(&out_testw[primID], weight);
				atomicAdd(&out_test[primID], weight * v);
			}
			return 0;
		},
		bvh, aabbs[idx]);
}

// Empty samples keep their true aw, so the FN/FP terms still see them.
__global__ void normalizeCUDA(int S, float occ_t, float floor, float* out_test, const float* out_testw)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= S)
		return;
	const float aw = out_testw[idx];
	if (aw <= occ_t)
		out_test[idx] = -1.0;
	else
		out_test[idx] = out_test[idx] / max(aw, floor);
}

void FORWARD::preprocess(int P, const float* means3D, const glm::vec3* scales,
	const glm::vec4* rotations, const float* weights, float* conics, cuBQL::box3f* aabbs)
{
	preprocessCUDA<<<(P + 255) / 256, 256>>>(P, means3D, scales, rotations, weights, conics, aabbs);
}

void FORWARD::render(int P, int S, const float* means3D, const float* values, const float* weights,
	const float* conics, const cuBQL::box3f* aabbs, const cuBQL::bvh3f& bvh, float occ_t,
	float floor, float* out_test, float* out_testw)
{
	renderCUDA<<<(P * 32 + 255) / 256, 256>>>(
		P, means3D, values, weights, gvSamplePos(), conics, aabbs, bvh, out_test, out_testw);
	normalizeCUDA<<<(S + 255) / 256, 256>>>(S, occ_t, floor, out_test, out_testw);
}
