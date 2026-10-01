#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include <cuBQL/bvh.h>
#include <cuBQL/traversal/fixedBoxQuery.h>

// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(const int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* weights,
	float* conics,
	cuBQL::box3f* aabbs)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// 3D world covariance (with epsilon); shared with the backward pass's box
	float cov[6];
	computeCov3D(scales[idx], scale_modifier, rotations[idx], cov);

	// Use 3D covariance to compute and store 3D conic
	const float a = cov[0]; // Sigma[0][0]
    const float b = cov[1]; // Sigma[0][1]
    const float c = cov[2]; // Sigma[0][2]
    const float d = cov[3]; // Sigma[1][1]
    const float e = cov[4]; // Sigma[1][2]
    const float f = cov[5]; // Sigma[2][2]
    const float det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    const float det_inv = 1.0 / det;
	conics[idx * 6 + 0] = (d * f - e * e) * det_inv;
	conics[idx * 6 + 1] = (c * e - b * f) * det_inv;
	conics[idx * 6 + 2] = (b * e - c * d) * det_inv;
	conics[idx * 6 + 3] = (a * f - c * c) * det_inv;
	conics[idx * 6 + 4] = (b * c - a * e) * det_inv;
	conics[idx * 6 + 5] = (a * d - b * b) * det_inv;

	const float m = truncationRadius(weights[idx]);
	const float3 position = { means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2] };
	aabbs[idx] = tightGaussianBox(position, cov, m);
}

__global__ void renderCUDA(const int P,
	const float* means3D,
	const float* values,
	const float* weights,
	const float* samples,
	const float* conics,
	const cuBQL::box3f* aabbs,
	const cuBQL::bvh3f bvh,
	float* out_test,
	float* out_testw,
	int* count_intersections)
{
	const int THREADS_PER_GAUSSIAN = 32; // One warp per Gaussian
	
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<THREADS_PER_GAUSSIAN>(block);
	
	int idx = blockIdx.x * (blockDim.x / THREADS_PER_GAUSSIAN) + (threadIdx.x / THREADS_PER_GAUSSIAN);
	int thread_in_warp = warp.thread_rank();

	// auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	const float3 position = { means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2] };
	const float conic[6] = {
		conics[idx * 6 + 0],
		conics[idx * 6 + 1],
		conics[idx * 6 + 2],
		conics[idx * 6 + 3],
		conics[idx * 6 + 4],
		conics[idx * 6 + 5]
	};

	int count = 0;
	cuBQL::fixedBoxQuery::forEachLeaf<float,3>(
	[&](uint32_t* primIDs, int count2) {
		// Distribute primitives across threads in the warp
		for (int i = thread_in_warp; i < count2; i += THREADS_PER_GAUSSIAN) {
			uint primID = primIDs[i]; 
			float3 d = make_float3(
				samples[primID * 3] - position.x, 
				samples[primID * 3 + 1] - position.y, 
				samples[primID * 3 + 2] - position.z
			);
			float quad_form = (
				d.x * (conic[0] * d.x + conic[1] * d.y + conic[2] * d.z) +
				d.y * (conic[1] * d.x + conic[3] * d.y + conic[4] * d.z) +
				d.z * (conic[2] * d.x + conic[4] * d.y + conic[5] * d.z)
			);
			float power = -0.5 * quad_form;
			if (power < -14.0 || power > 0.0) {continue;};
			float weight = weights[idx] * exp(power);
			atomicAdd(&out_testw[primID], weight);
			atomicAdd(&out_test[primID], weight * values[idx]);
			count++;
		}
		return 0;
    },
		bvh,
		aabbs[idx]
	);
	// Only one thread per warp writes the final count
	count = cg::reduce(warp, count, cg::plus<int>());
	if (thread_in_warp == 0) {
		count_intersections[idx] = count;
	}
}


__global__ void sampleRenderCUDA(const int S,
	const float* means3D,
	const float* values,
	const float* weights,
	const float* samples,
	const float* conics,
	const cuBQL::bvh3f bvh,
	float* out_test,
	float* out_testw,
	int* count_intersections)
{
	const int THREADS_PER_SAMPLE = 1; // One warp per sample
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<THREADS_PER_SAMPLE>(block);
	int idx = blockIdx.x * (blockDim.x / THREADS_PER_SAMPLE) + (threadIdx.x / THREADS_PER_SAMPLE);
	int thread_in_warp = warp.thread_rank();
	// auto idx = cg::this_grid().thread_rank();
	if (idx >= S)
		return;
	const float3 sample = { samples[3 * idx], samples[3 * idx + 1], samples[3 * idx + 2] };
	float acc_weight = 0.0;
	float acc_value = 0.0;

	int count = 0;
	cuBQL::fixedBoxQuery::forEachPrim<float,3>(
	[&](int primID) {
	// [&](uint32_t* primIDs, int prim_count) {
	// 	count+=prim_count;
		// for (int i = thread_in_warp; i < prim_count; i += THREADS_PER_SAMPLE) {
		// 	int primID = primIDs[i];
			const float3 position = { means3D[3 * primID], means3D[3 * primID + 1], means3D[3 * primID + 2] };
			const float conic[6] = {
				conics[primID * 6 + 0],
				conics[primID * 6 + 1],
				conics[primID * 6 + 2],
				conics[primID * 6 + 3],
				conics[primID * 6 + 4],
				conics[primID * 6 + 5]
			};
			float3 d = make_float3(
				sample.x - position.x, 
				sample.y - position.y, 
				sample.z - position.z
			);
			float quad_form = (
				d.x * (conic[0] * d.x + conic[1] * d.y + conic[2] * d.z) +
				d.y * (conic[1] * d.x + conic[3] * d.y + conic[4] * d.z) +
				d.z * (conic[2] * d.x + conic[4] * d.y + conic[5] * d.z)
			);
			float power = -0.5 * quad_form;
			if (power < -14.0 || power > 0.0) {return 0;};
			float weight = weights[primID] * exp(power);
			acc_weight += weight;
			acc_value += weight * values[primID];
			count++;
		// }
		return 0;
    },
		bvh,
		cuBQL::box3f(cuBQL::vec3f(sample.x, sample.y, sample.z))
	);
	acc_weight = cg::reduce(warp, acc_weight, cg::plus<float>());
	acc_value = cg::reduce(warp, acc_value, cg::plus<float>());
	count = cg::reduce(warp, count, cg::plus<int>());
	if (thread_in_warp == 0) {
#if SOFT_CUTOFF
		if (acc_weight <= 0.0f) {
			out_test[idx] = -1.0;
			out_testw[idx] = 0.0;
		} else {
			// Clamped denominator: weak cells render a small bounded value and keep
			// their true accumulated weight (feeds the FN mask) and their gradient.
			out_test[idx] = acc_value / SOFT_DENOM(acc_weight);
			out_testw[idx] = acc_weight;
		}
#else
		if (acc_weight <= WEIGHT_CUTOFF) {
			out_test[idx] = -1.0;
			out_testw[idx] = 0.0;
		} else {
			out_test[idx] = acc_value / acc_weight;
			out_testw[idx] = acc_weight;
		}
#endif
		count_intersections[idx] = count;
	}
}

__global__ void normalizeCUDA(const int S,
	float* out_test,
	float* out_testw)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= S)
		return;
#if SOFT_CUTOFF
	if (out_testw[idx] <= 0.0f) {
		out_test[idx] = -1.0;
		out_testw[idx] = 0.0;
	} else {
		out_test[idx] = out_test[idx] / SOFT_DENOM(out_testw[idx]);
	}
#else
	if (out_testw[idx] <= WEIGHT_CUTOFF) {
		out_test[idx] = -1.0;
		out_testw[idx] = 0.0;
	} else {
		out_test[idx] = out_test[idx] / out_testw[idx];
	}
#endif
}

void FORWARD::preprocess(const int P,
		const float* means3D,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* weights,
		float* conics,
		cuBQL::box3f* aabbs)
	{
		preprocessCUDA <<<(P + 255) / 256, 256>>> (
			P,
			means3D,
			scales,
			scale_modifier,
			rotations,
			weights,
			conics,
			aabbs
		);
	}

void FORWARD::render(const int P, const int S,
	const float* means3D,
	const float* values,
	const float* weights,
	const float* samples,
	const float* conics,
	const cuBQL::box3f* aabbs,
	const cuBQL::bvh3f& bvh,
	float* out_test,
	float* out_testw,
	int* count_intersections,
	const bool use_gaussian_bvh)
	{
		if (use_gaussian_bvh) {
			dim3 block(256);
			dim3 grid((S + block.x - 1) / block.x); // 1 threads per sample
			sampleRenderCUDA<<<grid, block>>> (
				S,
				means3D,
				values,
				weights,
				samples,
				conics,
				bvh,
				out_test,
				out_testw,
				count_intersections
			);
		} else {
			dim3 block(256);
			dim3 grid((P * 32 + block.x - 1) / block.x); // 32 threads per Gaussian
			renderCUDA <<<grid, block>>> (
				P,
				means3D,
				values,
				weights,
				samples,
				conics,
				aabbs,
				bvh,
				out_test,
				out_testw,
				count_intersections
			);

			normalizeCUDA <<<(S + 255) / 256, 256>>> (
				S,
				out_test,
				out_testw
			);
		}
	}
