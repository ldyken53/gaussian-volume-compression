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

	auto scale = scales[idx];
	auto rot = rotations[idx];
	
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale_modifier * scale.x;
	S[1][1] = scale_modifier * scale.y;
	S[2][2] = scale_modifier * scale.z;

	// Normalize quaternion to get valid rotation (commented out for some reason?)
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Normalize by epsilon to prevent numerical issues
	const float epsilon = max(max(abs(Sigma[0][0]), abs(Sigma[1][1])), abs(Sigma[2][2])) * 1e-5;
	const float cov[6] = {
		Sigma[0][0] + epsilon,
        Sigma[0][1],
        Sigma[0][2],
		Sigma[1][1] + epsilon,
        Sigma[1][2],
        Sigma[2][2] + epsilon,
	};

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

	// Scale S by 3 to include up to where the weight is a tenth the cutoff
	float m = sqrtf(-2 * logf((0.1 * WEIGHT_CUTOFF) / weights[idx]));
	const float3 scaled_S = { S[0][0] * m, S[1][1] * m, S[2][2] * m };

 	// Create array for corner computations
    const float n[2] = {-1.0f, 1.0f};
    
    // Initialize mins and maxes with gaussian position
	const float3 position = { means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2] };
    float3 mins = position;
    float3 maxes = position;

	// Compute corners using vector operations
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                float3 corner = make_float3(
					position.x + n[i] * R[0].x * scaled_S.x + n[j] * R[1].x * scaled_S.y +  n[k] * R[2].x * scaled_S.z,
					position.y + n[i] * R[0].y * scaled_S.x + n[j] * R[1].y * scaled_S.y +  n[k] * R[2].y * scaled_S.z,
					position.z + n[i] * R[0].z * scaled_S.x + n[j] * R[1].z * scaled_S.y +  n[k] * R[2].z * scaled_S.z
				);
                    
                mins = make_float3(
					min(mins.x, corner.x),
					min(mins.y, corner.y),
					min(mins.z, corner.z)
				);
                maxes = make_float3(
					max(maxes.x, corner.x),
					max(maxes.y, corner.y),
					max(maxes.z, corner.z)
				);
            }
        }
    }
	aabbs[idx] = cuBQL::box3f(cuBQL::vec3f(mins.x, mins.y, mins.z), cuBQL::vec3f(maxes.x, maxes.y, maxes.z));	
}


// Perform initial steps for each Gaussian prior to rasterization.
__global__ void intersectPreprocessCUDA(const int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	float* conics,
	cuBQL::box3f* aabbs)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	auto scale = scales[idx];
	auto rot = rotations[idx];
	
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale_modifier * scale.x;
	S[1][1] = scale_modifier * scale.y;
	S[2][2] = scale_modifier * scale.z;

	// Normalize quaternion to get valid rotation (commented out for some reason?)
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Normalize by epsilon to prevent numerical issues
	const float epsilon = max(max(abs(Sigma[0][0]), abs(Sigma[1][1])), abs(Sigma[2][2])) * 1e-5;
	const float cov[6] = {
		Sigma[0][0] + epsilon,
        Sigma[0][1],
        Sigma[0][2],
		Sigma[1][1] + epsilon,
        Sigma[1][2],
        Sigma[2][2] + epsilon,
	};

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

	// Scale S by 3 to include up to where the weight is a tenth the cutoff
	float m = 1.0;
	const float3 scaled_S = { S[0][0] * m, S[1][1] * m, S[2][2] * m };

 	// Create array for corner computations
    const float n[2] = {-1.0f, 1.0f};
    
    // Initialize mins and maxes with gaussian position
	const float3 position = { means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2] };
    float3 mins = position;
    float3 maxes = position;

	// Compute corners using vector operations
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                float3 corner = make_float3(
					position.x + n[i] * R[0].x * scaled_S.x + n[j] * R[1].x * scaled_S.y +  n[k] * R[2].x * scaled_S.z,
					position.y + n[i] * R[0].y * scaled_S.x + n[j] * R[1].y * scaled_S.y +  n[k] * R[2].y * scaled_S.z,
					position.z + n[i] * R[0].z * scaled_S.x + n[j] * R[1].z * scaled_S.y +  n[k] * R[2].z * scaled_S.z
				);
                    
                mins = make_float3(
					min(mins.x, corner.x),
					min(mins.y, corner.y),
					min(mins.z, corner.z)
				);
                maxes = make_float3(
					max(maxes.x, corner.x),
					max(maxes.y, corner.y),
					max(maxes.z, corner.z)
				);
            }
        }
    }
	aabbs[idx] = cuBQL::box3f(cuBQL::vec3f(mins.x, mins.y, mins.z), cuBQL::vec3f(maxes.x, maxes.y, maxes.z));	
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
		// for (int i = 0; i < count2; i++) {
		// 	uint primID = primIDs[i]; 
		// 	float3 d = make_float3(
		// 		samples[primID * 3] - position.x, 
		// 		samples[primID * 3 + 1] - position.y, 
		// 		samples[primID * 3 + 2] - position.z
		// 	);
		// 	float quad_form = (
		// 		d.x * (conic[0] * d.x + conic[1] * d.y + conic[2] * d.z) +
		// 		d.y * (conic[1] * d.x + conic[3] * d.y + conic[4] * d.z) +
		// 		d.z * (conic[2] * d.x + conic[4] * d.y + conic[5] * d.z)
		// 	);
		// 	float power = -0.5 * quad_form;
		// 	if (power < -14.0 || power > 0.0) continue;
		// 	float weight = weights[idx] * exp(power);
		// 	atomicAdd(&out_testw[primID], weight);
		// 	atomicAdd(&out_test[primID], weight * values[idx]);
		// }
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
			if (power < -14.0 || power > 0.0) continue;
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

// Perform initial steps for each Gaussian prior to rasterization.
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
	const int THREADS_PER_SAMPLE = 32; // One warp per sample
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
	cuBQL::fixedBoxQuery::forEachLeaf<float,3>(
	[&](uint32_t* primIDs, int prim_count) {
		for (int i = thread_in_warp; i < prim_count; i += THREADS_PER_SAMPLE) {
			int primID = primIDs[i];
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
			if (power < -14.0 || power > 0.0) continue;
			float weight = weights[primID] * exp(power);
			acc_weight += weight;
			acc_value += weight * values[primID];
			count++;
		}
		return 0;
    },
		bvh,
		cuBQL::box3f(cuBQL::vec3f(sample.x, sample.y, sample.z))
	);
	acc_weight = cg::reduce(warp, acc_weight, cg::plus<float>());
	acc_value = cg::reduce(warp, acc_value, cg::plus<float>());
	if (thread_in_warp == 0) {
		if (acc_weight <= WEIGHT_CUTOFF) {
			out_test[idx] = -1.0;
			out_testw[idx] = 0.0;
		} else {
			out_test[idx] = acc_value / acc_weight;
			out_testw[idx] = acc_weight;
		}
		count_intersections[idx] = count;
	}
}

__device__ float computeGaussianOverlap(
    const float3& pos1, const float conic1[6],
    const float3& pos2, const float conic2[6])
{
    // Precision matrices stored as upper triangular:
    // [0]=xx, [1]=xy, [2]=xz, [3]=yy, [4]=yz, [5]=zz
    
    // P_sum = P1 + P2
    float Ps[6];
    for (int i = 0; i < 6; i++) {
        Ps[i] = conic1[i] + conic2[i];
    }
    
    // Determinant of P_sum (3x3 symmetric matrix)
    // | a  b  c |
    // | b  d  e |  where a=Ps[0], b=Ps[1], c=Ps[2], d=Ps[3], e=Ps[4], f=Ps[5]
    // | c  e  f |
    float a = Ps[0], b = Ps[1], c = Ps[2], d = Ps[3], e = Ps[4], f = Ps[5];
    float det_Ps = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c);
    
    if (det_Ps <= 1e-10f) return 0.0f;  // Degenerate case
    
    // Inverse of P_sum using adjugate method: inv = adj(Ps) / det
    float inv_det = 1.0f / det_Ps;
    float Ps_inv[6];
    Ps_inv[0] = (d * f - e * e) * inv_det;       // inv_xx
    Ps_inv[1] = (c * e - b * f) * inv_det;       // inv_xy
    Ps_inv[2] = (b * e - c * d) * inv_det;       // inv_xz
    Ps_inv[3] = (a * f - c * c) * inv_det;       // inv_yy
    Ps_inv[4] = (b * c - a * e) * inv_det;       // inv_yz
    Ps_inv[5] = (a * d - b * b) * inv_det;       // inv_zz
    
    // d = pos1 - pos2
    float dx = pos1.x - pos2.x;
    float dy = pos1.y - pos2.y;
    float dz = pos1.z - pos2.z;
    
    // Compute P1 * d (symmetric matrix-vector multiply)
    float P1d_x = conic1[0] * dx + conic1[1] * dy + conic1[2] * dz;
    float P1d_y = conic1[1] * dx + conic1[3] * dy + conic1[4] * dz;
    float P1d_z = conic1[2] * dx + conic1[4] * dy + conic1[5] * dz;
    
    // Compute Ps_inv * (P1 * d)
    float tmp_x = Ps_inv[0] * P1d_x + Ps_inv[1] * P1d_y + Ps_inv[2] * P1d_z;
    float tmp_y = Ps_inv[1] * P1d_x + Ps_inv[3] * P1d_y + Ps_inv[4] * P1d_z;
    float tmp_z = Ps_inv[2] * P1d_x + Ps_inv[4] * P1d_y + Ps_inv[5] * P1d_z;
    
    // Compute P2 * (Ps_inv * P1 * d)
    float P2tmp_x = conic2[0] * tmp_x + conic2[1] * tmp_y + conic2[2] * tmp_z;
    float P2tmp_y = conic2[1] * tmp_x + conic2[3] * tmp_y + conic2[4] * tmp_z;
    float P2tmp_z = conic2[2] * tmp_x + conic2[4] * tmp_y + conic2[5] * tmp_z;
    
    // Quadratic form: d^T * P1 * Ps_inv * P2 * d
    float mahal = dx * P2tmp_x + dy * P2tmp_y + dz * P2tmp_z;
    
    // Overlap = (2*pi)^(3/2) / sqrt(det_Ps) * exp(-0.5 * mahal)
    const float CONST_2PI_3_2 = 15.7496099f;  // (2*pi)^(3/2)
    float overlap = CONST_2PI_3_2 * rsqrtf(det_Ps) * expf(-0.5f * mahal);
    
    return overlap;
}

__global__ void intersectCUDA(const int P,
	const float* means3D,
	const float* conics,
	const cuBQL::box3f* aabbs,
	const cuBQL::bvh3f bvh,
	float* intersections,
	float* intersection_weights)
{
	const int THREADS_PER_GAUSSIAN = 32; // One warp per Gaussian
	
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<THREADS_PER_GAUSSIAN>(block);
	
	int idx = blockIdx.x * (blockDim.x / THREADS_PER_GAUSSIAN) + (threadIdx.x / THREADS_PER_GAUSSIAN);
	int thread_in_warp = warp.thread_rank();

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

	float intersection_weight = 0.0;
	float intersection = 0.0;
	int count = 0;
	cuBQL::fixedBoxQuery::forEachLeaf<float,3>(
	[&](uint32_t* primIDs, int prim_count) {
		for (int i = thread_in_warp; i < prim_count; i += THREADS_PER_GAUSSIAN) {
			int primID = primIDs[i];
			const float3 position2 = { means3D[3 * primID], means3D[3 * primID + 1], means3D[3 * primID + 2] };
			const float conic2[6] = {
				conics[primID * 6 + 0],
				conics[primID * 6 + 1],
				conics[primID * 6 + 2],
				conics[primID * 6 + 3],
				conics[primID * 6 + 4],
				conics[primID * 6 + 5]
			};
			// Skip self-intersection
			if (primID != idx) {
				float w = computeGaussianOverlap(position, conic, position2, conic2);
				intersection_weight += w;
				intersection += 1.0;
			}
		}
		return 0;
    },
		bvh,
		aabbs[idx]
	);
	intersection = cg::reduce(warp, intersection, cg::plus<float>());
	intersection_weight = cg::reduce(warp, intersection_weight, cg::plus<float>());
	if (thread_in_warp == 0) {
		intersection_weights[idx] = intersection_weight;
		intersections[idx] = intersection;
	}
}

__global__ void normalizeCUDA(const int S,
	float* out_test,
	float* out_testw)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= S)
		return;
	if (out_testw[idx] <= WEIGHT_CUTOFF) {
		out_test[idx] = -1.0;
		out_testw[idx] = 0.0;
	} else {
		out_test[idx] = out_test[idx] / out_testw[idx];
	}
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

void FORWARD::intersect_preprocess(const int P,
		const float* means3D,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		float* conics,
		cuBQL::box3f* aabbs)
	{
		intersectPreprocessCUDA <<<(P + 255) / 256, 256>>> (
			P,
			means3D,
			scales,
			scale_modifier,
			rotations,
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
			dim3 grid((S * 32 + block.x - 1) / block.x); // 1 threads per sample
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

void FORWARD::intersect(const int P,
	const float* means3D,
	const float* conics,
	const cuBQL::box3f* aabbs,
	const cuBQL::bvh3f& bvh,
	float* intersections,
	float* intersection_weights)
	{
		dim3 block(256);
		dim3 grid((P * 32 + block.x - 1) / block.x); // 32 threads per Gaussian
		intersectCUDA <<<grid, block>>> (
			P,
			means3D,
			conics,
			aabbs,
			bvh,
			intersections,
			intersection_weights
		);
	}
