#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#include <cuBQL/bvh.h>
#include <cuBQL/traversal/fixedBoxQuery.h>

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* values,
	const float* weights,
	const float3 volume_mins,
	const float3 volume_maxes,
	const float* samples,
	const cuBQL::bvh3f bvh,
	const float* out_cells,
	const float* out_weights,
	const float* dL_dsamples,
	const float* dL_dsample_weights,
	float* dL_dmeans,
	float* dL_dvalues,
	float* dL_dweights,
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots)
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
	const float conic[6] = {
		(d * f - e * e) * det_inv,
		(c * e - b * f) * det_inv,
		(b * e - c * d) * det_inv,
		(a * f - c * c) * det_inv,
		(b * c - a * e) * det_inv,
		(a * d - b * b) * det_inv
	};

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

	float dL_dvalue = 0.0;
	float dL_dw = 0.0;
	float dL_dmean_x = 0.0;
	float dL_dmean_y = 0.0;
	float dL_dmean_z = 0.0;
	float dL_dxx = 0.0;
	float dL_dxy = 0.0;
	float dL_dxz = 0.0;
	float dL_dyy = 0.0;
	float dL_dyz = 0.0;
	float dL_dzz = 0.0;
	cuBQL::fixedBoxQuery::forEachPrim<float,3>(
	[&](int primID) {
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
		if (power < -14.0 || power > 0.0) return 0;

		float dL_doutv = dL_dsamples[primID];
		float dL_doutw = dL_dsample_weights[primID];
		float acc_weight = out_weights[primID];
		if (acc_weight <= WEIGHT_CUTOFF) return 0;

		float e = exp(power);
		float weight = weights[idx] * e;

		dL_dvalue += dL_doutv * weight / acc_weight;

		float dLv_dweight = dL_doutv * (values[idx] / acc_weight - out_cells[primID] / acc_weight);
		float dLv_dw = dLv_dweight * e;

		float dweight_dquad = -0.5f * weight;
		float dLv_dquad = dLv_dweight * dweight_dquad;

		float dLw_dw = dL_doutw * e;
		float dLw_dquad = dL_doutw * dweight_dquad;

		dL_dw += dLv_dw + dLw_dw;
		float dL_dquad = dLv_dquad + dLw_dquad;
		
		// Gradients for means
		dL_dmean_x += dL_dquad * 2 * -1 *
			(conic[0] * d.x + conic[1] * d.y + conic[2] * d.z);
		dL_dmean_y += dL_dquad * 2 * -1 *
			(conic[1] * d.x + conic[3] * d.y + conic[4] * d.z);
		dL_dmean_z += dL_dquad * 2 * -1 *
			(conic[2] * d.x + conic[4] * d.y + conic[5] * d.z);

		// Gradients for conic
		dL_dxx += dL_dquad * d.x * d.x;
		dL_dxy += dL_dquad * d.x * d.y;
		dL_dxz += dL_dquad * d.x * d.z;
		dL_dyy += dL_dquad * d.y * d.y;
		dL_dyz += dL_dquad * d.y * d.z;
		dL_dzz += dL_dquad * d.z * d.z;
		return 0;
    },
		bvh,
		cuBQL::box3f(cuBQL::vec3f(mins.x, mins.y, mins.z), cuBQL::vec3f(maxes.x, maxes.y, maxes.z))
	);

	dL_dvalues[idx] = dL_dvalue;
	dL_dweights[idx] = dL_dw;
	dL_dmeans[idx * 3] = dL_dmean_x;
	dL_dmeans[idx * 3 + 1] = dL_dmean_y;	
	dL_dmeans[idx * 3 + 2] = dL_dmean_z;

	float dL_dconic[6] = {
		dL_dxx,
		dL_dxy,
		dL_dxz,
		dL_dyy,
		dL_dyz,
		dL_dzz
	};
	// Compute dL_dcov as -conic * dL_dconic * conic
	// since conic is inverse of cov
	const float dL_dconic_conic[6] = {
		dL_dconic[0]*conic[0] + dL_dconic[1]*conic[1] + dL_dconic[2]*conic[2],
    	dL_dconic[0]*conic[1] + dL_dconic[1]*conic[3] + dL_dconic[2]*conic[4],
    	dL_dconic[0]*conic[2] + dL_dconic[1]*conic[4] + dL_dconic[2]*conic[5],
    	dL_dconic[1]*conic[1] + dL_dconic[3]*conic[3] + dL_dconic[4]*conic[4],
    	dL_dconic[1]*conic[2] + dL_dconic[3]*conic[4] + dL_dconic[4]*conic[5],
    	dL_dconic[2]*conic[2] + dL_dconic[4]*conic[4] + dL_dconic[5]*conic[5]
	};
	const float dL_dcov[6] = {
		-1.0f * (conic[0]*dL_dconic_conic[0] + conic[1]*dL_dconic_conic[1] + conic[2]*dL_dconic_conic[2]),
    	-1.0f * (conic[0]*dL_dconic_conic[1] + conic[1]*dL_dconic_conic[3] + conic[2]*dL_dconic_conic[4]),
    	-1.0f * (conic[0]*dL_dconic_conic[2] + conic[1]*dL_dconic_conic[4] + conic[2]*dL_dconic_conic[5]),
    	-1.0f * (conic[1]*dL_dconic_conic[1] + conic[3]*dL_dconic_conic[3] + conic[4]*dL_dconic_conic[4]),
    	-1.0f * (conic[1]*dL_dconic_conic[2] + conic[3]*dL_dconic_conic[4] + conic[4]*dL_dconic_conic[5]),
    	-1.0f * (conic[2]*dL_dconic_conic[2] + conic[4]*dL_dconic_conic[4] + conic[5]*dL_dconic_conic[5])
	};
	float abs_Sigma00 = abs(Sigma[0][0]);
	float abs_Sigma11 = abs(Sigma[1][1]);
	float abs_Sigma22 = abs(Sigma[2][2]);
	float depsilon_dSigma00 = (abs_Sigma00 >= abs_Sigma11 && abs_Sigma00 >= abs_Sigma22) ? 1e-5 * glm::sign(Sigma[0][0]) : 0.0f;
	float depsilon_dSigma11 = (abs_Sigma11 >= abs_Sigma00 && abs_Sigma11 >= abs_Sigma22) ? 1e-5 * glm::sign(Sigma[1][1]) : 0.0f;
	float depsilon_dSigma22 = (abs_Sigma22 >= abs_Sigma00 && abs_Sigma22 >= abs_Sigma11) ? 1e-5 * glm::sign(Sigma[2][2]) : 0.0f;
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov[0] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma00, dL_dcov[1], dL_dcov[2],
		dL_dcov[1], dL_dcov[3] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma11, dL_dcov[4],
		dL_dcov[2], dL_dcov[4], dL_dcov[5] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma22
	);
	glm::mat3 dL_dM = 2.f * dL_dSigma * M;
	glm::mat3 dL_dS = dL_dM * glm::transpose(R);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = dL_dS[0][0] * scale_modifier;
	dL_dscale->y = dL_dS[1][1] * scale_modifier;
	dL_dscale->z = dL_dS[2][2] * scale_modifier;

	dL_dM[0] *= scale_modifier * scale.x;
	dL_dM[1] *= scale_modifier * scale.y;
	dL_dM[2] *= scale_modifier * scale.z;
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dM[1][0] - dL_dM[0][1]) + 2 * y * (dL_dM[0][2] - dL_dM[2][0]) + 2 * x * (dL_dM[2][1] - dL_dM[1][2]);
	dL_dq.y = 2 * y * (dL_dM[0][1] + dL_dM[1][0]) + 2 * z * (dL_dM[0][2] + dL_dM[2][0]) + 2 * r * (dL_dM[2][1] - dL_dM[1][2]) - 4 * x * (dL_dM[2][2] + dL_dM[1][1]);
	dL_dq.z = 2 * x * (dL_dM[0][1] + dL_dM[1][0]) + 2 * r * (dL_dM[0][2] - dL_dM[2][0]) + 2 * z * (dL_dM[2][1] + dL_dM[1][2]) - 4 * y * (dL_dM[2][2] + dL_dM[0][0]);
	dL_dq.w = 2 * r * (dL_dM[1][0] - dL_dM[0][1]) + 2 * x * (dL_dM[0][2] + dL_dM[2][0]) + 2 * y * (dL_dM[2][1] + dL_dM[1][2]) - 4 * z * (dL_dM[1][1] + dL_dM[0][0]);
	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}


void BACKWARD::preprocess(
	int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* values,
	const float* weights,
	const float3 volume_mins,
	const float3 volume_maxes,
	const float* samples,
	const cuBQL::bvh3f& bvh,
	const float* out_cells,
	const float* out_weights,
	const float* dL_dsamples,
	const float* dL_dsample_weights,
	float* dL_dmean3D,
	float* dL_dvalue,
	float* dL_dweights,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{

	// Propagate gradients for remaining steps: using dL_dconics
	// propagate back to scales and rotations
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		values,
		weights,
		volume_mins, volume_maxes,
		samples,
		bvh,
		out_cells,
		out_weights,
		dL_dsamples,
		dL_dsample_weights,
		dL_dmean3D,
		dL_dvalue,
		dL_dweights,
		dL_dscale,
		dL_drot);
}