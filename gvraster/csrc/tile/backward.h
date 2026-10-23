#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
// Accumulates each Gaussian's gradients w.r.t. its activated parameters into its 12-float row
// of grad_rows: conic[6], mean[3], value, weight, pad.
void sample(dim3 grid, const uint2* ranges, const uint32_t* gaussian_ids, float3 bounds_min,
	uint3 cells, float3 cell_size, const float* jitter, const float4* packed,
	const float* out_values, const float* out_influence, const float* dL_dout_values,
	const float* dL_dout_influence, float* grad_rows, float threshold, float floor);

// Chains the rows back through the covariance and the activations to the raw parameters.
void prepare(int P, const float* raw_scales, const float* raw_rotations, const float* raw_values,
	const float* raw_weights, float max_scale, const float4* packed, const float* grad_rows,
	float* dL_dmeans, float* dL_dscales, float* dL_drotations, float* dL_dvalues,
	float* dL_dweights);
} // namespace BACKWARD
