#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
// Accumulates each Gaussian's render gradients into its 12-float row of grad_rows:
// conic[6], mean[3], value, weight, pad (activated parameters).
void render(dim3 grid, const uint2* ranges, const uint32_t* point_list, float3 volume_mins,
	uint3 num_cells, float3 cell_size, const float* jitter, const float4* packed,
	const float* out_cells, const float* out_weights, const float* dL_dcells,
	const float* dL_dcell_weights, float* grad_rows, float occ_t, float floor);

// Chains the rows back through the covariance and the activations to the raw parameters.
void preprocess(int P, const float* scales, const float* rotations, const float* values,
	const float* weights, float max_scale, const float4* packed, const float* grad_rows,
	float* dL_dmeans, float* dL_dscales, float* dL_drots, float* dL_dvalues, float* dL_dweights);
} // namespace BACKWARD

#endif
