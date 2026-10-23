#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
// Activates the raw parameters, and writes each Gaussian's packed record and the range of tiles
// its support ellipsoid overlaps.
void prepare(int P, const float* means, const float* raw_scales, const float* raw_rotations,
	const float* raw_values, const float* raw_weights, float max_scale, float3 bounds_min,
	float3 cell_size, dim3 grid, float4* packed, uint* tile_boxes, uint32_t* tiles_touched);

// V and I at every lattice point from the Gaussians binned to its tile, then the rule.
void sample(dim3 grid, dim3 block, const uint2* ranges, const uint32_t* gaussian_ids,
	float3 bounds_min, uint3 cells, float3 cell_size, const float* jitter, const float4* packed,
	float* out_influence, float* out_values, float threshold, float floor);
} // namespace FORWARD
