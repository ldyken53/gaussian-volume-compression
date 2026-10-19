#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
// Activates the raw parameters, writes each Gaussian's packed render record and the tile
// range its support ellipsoid overlaps.
void preprocess(int P, const float* means3D, const float* scales, const float* rotations,
	const float* values, const float* weights, float max_scale, float3 volume_mins,
	float3 cell_size, dim3 grid, float4* packed, uint* aabbs, uint32_t* blocks_touched);

void render(dim3 grid, dim3 block, const uint2* ranges, const uint32_t* point_list,
	float3 volume_mins, uint3 num_cells, float3 cell_size, const float* jitter,
	const float4* packed, float* out_weights, float* out_cells);
} // namespace FORWARD

#endif
