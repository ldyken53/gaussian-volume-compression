#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cuBQL/bvh.h>

namespace BACKWARD
{
	void preprocess(
		int P,
		const float3* means3D,
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
		float3* dL_dmean3D,
		float* dL_dvalue,
		float* dL_dweights,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot
	);

}

#endif
