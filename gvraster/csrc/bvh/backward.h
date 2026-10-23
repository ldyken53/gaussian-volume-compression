#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cuBQL/bvh.h>

namespace BACKWARD
{
// Gradients w.r.t. the activated parameters of FORWARD::sample; dL_dconics is P x 6 scratch.
void sample(int P, int S, const float* means, const glm::vec3* scales, const glm::vec4* rotations,
	const float* conics, const float* values, const float* weights, const float* points,
	const cuBQL::bvh3f& bvh, float threshold, float floor, const float* out_values,
	const float* out_influence, const float* dL_dout_values, const float* dL_dout_influence,
	float* dL_dmeans, float* dL_dvalues, float* dL_dweights, glm::vec3* dL_dscales,
	glm::vec4* dL_drotations, float* dL_dconics);
} // namespace BACKWARD
