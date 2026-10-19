#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cuBQL/bvh.h>

namespace BACKWARD
{
// Gradients w.r.t. the activated parameters for the per-Gaussian forward pass.
// dL_dconics is P x 6 scratch.
void render(int P, int S, const float* means3D, const glm::vec3* scales, const glm::vec4* rotations,
	const float* conics, const float* values, const float* weights, const float* samples,
	const cuBQL::bvh3f& bvh, bool hard, const float* out_cells, const float* out_weights,
	const float* dL_dsamples, const float* dL_dsample_weights, float* dL_dmean3D, float* dL_dvalue,
	float* dL_dweights, glm::vec3* dL_dscale, glm::vec4* dL_drot, float* dL_dconics);
} // namespace BACKWARD

#endif
