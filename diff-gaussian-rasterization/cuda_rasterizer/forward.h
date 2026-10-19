#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cuBQL/bvh.h>

namespace FORWARD
{
// Conic and support box of every Gaussian.
void preprocess(int P, const float* means3D, const glm::vec3* scales, const glm::vec4* rotations,
	const float* weights, float* conics, cuBQL::box3f* aabbs);

// Accumulates each Gaussian into the samples of the BVH leaves its box overlaps, then
// normalizes: out_test = S / aw (hard) or S / max(aw, WEIGHT_CUTOFF), -1 for empty samples.
void render(int P, int S, const float* means3D, const float* values, const float* weights,
	const float* conics, const cuBQL::box3f* aabbs, const cuBQL::bvh3f& bvh, bool hard,
	float* out_test, float* out_testw);
} // namespace FORWARD

#endif
