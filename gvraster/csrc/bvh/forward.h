#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cuBQL/bvh.h>

namespace FORWARD
{
// The conic and support box of every Gaussian.
void prepare(int P, const float* means, const glm::vec3* scales, const glm::vec4* rotations,
	const float* weights, float* conics, cuBQL::box3f* boxes);

// V and I at the points in the BVH leaves each Gaussian's box overlaps, then the rule:
// out_values = V / max(I, floor), or -1 where I <= threshold. reach (P, or null): per Gaussian,
// the largest w G at a point whose packed 4th float is > 0.
void sample(int P, int S, const float* means, const float* values, const float* weights,
	const float* conics, const cuBQL::box3f* boxes, const float4* points, const cuBQL::bvh3f& bvh,
	float threshold, float floor, float* out_values, float* out_influence, float* reach);
} // namespace FORWARD
