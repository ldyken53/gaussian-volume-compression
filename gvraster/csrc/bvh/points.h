// The points every sampleForward / sampleBackward evaluates at, set by setPoints.
#pragma once
#include <torch/extension.h>
#include <cuBQL/bvh.h>

struct Points {
	torch::Tensor xyz;
	// (x, y, z, 0) per point: one aligned load per lane.
	const float4* packed = nullptr;
	cuBQL::bvh3f bvh;
};

const Points& currentPoints();

// A BVH over n points, owned by the caller.
void buildPointBVH(const float* xyz, int n, cuBQL::bvh3f& out);
