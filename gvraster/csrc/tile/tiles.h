// Binning the Gaussians to the lattice's tiles, around the FORWARD and BACKWARD kernels.
#pragma once

#include <functional>
#include <vector_types.h>

namespace tiles
{
// Returns n bytes of storage that lasts from the forward pass to the backward pass.
using Allocator = std::function<char*(size_t)>;

// Samples the Gaussians on the jittered lattice; returns the number of (Gaussian, tile) pairs.
int forward(Allocator geom_buffer, Allocator binning_buffer, Allocator range_buffer, int P,
	const float* means, const float* raw_scales, const float* raw_rotations,
	const float* raw_values, const float* raw_weights, float max_scale, const float* jitter,
	float3 bounds_min, float3 bounds_max, uint3 cells, float* out_values, float* out_influence,
	float threshold, float floor);

// Gradients w.r.t. the raw parameters; grad_rows is a zeroed P x 12 scratch buffer.
void backward(int P, int pairs, const float* raw_scales, const float* raw_rotations,
	const float* raw_values, const float* raw_weights, float max_scale, const float* jitter,
	float3 bounds_min, float3 bounds_max, uint3 cells, const float* out_values,
	const float* out_influence, char* geom_buffer, char* binning_buffer, char* range_buffer,
	const float* dL_dout_values, const float* dL_dout_influence, float* grad_rows, float* dL_dmeans,
	float* dL_dscales, float* dL_drotations, float* dL_dvalues, float* dL_dweights, float threshold,
	float floor);
} // namespace tiles
