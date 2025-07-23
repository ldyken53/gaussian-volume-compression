#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <cuBQL/bvh.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
void CudaRasterizer::Rasterizer::forward(
	const int P, const int S,
	const float* means3D,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* values,
	const float* weights,
	const float3 volume_mins,
	const float3 volume_maxes,
	const float* samples,
	const cuBQL::bvh3f& bvh,
	float* out_test,
	float* out_testw,
	bool debug)
{	
	// Create CUDA events for timing (only when debug is enabled)
	cudaEvent_t events[2]; // 8 pairs of start/stop events
	if (debug) {
		for (int i = 0; i < 2; i++) {
			cudaEventCreate(&events[i]);
		}
	}

	// Preprocessing
	if (debug) cudaEventRecord(events[0]);
	CHECK_CUDA(FORWARD::preprocess(
		P, S,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		values,
		weights,
		volume_mins, volume_maxes,
		samples,
		bvh,
		out_test,
		out_testw
	), debug)
	if (debug) cudaEventRecord(events[1]);

	// Calculate and print timing (only when debug is enabled)
	if (debug) {
		cudaDeviceSynchronize();
		
		float elapsed_time;
		const char* operation_names[] = {
			"Preprocessing"
		};
		
		for (int i = 0; i < 1; i++) {
			cudaEventElapsedTime(&elapsed_time, events[i*2], events[i*2+1]);
			std::cout << operation_names[i] << " time: " << elapsed_time << " ms" << std::endl;
		}
		
		// Clean up events
		for (int i = 0; i < 2; i++) {
			cudaEventDestroy(events[i]);
		}
	}
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P,
	const float* means3D,
	const float* scales,
	const float scale_modifier,
	const float3 volume_mins, const float3 volume_maxes,
	const float* rotations,
	const float* values,
	const float* weights,
	const float* samples,
	const cuBQL::bvh3f& bvh,
	const float* out_cells,
	const float* out_weights,
	const float* dL_dsamples,
	const float* dL_dsample_weights,
	float* dL_dmean3D,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dvalue,
	float* dL_dweights,
	bool debug)
{

	// Create CUDA events for timing (only when debug is enabled)
	cudaEvent_t events[2]; // 2 pairs of start/stop events
	if (debug) {
		for (int i = 0; i < 2; i++) {
			cudaEventCreate(&events[i]);
		}
	}

	if (debug) cudaEventRecord(events[0]);
	// Take care of the rest of preprocessing, compute loss w.r.t
	// scales and rotation from conic gradients.
	CHECK_CUDA(BACKWARD::preprocess(P,
		(float3*) means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		values,
		weights,
		volume_mins, volume_maxes,
		samples,
		bvh,
		out_cells,
		out_weights,
		dL_dsamples,
		dL_dsample_weights,
		(float3*) dL_dmean3D,
		dL_dvalue,
		dL_dweights,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug);
	if (debug) cudaEventRecord(events[1]);

	if (debug) {
		cudaDeviceSynchronize(); // ensure all events are completed
		float elapsed_time;
		const char* operation_names[] = { "Backward Preprocess" };

		for (int i = 0; i < 1; ++i) {
			cudaEventElapsedTime(&elapsed_time, events[i * 2], events[i * 2 + 1]);
			std::cout << operation_names[i] << " time: " << elapsed_time << " ms" << std::endl;
		}

		for (int i = 0; i < 2; ++i) {
			cudaEventDestroy(events[i]);
		}
	}
}
