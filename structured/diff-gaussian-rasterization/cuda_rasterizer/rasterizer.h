#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <functional>
#include <vector_types.h>

namespace CudaRasterizer
{
class Rasterizer
{
public:
	// Renders the raw-parameter Gaussians onto the jittered lattice; returns the number of
	// (Gaussian, tile) pairs.
	static int forward(std::function<char*(size_t)> geometryBuffer,
		std::function<char*(size_t)> binningBuffer, std::function<char*(size_t)> imageBuffer, int P,
		const float* means3D, const float* scales, const float* rotations, const float* values,
		const float* weights, float max_scale, const float* jitter, float3 volume_mins,
		float3 volume_maxes, uint3 num_cells, float* out_cells, float* out_weights, float occ_t,
		float floor);

	// Gradients w.r.t. the raw parameters. grad_rows is a zeroed P x 12 scratch buffer.
	static void backward(int P, int R, const float* scales, const float* rotations,
		const float* values, const float* weights, float max_scale, const float* jitter,
		float3 volume_mins, float3 volume_maxes, uint3 num_cells, const float* out_cells,
		const float* out_weights, char* geom_buffer, char* binning_buffer, char* img_buffer,
		const float* dL_dcells, const float* dL_dcell_weights, float* grad_rows, float* dL_dmeans,
		float* dL_dscales, float* dL_drots, float* dL_dvalues, float* dL_dweights, float occ_t,
		float floor);
};
}; // namespace CudaRasterizer

#endif
