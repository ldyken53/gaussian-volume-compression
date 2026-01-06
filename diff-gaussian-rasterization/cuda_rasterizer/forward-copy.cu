#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* values,
	const float* weights,
	bool* clamped,
	const float3 volume_mins,
	const float3 volume_maxes,
	const uint3 num_cells,
	const float3 cell_size,
	int* radii,
	float3* means,
	float* values_out,
	float* weights_out,
	float* volumes,
	float* conic,
	uint* aabbs,
	const dim3 grid,
	uint32_t* blocks_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize touched blocks to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	blocks_touched[idx] = 0;
	radii[idx] = 0;

	auto scale = scales[idx];
	auto rot = rotations[idx];
	
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale_modifier * scale.x;
	S[1][1] = scale_modifier * scale.y;
	S[2][2] = scale_modifier * scale.z;

	// Normalize quaternion to get valid rotation (commented out for some reason?)
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Normalize by epsilon to prevent numerical issues
	const float epsilon = max(max(abs(Sigma[0][0]), abs(Sigma[1][1])), abs(Sigma[2][2])) * 1e-5;
	const float cov[6] = {
		Sigma[0][0] + epsilon,
        Sigma[0][1],
        Sigma[0][2],
		Sigma[1][1] + epsilon,
        Sigma[1][2],
        Sigma[2][2] + epsilon,
	};

	// Use 3D covariance to compute and store 3D conic
	const float a = cov[0]; // Sigma[0][0]
    const float b = cov[1]; // Sigma[0][1]
    const float c = cov[2]; // Sigma[0][2]
    const float d = cov[3]; // Sigma[1][1]
    const float e = cov[4]; // Sigma[1][2]
    const float f = cov[5]; // Sigma[2][2]
    const float det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    const float det_inv = 1.0 / det;
    conic[idx * 6] = (d * f - e * e) * det_inv;
    conic[idx * 6 + 1] = (c * e - b * f) * det_inv;
    conic[idx * 6 + 2] = (b * e - c * d) * det_inv;
    conic[idx * 6 + 3] = (a * f - c * c) * det_inv;
    conic[idx * 6 + 4] = (b * c - a * e) * det_inv;
    conic[idx * 6 + 5] = (a * d - b * b) * det_inv;

	// Scale S by 3 to include up to three std from Gaussian position
	// const float m = 3.0;
	float m = sqrtf(-2 * logf((0.1 * WEIGHT_CUTOFF) / weights[idx]));
	const float3 scaled_S = { S[0][0] * m, S[1][1] * m, S[2][2] * m };

 	// Create array for corner computations
    const float n[2] = {-1.0f, 1.0f};
    
    // Initialize mins and maxes with gaussian position
	const float3 position = { means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2] };
	means[idx] = position;
    float3 mins = position;
    float3 maxes = position;

	// Compute corners using vector operations
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                float3 corner = make_float3(
					position.x + n[i] * R[0].x * scaled_S.x + n[j] * R[1].x * scaled_S.y +  n[k] * R[2].x * scaled_S.z,
					position.y + n[i] * R[0].y * scaled_S.x + n[j] * R[1].y * scaled_S.y +  n[k] * R[2].y * scaled_S.z,
					position.z + n[i] * R[0].z * scaled_S.x + n[j] * R[1].z * scaled_S.y +  n[k] * R[2].z * scaled_S.z
				);
                    
                mins = make_float3(
					min(mins.x, corner.x),
					min(mins.y, corner.y),
					min(mins.z, corner.z)
				);
                maxes = make_float3(
					max(maxes.x, corner.x),
					max(maxes.y, corner.y),
					max(maxes.z, corner.z)
				);
            }
        }
    }

	// Calculate block size in world coordinates
	const float block_size_x = cell_size.x * BLOCK_X;
	const float block_size_y = cell_size.y * BLOCK_Y;
	const float block_size_z = cell_size.z * BLOCK_Z;

	// Find which blocks the Gaussian intersects
	uint3 start_block = make_uint3(
		max(0, min(static_cast<int>(grid.x), static_cast<int>(floor((mins.x - volume_mins.x) / block_size_x)))),
		max(0, min(static_cast<int>(grid.y), static_cast<int>(floor((mins.y - volume_mins.y) / block_size_y)))),
		max(0, min(static_cast<int>(grid.z), static_cast<int>(floor((mins.z - volume_mins.z) / block_size_z))))
	);    
	uint3 end_block = make_uint3(
		max(0, min(static_cast<int>(grid.x), static_cast<int>(ceil((maxes.x - volume_mins.x) / block_size_x)))),
		max(0, min(static_cast<int>(grid.y), static_cast<int>(ceil((maxes.y - volume_mins.y) / block_size_y)))),
		max(0, min(static_cast<int>(grid.z), static_cast<int>(ceil((maxes.z - volume_mins.z) / block_size_z))))
	);
    uint3 block_dims = make_uint3(
		end_block.x - start_block.x,
		end_block.y - start_block.y,
		end_block.z - start_block.z
	);

    // Store results
    blocks_touched[idx] = static_cast<int>(block_dims.x * block_dims.y * block_dims.z);
	radii[idx] = 1;
    aabbs[idx * 6] = start_block.x;
	aabbs[idx * 6 + 1] = start_block.y;
    aabbs[idx * 6 + 2] = start_block.z;
    aabbs[idx * 6 + 3] = end_block.x;
    aabbs[idx * 6 + 4] = end_block.y;
	aabbs[idx * 6 + 5] = end_block.z;
	// Clamping may not be necessary since these are stored with sigmoid activation?
	// clamped[idx] = (values[idx] < 0.0f) || (values[idx] > 1.0f);
    // values_out[idx] = glm::clamp(values[idx], 0.0f, 1.0f); 
	// weights_out[idx] = glm::clamp(weights[idx], 0.0f, 1.0f); 
	clamped[idx] = false;
    values_out[idx] = values[idx]; 
	weights_out[idx] = weights[idx]; 
    volumes[idx] = static_cast<float>(block_dims.x * block_dims.y * block_dims.z);
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(256)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const dim3 grid,
	const float3 volume_mins,
	const uint3 num_cells,
	const float3 cell_size,
	const float* __restrict__ jitter,
	const float3* __restrict__ means,
	const float* __restrict__ values,
	const float* __restrict__ weights,
	const float* __restrict__ volumes,
	const float* __restrict__ conic,
	float* __restrict__ accumulated_weights,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ out_cells)
{
	// Compute global cell index from 1D grid of threads
	uint32_t cell_id = blockIdx.x * 256 + threadIdx.x;
	uint32_t total_cells = num_cells.x * num_cells.y * num_cells.z;
	
	if (cell_id >= total_cells)
		return;

	// Convert linear cell_id to 3D coordinates
	uint32_t cell_x = cell_id % num_cells.x;
	uint32_t cell_y = (cell_id / num_cells.x) % num_cells.y;
	uint32_t cell_z = cell_id / (num_cells.x * num_cells.y);

	float3 cell_pos = make_float3(
		static_cast<float>(cell_x) * cell_size.x + volume_mins.x + jitter[cell_id * 3], 
		static_cast<float>(cell_y) * cell_size.y + volume_mins.y + jitter[cell_id * 3 + 1], 
		static_cast<float>(cell_z) * cell_size.z + volume_mins.z + jitter[cell_id * 3 + 2]
	);

	// Compute tile index for range lookup
	uint32_t tile_x = cell_x / BLOCK_X;
	uint32_t tile_y = cell_y / BLOCK_Y;
	uint32_t tile_z = cell_z / BLOCK_Z;
	uint32_t tile_id = tile_z * grid.y * grid.x + tile_y * grid.x + tile_x;
	
	uint2 range = ranges[tile_id];

	// Initialize accumulators
	float accumulated_weight = 0;
	float accumulated_value = 0;
	uint32_t n_contributor = 0;

	// Each thread independently iterates over its range
	for (uint32_t idx = range.x; idx < range.y; idx++)
	{
		n_contributor++;
		
		int pt_id = point_list[idx];
		float3 mean = means[pt_id];
		
		float3 d = make_float3(cell_pos.x - mean.x, cell_pos.y - mean.y, cell_pos.z - mean.z);
		
		float c0 = conic[pt_id * 6];
		float c1 = conic[pt_id * 6 + 1];
		float c2 = conic[pt_id * 6 + 2];
		float c3 = conic[pt_id * 6 + 3];
		float c4 = conic[pt_id * 6 + 4];
		float c5 = conic[pt_id * 6 + 5];
		
		float quad_form = (
			d.x * (c0 * d.x + c1 * d.y + c2 * d.z) +
			d.y * (c1 * d.x + c3 * d.y + c4 * d.z) +
			d.z * (c2 * d.x + c4 * d.y + c5 * d.z)
		);
		
		float power = -0.5f * quad_form;
		if (power < -14.0f || power > 0.0f) continue;
		
		float weight = weights[pt_id] * expf(power);
		accumulated_value += values[pt_id] * weight;
		accumulated_weight += weight;
	}

	// Write output
	if (accumulated_weight > WEIGHT_CUTOFF) {
		out_cells[cell_id] = accumulated_value / accumulated_weight;
		accumulated_weights[cell_id] = accumulated_weight;
		n_contrib[cell_id] = n_contributor;
	} else {
		out_cells[cell_id] = -1.0f;
		accumulated_weights[cell_id] = 0.0f;
		n_contrib[cell_id] = n_contributor;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const float3 volume_mins,
	const uint3 num_cells,
	const float3 cell_size,
	const float* jitter,
	const float3* means,
	const float* values,
	const float* weights,
	const float* volumes,
	const float* conic,
	float* accumulated_weights,
	uint32_t* n_contrib,
	float* out_cells)
{
	uint32_t total_cells = num_cells.x * num_cells.y * num_cells.z;
	uint32_t num_blocks = (total_cells + 256 - 1) / 256;
	
	renderCUDA<NUM_CHANNELS> <<<num_blocks, 256>>> (
		ranges,
		point_list,
		grid,  // still needed for tile->range lookup
		volume_mins,
		num_cells,
		cell_size,
		jitter,
		means,
		values,
		weights,
		volumes,
		conic,
		accumulated_weights,
		n_contrib,
		out_cells);
}


void FORWARD::preprocess(int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* values,
	const float* weights,
	bool* clamped,
	const float3 volume_mins,
	const float3 volume_maxes,
	const uint3 num_cells,
	const float3 cell_size,
	int* radii,
	float3* means,
	float* values_out,
	float* weights_out,
	float* volumes,
	float* conic,
	uint* aabbs,
	const dim3 grid,
	uint32_t* blocks_touched)
{
	preprocessCUDA<NUM_CHANNELS> <<<(P + 255) / 256, 256>>> (
		P,
		means3D,
		scales,
		scale_modifier,
		rotations,
		values,
		weights,
		clamped,
		volume_mins,
		volume_maxes,
		num_cells,
		cell_size,
		radii,
		means,
		values_out, 
		weights_out,
		volumes,
		conic,
		aabbs,
		grid,
		blocks_touched
	);
}
