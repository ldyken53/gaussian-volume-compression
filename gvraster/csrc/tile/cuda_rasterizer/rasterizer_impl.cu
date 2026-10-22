#include "rasterizer_impl.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "backward.h"
#include "forward.h"

// Number of bits needed to hold n.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1) {
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Writes one (tile, Gaussian) key/value pair per overlap, 2^log2g threads per Gaussian.
// Each pair's slot is independent of the group size, so the sort input is too.
__global__ void duplicateWithKeys(int P, int log2g, const uint* aabbs, const uint32_t* offsets,
	uint32_t* gaussian_keys_unsorted, uint32_t* gaussian_values_unsorted, dim3 grid)
{
	const unsigned long long t = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
	const unsigned int idx = (unsigned int)(t >> log2g);
	if (idx >= (unsigned int)P)
		return;
	const unsigned int g = 1u << log2g;
	const unsigned int lane = (unsigned int)t & (g - 1);

	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	if (offsets[idx] == off)
		return;

	uint x0 = aabbs[idx * 6 + 0], y0 = aabbs[idx * 6 + 1], z0 = aabbs[idx * 6 + 2];
	uint nx = aabbs[idx * 6 + 3] - x0, ny = aabbs[idx * 6 + 4] - y0, nz = aabbs[idx * 6 + 5] - z0;
	uint total = nx * ny * nz;
	for (uint i = lane; i < total; i += g) {
		uint iz = i / (nx * ny), rem = i - iz * (nx * ny);
		uint iy = rem / nx, ix = rem - iy * nx;
		gaussian_keys_unsorted[off + i] =
			(z0 + iz) * grid.x * grid.y + (y0 + iy) * grid.x + (x0 + ix);
		gaussian_values_unsorted[off + i] = idx;
	}
}

// Marks the start and end of each tile's run in the sorted pair list.
__global__ void identifyTileRanges(int L, uint32_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	uint32_t currcell = point_list_keys[idx];
	if (idx == 0)
		ranges[currcell].x = 0;
	else {
		uint32_t prevcell = point_list_keys[idx - 1];
		if (currcell != prevcell) {
			ranges[prevcell].y = idx;
			ranges[currcell].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currcell].y = L;
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.packed, P * 3, 128);
	obtain(chunk, geom.aabbs, P * 6, 128);
	obtain(chunk, geom.blocks_touched, P, 128);
	cub::DeviceScan::InclusiveSum(
		nullptr, geom.scan_size, geom.blocks_touched, geom.blocks_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(nullptr, binning.sorting_size, binning.point_list_keys_unsorted,
		binning.point_list_keys, binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

static float3 cellSize(float3 volume_mins, float3 volume_maxes, uint3 num_cells)
{
	return make_float3((volume_maxes.x - volume_mins.x) / float(num_cells.x - 1),
		(volume_maxes.y - volume_mins.y) / float(num_cells.y - 1),
		(volume_maxes.z - volume_mins.z) / float(num_cells.z - 1));
}

static dim3 tileGrid(uint3 num_cells)
{
	return dim3((num_cells.x + BLOCK_X - 1) / BLOCK_X, (num_cells.y + BLOCK_Y - 1) / BLOCK_Y,
		(num_cells.z + BLOCK_Z - 1) / BLOCK_Z);
}

int CudaRasterizer::Rasterizer::forward(std::function<char*(size_t)> geometryBuffer,
	std::function<char*(size_t)> binningBuffer, std::function<char*(size_t)> imageBuffer, int P,
	const float* means3D, const float* scales, const float* rotations, const float* values,
	const float* weights, float max_scale, const float* jitter, float3 volume_mins,
	float3 volume_maxes, uint3 num_cells, float* out_cells, float* out_weights, float occ_t,
	float floor)
{
	const float3 cell_size = cellSize(volume_mins, volume_maxes, num_cells);
	const dim3 block_grid = tileGrid(num_cells);
	const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
	const size_t n_tiles = block_grid.x * block_grid.y * block_grid.z;

	char* chunkptr = geometryBuffer(required<GeometryState>(P));
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);
	const size_t N = num_cells.x * num_cells.y * num_cells.z;
	char* img_chunkptr = imageBuffer(required<ImageState>(N));
	ImageState imgState = ImageState::fromChunk(img_chunkptr, N);

	FORWARD::preprocess(P, means3D, scales, rotations, values, weights, max_scale, volume_mins,
		cell_size, block_grid, geomState.packed, geomState.aabbs, geomState.blocks_touched);

	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
		geomState.blocks_touched, geomState.point_offsets, P);
	int num_intersections;
	cudaMemcpy(
		&num_intersections, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);

	char* binning_chunkptr = binningBuffer(required<BinningState>(num_intersections));
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_intersections);

	// Group size tracks the mean tile count: a full warp idles on Gaussians spanning a few
	// tiles, one thread serialises those spanning ~100.
	int log2g = 0;
	const double mean_tiles = P > 0 ? (double)num_intersections / P : 0.0;
	while (log2g < 5 && (double)(1 << log2g) < mean_tiles)
		log2g++;
	const unsigned long long dup_threads = (unsigned long long)P << log2g;
	duplicateWithKeys<<<(unsigned int)((dup_threads + 255) / 256), 256>>>(P, log2g, geomState.aabbs,
		geomState.point_offsets, binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted, block_grid);

	// Bits in the largest key, not the tile count: one extra bit costs CUB another pass.
	int bit = getHigherMsb(n_tiles - 1);
	cub::DeviceRadixSort::SortPairs(binningState.list_sorting_space, binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list, num_intersections, 0, bit);

	cudaMemset(imgState.ranges, 0, n_tiles * sizeof(uint2));
	if (num_intersections > 0)
		identifyTileRanges<<<(num_intersections + 255) / 256, 256>>>(
			num_intersections, binningState.point_list_keys, imgState.ranges);

	FORWARD::render(block_grid, block, imgState.ranges, binningState.point_list, volume_mins,
		num_cells, cell_size, jitter, geomState.packed, out_weights, out_cells, occ_t, floor);

	return num_intersections;
}

void CudaRasterizer::Rasterizer::backward(int P, int R, const float* scales, const float* rotations,
	const float* values, const float* weights, float max_scale, const float* jitter,
	float3 volume_mins, float3 volume_maxes, uint3 num_cells, const float* out_cells,
	const float* out_weights, char* geom_buffer, char* binning_buffer, char* img_buffer,
	const float* dL_dcells, const float* dL_dcell_weights, float* grad_rows, float* dL_dmeans,
	float* dL_dscales, float* dL_drots, float* dL_dvalues, float* dL_dweights, float occ_t,
	float floor)
{
	const float3 cell_size = cellSize(volume_mins, volume_maxes, num_cells);
	const dim3 block_grid = tileGrid(num_cells);

	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState =
		ImageState::fromChunk(img_buffer, num_cells.x * num_cells.y * num_cells.z);

	BACKWARD::render(block_grid, imgState.ranges, binningState.point_list, volume_mins, num_cells,
		cell_size, jitter, geomState.packed, out_cells, out_weights, dL_dcells, dL_dcell_weights,
		grad_rows, occ_t, floor);
	BACKWARD::preprocess(P, scales, rotations, values, weights, max_scale, geomState.packed,
		grad_rows, dL_dmeans, dL_dscales, dL_drots, dL_dvalues, dL_dweights);
}
