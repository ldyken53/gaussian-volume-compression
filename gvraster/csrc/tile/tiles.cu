#include "tiles.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "activation.h"
#include "backward.h"
#include "forward.h"

// The number of bits needed to hold n.
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

// Writes one (tile, Gaussian) key/value pair per overlap, 2^log2g threads per Gaussian. Each
// pair's slot is independent of the group size, so the sort input is too.
__global__ void duplicateWithKeys(int P, int log2g, const uint* tile_boxes,
	const uint32_t* pair_offsets, uint32_t* tile_keys_unsorted, uint32_t* gaussian_ids_unsorted,
	dim3 grid)
{
	const unsigned long long t = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
	const unsigned int idx = (unsigned int)(t >> log2g);
	if (idx >= (unsigned int)P)
		return;
	const unsigned int g = 1u << log2g;
	const unsigned int lane = (unsigned int)t & (g - 1);

	uint32_t off = (idx == 0) ? 0 : pair_offsets[idx - 1];
	if (pair_offsets[idx] == off)
		return;

	uint x0 = tile_boxes[idx * 6 + 0], y0 = tile_boxes[idx * 6 + 1], z0 = tile_boxes[idx * 6 + 2];
	uint nx = tile_boxes[idx * 6 + 3] - x0, ny = tile_boxes[idx * 6 + 4] - y0,
		 nz = tile_boxes[idx * 6 + 5] - z0;
	uint total = nx * ny * nz;
	for (uint i = lane; i < total; i += g) {
		uint iz = i / (nx * ny), rem = i - iz * (nx * ny);
		uint iy = rem / nx, ix = rem - iy * nx;
		tile_keys_unsorted[off + i] = (z0 + iz) * grid.x * grid.y + (y0 + iy) * grid.x + (x0 + ix);
		gaussian_ids_unsorted[off + i] = idx;
	}
}

// Marks the start and end of each tile's run in the sorted pairs.
__global__ void identifyTileRanges(int L, uint32_t* tile_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	uint32_t currtile = tile_keys[idx];
	if (idx == 0)
		ranges[currtile].x = 0;
	else {
		uint32_t prevtile = tile_keys[idx - 1];
		if (currtile != prevtile) {
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

namespace
{

// Takes an aligned array of count Ts from the front of chunk.
template <typename T> void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
{
	std::size_t offset =
		(reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
	ptr = reinterpret_cast<T*>(offset);
	chunk = reinterpret_cast<char*>(ptr + count);
}

struct GeometryState {
	size_t scan_size;
	char* scan_space;
	// Three float4s per Gaussian: (mean, value), (weight, cxx, cxy, cxz), (cyy, cyz, czz, qmax).
	float4* packed;
	uint* tile_boxes;
	uint32_t* pair_offsets;
	uint32_t* tiles_touched;

	static GeometryState fromChunk(char*& chunk, size_t P)
	{
		GeometryState geom;
		obtain(chunk, geom.packed, P * 3, 128);
		obtain(chunk, geom.tile_boxes, P * 6, 128);
		obtain(chunk, geom.tiles_touched, P, 128);
		cub::DeviceScan::InclusiveSum(
			nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
		obtain(chunk, geom.scan_space, geom.scan_size, 128);
		obtain(chunk, geom.pair_offsets, P, 128);
		return geom;
	}
};

struct TileState {
	uint2* ranges;

	static TileState fromChunk(char*& chunk, size_t n_tiles)
	{
		TileState tiles;
		obtain(chunk, tiles.ranges, n_tiles, 128);
		return tiles;
	}
};

struct BinningState {
	size_t sort_size;
	uint32_t* tile_keys_unsorted;
	uint32_t* tile_keys;
	uint32_t* gaussian_ids_unsorted;
	uint32_t* gaussian_ids;
	char* sort_space;

	static BinningState fromChunk(char*& chunk, size_t pairs)
	{
		BinningState binning;
		obtain(chunk, binning.gaussian_ids, pairs, 128);
		obtain(chunk, binning.gaussian_ids_unsorted, pairs, 128);
		obtain(chunk, binning.tile_keys, pairs, 128);
		obtain(chunk, binning.tile_keys_unsorted, pairs, 128);
		cub::DeviceRadixSort::SortPairs(nullptr, binning.sort_size, binning.tile_keys_unsorted,
			binning.tile_keys, binning.gaussian_ids_unsorted, binning.gaussian_ids, pairs);
		obtain(chunk, binning.sort_space, binning.sort_size, 128);
		return binning;
	}
};

// The bytes State::fromChunk takes for n elements, plus the alignment slack.
template <typename State> size_t required(size_t n)
{
	char* size = nullptr;
	State::fromChunk(size, n);
	return ((size_t)size) + 128;
}

float3 cellSize(float3 bounds_min, float3 bounds_max, uint3 cells)
{
	return make_float3((bounds_max.x - bounds_min.x) / float(cells.x - 1),
		(bounds_max.y - bounds_min.y) / float(cells.y - 1),
		(bounds_max.z - bounds_min.z) / float(cells.z - 1));
}

dim3 tileGrid(uint3 cells)
{
	return dim3((cells.x + BLOCK_X - 1) / BLOCK_X, (cells.y + BLOCK_Y - 1) / BLOCK_Y,
		(cells.z + BLOCK_Z - 1) / BLOCK_Z);
}

} // namespace

int tiles::forward(Allocator geom_buffer, Allocator binning_buffer, Allocator range_buffer, int P,
	const float* means, const float* raw_scales, const float* raw_rotations,
	const float* raw_values, const float* raw_weights, float max_scale, const float* jitter,
	float3 bounds_min, float3 bounds_max, uint3 cells, float* out_values, float* out_influence,
	float threshold, float floor)
{
	const float3 cell_size = cellSize(bounds_min, bounds_max, cells);
	const dim3 grid = tileGrid(cells);
	const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
	const size_t n_tiles = grid.x * grid.y * grid.z;

	char* chunk = geom_buffer(required<GeometryState>(P));
	GeometryState geom = GeometryState::fromChunk(chunk, P);
	chunk = range_buffer(required<TileState>(n_tiles));
	TileState tile_state = TileState::fromChunk(chunk, n_tiles);

	FORWARD::prepare(P, means, raw_scales, raw_rotations, raw_values, raw_weights, max_scale,
		bounds_min, cell_size, grid, geom.packed, geom.tile_boxes, geom.tiles_touched);

	cub::DeviceScan::InclusiveSum(
		geom.scan_space, geom.scan_size, geom.tiles_touched, geom.pair_offsets, P);
	int pairs;
	cudaMemcpy(&pairs, geom.pair_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);

	chunk = binning_buffer(required<BinningState>(pairs));
	BinningState binning = BinningState::fromChunk(chunk, pairs);

	// The group size tracks the mean tile count: a full warp idles on Gaussians spanning a few
	// tiles, one thread serialises those spanning ~100.
	int log2g = 0;
	const double mean_tiles = P > 0 ? (double)pairs / P : 0.0;
	while (log2g < 5 && (double)(1 << log2g) < mean_tiles)
		log2g++;
	const unsigned long long dup_threads = (unsigned long long)P << log2g;
	duplicateWithKeys<<<(unsigned int)((dup_threads + 255) / 256), 256>>>(P, log2g, geom.tile_boxes,
		geom.pair_offsets, binning.tile_keys_unsorted, binning.gaussian_ids_unsorted, grid);

	// Bits in the largest key, not the tile count: one extra bit costs CUB another pass.
	int bit = getHigherMsb(n_tiles - 1);
	cub::DeviceRadixSort::SortPairs(binning.sort_space, binning.sort_size,
		binning.tile_keys_unsorted, binning.tile_keys, binning.gaussian_ids_unsorted,
		binning.gaussian_ids, pairs, 0, bit);

	cudaMemset(tile_state.ranges, 0, n_tiles * sizeof(uint2));
	if (pairs > 0)
		identifyTileRanges<<<(pairs + 255) / 256, 256>>>(
			pairs, binning.tile_keys, tile_state.ranges);

	FORWARD::sample(grid, block, tile_state.ranges, binning.gaussian_ids, bounds_min, cells,
		cell_size, jitter, geom.packed, out_influence, out_values, threshold, floor);

	return pairs;
}

void tiles::backward(int P, int pairs, const float* raw_scales, const float* raw_rotations,
	const float* raw_values, const float* raw_weights, float max_scale, const float* jitter,
	float3 bounds_min, float3 bounds_max, uint3 cells, const float* out_values,
	const float* out_influence, char* geom_buffer, char* binning_buffer, char* range_buffer,
	const float* dL_dout_values, const float* dL_dout_influence, float* grad_rows, float* dL_dmeans,
	float* dL_dscales, float* dL_drotations, float* dL_dvalues, float* dL_dweights, float threshold,
	float floor)
{
	const float3 cell_size = cellSize(bounds_min, bounds_max, cells);
	const dim3 grid = tileGrid(cells);

	GeometryState geom = GeometryState::fromChunk(geom_buffer, P);
	BinningState binning = BinningState::fromChunk(binning_buffer, pairs);
	TileState tile_state = TileState::fromChunk(range_buffer, grid.x * grid.y * grid.z);

	BACKWARD::sample(grid, tile_state.ranges, binning.gaussian_ids, bounds_min, cells, cell_size,
		jitter, geom.packed, out_values, out_influence, dL_dout_values, dL_dout_influence,
		grad_rows, threshold, floor);
	BACKWARD::prepare(P, raw_scales, raw_rotations, raw_values, raw_weights, max_scale, geom.packed,
		grad_rows, dL_dmeans, dL_dscales, dL_drotations, dL_dvalues, dL_dweights);
}
