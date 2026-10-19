#pragma once

#include "rasterizer.h"
#include <cstdint>
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
template <typename T>
static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
{
	std::size_t offset =
		(reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
	ptr = reinterpret_cast<T*>(offset);
	chunk = reinterpret_cast<char*>(ptr + count);
}

struct GeometryState {
	size_t scan_size;
	char* scanning_space;
	// Three float4s per Gaussian: (mean, value), (weight, cxx, cxy, cxz), (cyy, cyz, czz, qmax).
	float4* packed;
	uint* aabbs;
	uint32_t* point_offsets;
	uint32_t* blocks_touched;

	static GeometryState fromChunk(char*& chunk, size_t P);
};

struct ImageState {
	uint2* ranges;

	static ImageState fromChunk(char*& chunk, size_t N);
};

struct BinningState {
	size_t sorting_size;
	uint32_t* point_list_keys_unsorted;
	uint32_t* point_list_keys;
	uint32_t* point_list_unsorted;
	uint32_t* point_list;
	char* list_sorting_space;

	static BinningState fromChunk(char*& chunk, size_t P);
};

template <typename T> size_t required(size_t P)
{
	char* size = nullptr;
	T::fromChunk(size, P);
	return ((size_t)size) + 128;
}
}; // namespace CudaRasterizer
