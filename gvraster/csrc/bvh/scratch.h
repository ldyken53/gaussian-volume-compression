#pragma once
#include <cstddef>
#include <cuda_runtime.h>

// A grow-only device buffer of count Ts: cudaMalloc and cudaFree synchronize the device.
template <typename T> T* growScratch(T*& ptr, size_t& bytes, size_t count)
{
	if (count * sizeof(T) > bytes) {
		if (ptr)
			cudaFree(ptr);
		cudaMalloc(&ptr, count * sizeof(T));
		bytes = count * sizeof(T);
	}
	return ptr;
}
