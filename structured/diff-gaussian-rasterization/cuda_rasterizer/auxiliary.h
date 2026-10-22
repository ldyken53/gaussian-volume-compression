#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"

#define ACT_CAP_EPS 1e-8f
#define ACT_NORM_EPS 1e-12f

__device__ __forceinline__ float act_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

#endif
