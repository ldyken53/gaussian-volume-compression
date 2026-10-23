#pragma once

#include "config.h"

// The activations of gvraster.bvh.activate, recomputed from the raw parameters.
#define ACT_CAP_EPS 1e-8f
#define ACT_NORM_EPS 1e-12f

__device__ __forceinline__ float act_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
