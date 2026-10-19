/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include <cstdlib>
#include <glm/glm.hpp>
#include <cuBQL/bvh.h>

#define BLOCK_SIZE (BLOCK_X * BLOCK_Y * BLOCK_Z)
#define NUM_WARPS (BLOCK_SIZE/32)
#define DGR_FIX_AA

// 3D covariance exactly as forward preprocessCUDA has always built it, including the
// epsilon (and its double literal). The conic is the inverse of these exact values, and
// the forward and backward Gaussian boxes are both derived from them, so this is the one
// place they are computed.
__forceinline__ __device__ void computeCov3D(const glm::vec3 scale, const float mod,
	const glm::vec4 rot, float cov[6])
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Quaternion is used as given (the caller passes the normalised get_rotation)
	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Normalize by epsilon to prevent numerical issues
	const float epsilon = max(max(abs(Sigma[0][0]), abs(Sigma[1][1])), abs(Sigma[2][2])) * 1e-5;
	cov[0] = Sigma[0][0] + epsilon;
	cov[1] = Sigma[0][1];
	cov[2] = Sigma[0][2];
	cov[3] = Sigma[1][1] + epsilon;
	cov[4] = Sigma[1][2];
	cov[5] = Sigma[2][2] + epsilon;
}

// Per-Gaussian support threshold T (see SUPPORT_T in config.h). Each compiled file holds its
// own copy in constant memory; ensureSupportT() fills it once per process from the SUPPORT_T env
// (else the config.h default), so forward and backward always agree.
static __constant__ float c_support_t = (float)SUPPORT_T;
// Blend floor of the SOFT_CUTOFF render: cells render S / max(aw, floor). DENOM_FLOOR env
// overrides WEIGHT_CUTOFF (0 = no floor). From struct 4e1c7ce: training with the floor and
// rendering without it was free there (+-0.01 dB).
static __constant__ float c_denom_floor = (float)WEIGHT_CUTOFF;
// Occupancy threshold: a cell is empty (-1) iff its accumulated weight aw <= c_occ_t. OCC_T env
// overrides the build default -- WEIGHT_CUTOFF for the hard cutoff, 0 for SOFT_CUTOFF (empty
// only when no support reaches the cell). Separate from the blend floor and SUPPORT_T so a
// surface model can put its inside/outside boundary at an iso-level of aw of its choosing.
#if SOFT_CUTOFF
static __constant__ float c_occ_t = 0.0f;
#else
static __constant__ float c_occ_t = (float)WEIGHT_CUTOFF;
#endif
// OCC_KEEP_AW=1: an empty cell still reports its true aw (instead of 0) and passes the weight
// gradient (not the value gradient) back, so the FN/FP terms keep acting on cells just under
// the occupancy threshold -- a margin classifier on aw rather than a cliff at it.
static __constant__ int c_occ_keep = 0;
static inline float ensureSupportT()
{
	static float t = -1.0f;
	if (t < 0.0f) {
		const char* e = getenv("SUPPORT_T");
		t = e ? (float)atof(e) : (float)SUPPORT_T;
		cudaMemcpyToSymbol(c_support_t, &t, sizeof(float));
		if (e) printf("[rasterizer] SUPPORT_T override: %.1e\n", t);
		const char* o = getenv("OCC_T");
		if (o) {
			const float ot = (float)atof(o);
			cudaMemcpyToSymbol(c_occ_t, &ot, sizeof(float));
			printf("[rasterizer] OCC_T override: %.1e\n", ot);
		}
		if (getenv("OCC_KEEP_AW") && atoi(getenv("OCC_KEEP_AW"))) {
			const int one = 1;
			cudaMemcpyToSymbol(c_occ_keep, &one, sizeof(int));
			printf("[rasterizer] OCC_KEEP_AW on\n");
		}
		const char* f = getenv("DENOM_FLOOR");
		const float fl = f ? (float)atof(f) : (float)WEIGHT_CUTOFF;
		cudaMemcpyToSymbol(c_denom_floor, &fl, sizeof(float));
		if (f) printf("[rasterizer] DENOM_FLOOR override: %.1e\n", fl);
	}
	return t;
}

// Squared support radius of a Gaussian of weight w: it contributes where q <= qmax, i.e.
// w * exp(-q/2) >= T. Negative when w < T (no support at all).
__forceinline__ __device__ float supportQmax(const float w)
{
	return 2.0f * logf(w / c_support_t);
}

// Support radius in sigmas, clamped at 0 for w < T: the old corner box swallowed the NaN this
// would otherwise be (min/max against the position drop it, leaving a zero-size box), but a NaN
// fed into the tight box makes a box that never rejects, so the BVH query visits every sample.
__forceinline__ __device__ float truncationRadius(const float w)
{
	return sqrtf(fmaxf(0.0f, supportQmax(w)));
}

// Tight axis-aligned bounds of the m-sigma ellipsoid {(x-p)^T cov^-1 (x-p) <= m^2}, the
// region where the conic keeps a Gaussian's contribution above SUPPORT_T:
// half-width m * sqrt(cov_dd) on each axis. This is the smallest AABB containing the
// ellipsoid (it touches it on all six faces). It replaces the AABB of the ellipsoid's
// oriented box, whose per-axis half-width is the L1 sum m * sum_k |R_dk| s_k -- up to
// sqrt(3) wider per axis (5.2x the volume) -- and which also paired R's rows with the
// scale index where Sigma pairs its columns, so for a few percent of Gaussians it came out
// narrower than the ellipsoid and clipped their support. No rotation matrix is involved
// here, so there is no pairing to get wrong. Forward and backward must both call this so
// the backward sample-BVH query visits exactly the samples the forward pass did.
__forceinline__ __device__ cuBQL::box3f tightGaussianBox(const float3 p, const float cov[6], const float m)
{
	const float3 e = { m * sqrtf(cov[0]), m * sqrtf(cov[3]), m * sqrtf(cov[5]) };
	return cuBQL::box3f(cuBQL::vec3f(p.x - e.x, p.y - e.y, p.z - e.z),
	                    cuBQL::vec3f(p.x + e.x, p.y + e.y, p.z + e.z));
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
