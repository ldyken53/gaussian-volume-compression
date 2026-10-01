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

// Number of std devs at which a Gaussian of weight w falls to TRUNC_FRAC * WEIGHT_CUTOFF.
// Clamped at 0: for w below that level the log is positive and the sqrt would be NaN. The
// old corner box swallowed the NaN (min/max against the position drop it, leaving a
// zero-size box), but a NaN fed into the tight box makes a box that never rejects, so the
// BVH query visits every sample. m = 0 keeps the old behaviour: such a Gaussian never
// reaches the truncation level, so it gets an empty box.
__forceinline__ __device__ float truncationRadius(const float w)
{
	return sqrtf(fmaxf(0.0f, -2 * logf((TRUNC_FRAC * WEIGHT_CUTOFF) / w)));
}

// Tight axis-aligned bounds of the m-sigma ellipsoid {(x-p)^T cov^-1 (x-p) <= m^2}, the
// region where the conic keeps a Gaussian's contribution above TRUNC_FRAC*WEIGHT_CUTOFF:
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
