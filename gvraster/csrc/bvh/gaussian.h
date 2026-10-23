#pragma once

// A Gaussian reaches a point only where w G >= SUPPORT_T.
#define SUPPORT_T 1e-4f

#include <cuBQL/bvh.h>
#include <glm/glm.hpp>

// The covariance with a relative epsilon on its diagonal; the conic and both passes' support
// boxes derive from it.
__forceinline__ __device__ void computeCov3D(
	const glm::vec3 scale, const glm::vec4 rot, float cov[6])
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;

	float r = rot.x;
	float x = rot.y;
	float y = rot.z;
	float z = rot.w;
	glm::mat3 R =
		glm::mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
			2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
			2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	const float epsilon = max(max(abs(Sigma[0][0]), abs(Sigma[1][1])), abs(Sigma[2][2])) * 1e-5;
	cov[0] = Sigma[0][0] + epsilon;
	cov[1] = Sigma[0][1];
	cov[2] = Sigma[0][2];
	cov[3] = Sigma[1][1] + epsilon;
	cov[4] = Sigma[1][2];
	cov[5] = Sigma[2][2] + epsilon;
}

// w exp(-q/2) >= SUPPORT_T exactly where q <= qmax.
__forceinline__ __device__ float supportQmax(const float w) { return 2.0f * logf(w / SUPPORT_T); }

// The support radius in standard deviations; 0 rather than NaN when w < SUPPORT_T.
__forceinline__ __device__ float supportRadius(const float w)
{
	return sqrtf(fmaxf(0.0f, supportQmax(w)));
}

// The axis-aligned box of the m-sigma ellipsoid: half-width m sqrt(cov_dd) per axis.
__forceinline__ __device__ cuBQL::box3f supportBox(
	const float3 p, const float cov[6], const float m)
{
	const float3 e = {m * sqrtf(cov[0]), m * sqrtf(cov[3]), m * sqrtf(cov[5])};
	return cuBQL::box3f(cuBQL::vec3f(p.x - e.x, p.y - e.y, p.z - e.z),
		cuBQL::vec3f(p.x + e.x, p.y + e.y, p.z + e.z));
}
