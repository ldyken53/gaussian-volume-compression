#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuBQL/traversal/fixedBoxQuery.h>
namespace cg = cooperative_groups;

// dL/dconic -> dL/dscale and dL/drotation (unnormalized quaternion).
__global__ void preprocessCUDA(int P, const glm::vec3* scales, const glm::vec4* rotations,
	const float* conics, glm::vec3* dL_dscales, glm::vec4* dL_drots, float* dL_dconics)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= P)
		return;

	auto scale = scales[idx];
	auto rot = rotations[idx];

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

	const float dL_dconic[6] = {dL_dconics[idx * 6 + 0], dL_dconics[idx * 6 + 1],
		dL_dconics[idx * 6 + 2], dL_dconics[idx * 6 + 3], dL_dconics[idx * 6 + 4],
		dL_dconics[idx * 6 + 5]};
	const float conic[6] = {conics[idx * 6 + 0], conics[idx * 6 + 1], conics[idx * 6 + 2],
		conics[idx * 6 + 3], conics[idx * 6 + 4], conics[idx * 6 + 5]};

	glm::mat3 C(
		conic[0], conic[1], conic[2], conic[1], conic[3], conic[4], conic[2], conic[4], conic[5]);
	glm::mat3 G(dL_dconic[0], dL_dconic[1], dL_dconic[2], dL_dconic[1], dL_dconic[3], dL_dconic[4],
		dL_dconic[2], dL_dconic[4], dL_dconic[5]);

	// C = Sigma^-1, so dL/dSigma = -C G C (G holds per-entry gradients).
	glm::mat3 dL_dSigma_mat = -C * G * C;
	float dL_dcov[6] = {dL_dSigma_mat[0][0], dL_dSigma_mat[0][1], dL_dSigma_mat[0][2],
		dL_dSigma_mat[1][1], dL_dSigma_mat[1][2], dL_dSigma_mat[2][2]};

	// Through the epsilon, which follows the largest diagonal entry.
	float abs_Sigma00 = abs(Sigma[0][0]);
	float abs_Sigma11 = abs(Sigma[1][1]);
	float abs_Sigma22 = abs(Sigma[2][2]);
	float depsilon_dSigma00 = (abs_Sigma00 >= abs_Sigma11 && abs_Sigma00 >= abs_Sigma22)
								  ? 1e-5 * glm::sign(Sigma[0][0])
								  : 0.0f;
	float depsilon_dSigma11 = (abs_Sigma11 >= abs_Sigma00 && abs_Sigma11 >= abs_Sigma22)
								  ? 1e-5 * glm::sign(Sigma[1][1])
								  : 0.0f;
	float depsilon_dSigma22 = (abs_Sigma22 >= abs_Sigma00 && abs_Sigma22 >= abs_Sigma11)
								  ? 1e-5 * glm::sign(Sigma[2][2])
								  : 0.0f;
	glm::mat3 dL_dSigma =
		glm::mat3(dL_dcov[0] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma00,
			dL_dcov[1], dL_dcov[2], dL_dcov[1],
			dL_dcov[3] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma11, dL_dcov[4],
			dL_dcov[2], dL_dcov[4],
			dL_dcov[5] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma22);
	// Sigma = M^T M, so dL/dM = 2 M dL/dSigma.
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= scale.x;
	dL_dMt[1] *= scale.y;
	dL_dMt[2] *= scale.z;
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) +
			  2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) +
			  2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) +
			  2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) +
			  2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w};
}

// One warp per Gaussian over the same box and support test as the forward pass. The bound
// (64 registers) schedules the pair loop ~20% faster than the compiler's own choice.
__global__ void __launch_bounds__(256, 4)
	renderCUDA(int P, const float* means3D, const glm::vec3* scales, const glm::vec4* rotations,
		const float* conics, const float* values, const float* weights, const cuBQL::bvh3f bvh,
		float* dL_dmeans, float* dL_dvalues, float* dL_dweights, float* dL_dconics,
		const float4* __restrict__ sample_pack, const float* __restrict__ sample_fc)
{
	auto warp = cg::tiled_partition<32>(cg::this_thread_block());
	int idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
	int lane = warp.thread_rank();
	if (idx >= P)
		return;

	auto scale = scales[idx];
	auto rot = rotations[idx];
	const float conic[6] = {conics[idx * 6 + 0], conics[idx * 6 + 1], conics[idx * 6 + 2],
		conics[idx * 6 + 3], conics[idx * 6 + 4], conics[idx * 6 + 5]};
	const float3 position = {means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2]};

	float cov[6];
	computeCov3D(scale, rot, cov);
	const float m = truncationRadius(weights[idx]);
	const cuBQL::box3f box = tightGaussianBox(position, cov, m);

	// sample_pack.w = dvo = dL/dR / D and sample_fc = dL/dw_sample - dvo * R (both 0 when
	// uncovered), so with e = exp(-q/2) every term builds from e and qe = (dvo v + fc) e.
	const float w = weights[idx], v = values[idx];
	const float qmax = supportQmax(w);
	const float cxx = conic[0], cxy = conic[1], cxz = conic[2];
	const float cyy = conic[3], cyz = conic[4], czz = conic[5];
	const float c2xy = 2.0f * cxy, c2xz = 2.0f * cxz, c2yz = 2.0f * cyz;

	float s_v = 0.0f, s_q = 0.0f, Gx = 0.0f, Gy = 0.0f, Gz = 0.0f;
	float xx = 0.0f, xy = 0.0f, xz = 0.0f, yy = 0.0f, yz = 0.0f, zz = 0.0f;
	cuBQL::fixedBoxQuery::forEachLeaf<float, 3>(
		[&](uint32_t* primIDs, int count) {
			for (int i = lane; i < count; i += 32) {
				const int primID = primIDs[i];
				const float4 sp = sample_pack[primID];
				const float dx = sp.x - position.x, dy = sp.y - position.y, dz = sp.z - position.z;
				const float t0 = fmaf(c2xz, dz, fmaf(c2xy, dy, cxx * dx));
				const float t1 = fmaf(c2yz, dz, cyy * dy);
				const float q = fmaf(dx, t0, fmaf(dy, t1, czz * dz * dz));
				if (!(q >= 0.0f && q <= qmax))
					continue;
				const float e = expf(-0.5f * q);
				s_v = fmaf(sp.w, e, s_v);
				const float qe = fmaf(sp.w, v, sample_fc[primID]) * e;
				s_q += qe;
				const float ax = qe * dx, ay = qe * dy, az = qe * dz;
				Gx += ax;
				Gy += ay;
				Gz += az;
				xx = fmaf(ax, dx, xx);
				xy = fmaf(ax, dy, xy);
				xz = fmaf(ax, dz, xz);
				yy = fmaf(ay, dy, yy);
				yz = fmaf(ay, dz, yz);
				zz = fmaf(az, dz, zz);
			}
			return 0;
		},
		bvh, box);

	s_v = cg::reduce(warp, s_v, cg::plus<float>());
	s_q = cg::reduce(warp, s_q, cg::plus<float>());
	Gx = cg::reduce(warp, Gx, cg::plus<float>());
	Gy = cg::reduce(warp, Gy, cg::plus<float>());
	Gz = cg::reduce(warp, Gz, cg::plus<float>());
	xx = cg::reduce(warp, xx, cg::plus<float>());
	xy = cg::reduce(warp, xy, cg::plus<float>());
	xz = cg::reduce(warp, xz, cg::plus<float>());
	yy = cg::reduce(warp, yy, cg::plus<float>());
	yz = cg::reduce(warp, yz, cg::plus<float>());
	zz = cg::reduce(warp, zz, cg::plus<float>());

	if (lane == 0) {
		// dL/dquad = h qe with h = -w/2.
		const float h = -0.5f * w;
		dL_dvalues[idx] = w * s_v;
		dL_dweights[idx] = s_q;
		dL_dmeans[idx * 3] = w * (cxx * Gx + cxy * Gy + cxz * Gz);
		dL_dmeans[idx * 3 + 1] = w * (cxy * Gx + cyy * Gy + cyz * Gz);
		dL_dmeans[idx * 3 + 2] = w * (cxz * Gx + cyz * Gy + czz * Gz);
		// Per-entry gradients, not the doubled per-parameter off-diagonals: preprocessCUDA
		// expands them into the symmetric G.
		dL_dconics[idx * 6 + 0] = h * xx;
		dL_dconics[idx * 6 + 1] = h * xy;
		dL_dconics[idx * 6 + 2] = h * xz;
		dL_dconics[idx * 6 + 3] = h * yy;
		dL_dconics[idx * 6 + 4] = h * yz;
		dL_dconics[idx * 6 + 5] = h * zz;
	}
}

// Per-sample terms, computed once per sample instead of once per (Gaussian, sample) pair.
// An empty sample keeps its weight adjoint (the FN/FP terms) but has no value adjoint.
__global__ void samplePrepCUDA(int S, bool hard, const float* __restrict__ samples,
	const float* __restrict__ out_cells, const float* __restrict__ out_weights,
	const float* __restrict__ dL_dsamples, const float* __restrict__ dL_dsample_weights,
	float4* __restrict__ sample_pack, float* __restrict__ sample_fc)
{
	const int s = blockIdx.x * blockDim.x + threadIdx.x;
	if (s >= S)
		return;
	const float aw = out_weights[s];
	const bool covered = aw > (hard ? WEIGHT_CUTOFF : 0.0f);
	// The soft path keeps the (v - R) adjoint at the floored denominator on purpose.
	const float denom = covered ? (hard ? aw : max(aw, WEIGHT_CUTOFF)) : 1.0f;
	const float dvo = covered ? dL_dsamples[s] / denom : 0.0f;
	sample_pack[s] = make_float4(samples[3 * s], samples[3 * s + 1], samples[3 * s + 2], dvo);
	sample_fc[s] = covered ? dL_dsample_weights[s] - dvo * out_cells[s]
						   : (aw > 0.0f ? dL_dsample_weights[s] : 0.0f);
}

void BACKWARD::render(int P, int S, const float* means3D, const glm::vec3* scales,
	const glm::vec4* rotations, const float* conics, const float* values, const float* weights,
	const float* samples, const cuBQL::bvh3f& bvh, bool hard, const float* out_cells,
	const float* out_weights, const float* dL_dsamples, const float* dL_dsample_weights,
	float* dL_dmean3D, float* dL_dvalue, float* dL_dweights, glm::vec3* dL_dscale,
	glm::vec4* dL_drot, float* dL_dconics)
{
	static float4* sample_pack = nullptr;
	static float* sample_fc = nullptr;
	static int prep_cap = 0;
	if (S > prep_cap) {
		if (sample_pack) {
			cudaFree(sample_pack);
			cudaFree(sample_fc);
		}
		cudaMalloc(&sample_pack, sizeof(float4) * S);
		cudaMalloc(&sample_fc, sizeof(float) * S);
		prep_cap = S;
	}
	samplePrepCUDA<<<(S + 255) / 256, 256>>>(S, hard, samples, out_cells, out_weights, dL_dsamples,
		dL_dsample_weights, sample_pack, sample_fc);
	renderCUDA<<<(P * 32 + 255) / 256, 256>>>(P, means3D, scales, rotations, conics, values,
		weights, bvh, dL_dmean3D, dL_dvalue, dL_dweights, dL_dconics, sample_pack, sample_fc);
	preprocessCUDA<<<(P + 255) / 256, 256>>>(
		P, scales, rotations, conics, dL_dscale, dL_drot, dL_dconics);
}
