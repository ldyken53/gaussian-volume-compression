#include "backward.h"
#include "activation.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void prepareBackwardCUDA(int P, const float* rs, const float* rq,
	const float* __restrict__ raw_values, const float* __restrict__ raw_weights,
	const float max_scale, const float* __restrict__ packed, const float* __restrict__ grad_rows,
	float* __restrict__ dL_dmeans, glm::vec3* dL_dscales, glm::vec4* dL_drots,
	float* __restrict__ dL_dvalues, float* __restrict__ dL_dweights)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// The activations, exactly as prepareCUDA computed them.
	const float es0 = expf(rs[3 * idx]), es1 = expf(rs[3 * idx + 1]), es2 = expf(rs[3 * idx + 2]);
	const float s_nrm = sqrtf(es0 * es0 + es1 * es1 + es2 * es2);
	const float s_r = s_nrm + ACT_CAP_EPS;
	const float s_t = tanhf(s_r / max_scale);
	const float s_f = max_scale * s_t / s_r;
	const glm::vec3 scale(es0 * s_f, es1 * s_f, es2 * s_f);
	const float qx0 = rq[4 * idx], qx1 = rq[4 * idx + 1], qx2 = rq[4 * idx + 2],
				qx3 = rq[4 * idx + 3];
	const float q_n = sqrtf(qx0 * qx0 + qx1 * qx1 + qx2 * qx2 + qx3 * qx3);
	const float q_inv = 1.0f / fmaxf(q_n, ACT_NORM_EPS);
	const float r = qx0 * q_inv, x = qx1 * q_inv, y = qx2 * q_inv, z = qx3 * q_inv;

	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale.x;
	S[1][1] = scale.y;
	S[2][2] = scale.z;
	glm::mat3 R =
		glm::mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
			2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
			2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));
	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	// The conic from the packed record: (mean, value), (weight, cxx, cxy, cxz), (cyy, cyz, czz, -).
	const float* cn = packed + 12 * idx;
	const float* gr = grad_rows + 12 * idx;
	glm::mat3 Co(cn[5], cn[6], cn[7], cn[6], cn[8], cn[9], cn[7], cn[9], cn[10]);
	glm::mat3 G(gr[0], gr[1], gr[2], gr[1], gr[3], gr[4], gr[2], gr[4], gr[5]);
	// The conic is the inverse covariance: dL/dSigma = -C G C.
	glm::mat3 dS = -Co * G * Co;
	const float dcov[6] = {dS[0][0], dS[0][1], dS[0][2], dS[1][1], dS[1][2], dS[2][2]};

	// Through the diagonal epsilon, which follows the largest |Sigma_dd|.
	const float a0 = abs(Sigma[0][0]), a1 = abs(Sigma[1][1]), a2 = abs(Sigma[2][2]);
	const float de0 = (a0 >= a1 && a0 >= a2) ? 1e-5 * glm::sign(Sigma[0][0]) : 0.0f;
	const float de1 = (a1 >= a0 && a1 >= a2) ? 1e-5 * glm::sign(Sigma[1][1]) : 0.0f;
	const float de2 = (a2 >= a0 && a2 >= a1) ? 1e-5 * glm::sign(Sigma[2][2]) : 0.0f;
	const float tr = dcov[0] + dcov[3] + dcov[5];
	glm::mat3 dL_dSigma = glm::mat3(dcov[0] + tr * de0, dcov[1], dcov[2], dcov[1],
		dcov[3] + tr * de1, dcov[4], dcov[2], dcov[4], dcov[5] + tr * de2);
	// Sigma = M^T M, so dL/dM = 2 M dL/dSigma.
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;
	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// The activated scale, then the soft cap and the exp.
	const float gsx = glm::dot(Rt[0], dL_dMt[0]);
	const float gsy = glm::dot(Rt[1], dL_dMt[1]);
	const float gsz = glm::dot(Rt[2], dL_dMt[2]);
	const float fp = ((1.0f - s_t * s_t) * s_r - max_scale * s_t) / (s_r * s_r);
	const float cs = (s_nrm > 0.0f) ? (gsx * es0 + gsy * es1 + gsz * es2) * fp / s_nrm : 0.0f;
	dL_dscales[idx] = glm::vec3(
		(gsx * s_f + cs * es0) * es0, (gsy * s_f + cs * es1) * es1, (gsz * s_f + cs * es2) * es2);

	// The normalized quaternion, then the normalization.
	dL_dMt[0] *= scale.x;
	dL_dMt[1] *= scale.y;
	dL_dMt[2] *= scale.z;
	glm::vec4 dq;
	dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) +
		   2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) +
		   2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) +
		   2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) +
		   2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);
	if (q_n >= ACT_NORM_EPS) {
		const float inv = 1.0f / q_n;
		const float y0 = qx0 * inv, y1 = qx1 * inv, y2 = qx2 * inv, y3 = qx3 * inv;
		const float yg = y0 * dq.x + y1 * dq.y + y2 * dq.z + y3 * dq.w;
		dL_drots[idx] = glm::vec4((dq.x - y0 * yg) * inv, (dq.y - y1 * yg) * inv,
			(dq.z - y2 * yg) * inv, (dq.w - y3 * yg) * inv);
	} else {
		const float inv = 1.0f / ACT_NORM_EPS;
		dL_drots[idx] = glm::vec4(dq.x * inv, dq.y * inv, dq.z * inv, dq.w * inv);
	}

	dL_dmeans[3 * idx] = gr[6];
	dL_dmeans[3 * idx + 1] = gr[7];
	dL_dmeans[3 * idx + 2] = gr[8];
	const float v = act_sigmoid(raw_values[idx]), w = act_sigmoid(raw_weights[idx]);
	dL_dvalues[idx] = gr[9] * v * (1.0f - v);
	dL_dweights[idx] = gr[10] * w * (1.0f - w);
}

// One block per tile; each thread takes one of the tile's Gaussians at a time, sums its gradient
// over the tile's lattice points in registers and adds it to the Gaussian's row once.
__global__ void sampleBackwardCUDA(const dim3 grid, const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ gaussian_ids, const float3 bounds_min, const uint3 cells,
	const float3 cell_size, const float* __restrict__ jitter, const float4* __restrict__ packed,
	const float* __restrict__ out_values, const float* __restrict__ out_influence,
	const float* __restrict__ dL_dout_values, const float* __restrict__ dL_dout_influence,
	float* __restrict__ grad_rows, const float threshold, const float floor)
{
	const uint3 cmin = {blockIdx.x * BLOCK_X, blockIdx.y * BLOCK_Y, blockIdx.z * BLOCK_Z};
	const uint3 cmax = {min(cmin.x + BLOCK_X, cells.x), min(cmin.y + BLOCK_Y, cells.y),
		min(cmin.z + BLOCK_Z, cells.z)};
	const uint32_t bx = cmax.x - cmin.x, by = cmax.y - cmin.y, bz = cmax.z - cmin.z;
	const uint32_t n_cells = bx * by * bz;

	// Per lattice point: its position and a = dL/dV, and b = dL/dI; both 0 where it is undefined.
	__shared__ float4 s_cell[BLOCK_SIZE];
	__shared__ float s_b[BLOCK_SIZE];
	for (uint32_t i = threadIdx.x; i < n_cells; i += blockDim.x) {
		const uint32_t lz = i / (bx * by), ly = (i % (bx * by)) / bx, lx = i % bx;
		const uint3 cell = {cmin.x + lx, cmin.y + ly, cmin.z + lz};
		const uint32_t cell_id = cell.z * cells.x * cells.y + cell.y * cells.x + cell.x;
		const float influence = out_influence[cell_id];
		const bool defined = influence > threshold;
		const float inv_denom = defined ? 1.0f / max(influence, floor) : 0.0f;
		const float a = defined ? dL_dout_values[cell_id] * inv_denom : 0.0f;
		s_cell[i] = make_float4(
			static_cast<float>(cell.x) * cell_size.x + bounds_min.x + jitter[cell_id * 3],
			static_cast<float>(cell.y) * cell_size.y + bounds_min.y + jitter[cell_id * 3 + 1],
			static_cast<float>(cell.z) * cell_size.z + bounds_min.z + jitter[cell_id * 3 + 2], a);
		// Below the floor the denominator is constant and the exact b has no -a value term;
		// keeping it trains better.
		s_b[i] = defined ? dL_dout_influence[cell_id] - a * out_values[cell_id] : 0.0f;
	}
	__syncthreads();

	const uint2 range = ranges[blockIdx.z * grid.y * grid.x + blockIdx.y * grid.x + blockIdx.x];
	const uint32_t n_gauss = range.y - range.x;
	for (uint32_t g = threadIdx.x; g < n_gauss; g += blockDim.x) {
		const int id = gaussian_ids[range.x + g];
		const float4 p0 = packed[3 * id], p1 = packed[3 * id + 1], p2 = packed[3 * id + 2];
		const float3 mean = make_float3(p0.x, p0.y, p0.z);
		const float value = p0.w, w = p1.x, qmax = p2.w;
		const float con_xx = p1.y, con_xy = p1.z, con_xz = p1.w;
		const float con_yy = p2.x, con_yz = p2.y, con_zz = p2.z;
		const float c2xy = 2.0f * con_xy, c2xz = 2.0f * con_xz, c2yz = 2.0f * con_yz;

		// With e = exp(-q/2), dL/dq = -w/2 (a v + b) e.
		float s_v = 0.0f, s_q = 0.0f, Gx = 0.0f, Gy = 0.0f, Gz = 0.0f;
		float xx = 0.0f, xy = 0.0f, xz = 0.0f, yy = 0.0f, yz = 0.0f, zz = 0.0f;
		bool hit = false;
		for (uint32_t c = 0; c < n_cells; c++) {
			const float4 cp = s_cell[c];
			const float dx = cp.x - mean.x, dy = cp.y - mean.y, dz = cp.z - mean.z;
			const float t0 = fmaf(c2xz, dz, fmaf(c2xy, dy, con_xx * dx));
			const float t1 = fmaf(c2yz, dz, con_yy * dy);
			const float q = fmaf(dx, t0, fmaf(dy, t1, con_zz * dz * dz));
			if (!(q >= 0.0f && q <= qmax))
				continue;
			const float e = __expf(-0.5f * q);
			hit = true;
			s_v = fmaf(cp.w, e, s_v);
			const float qe = fmaf(cp.w, value, s_b[c]) * e;
			s_q += qe;
			const float a = qe * dx, b = qe * dy, cz = qe * dz;
			Gx += a;
			Gy += b;
			Gz += cz;
			xx = fmaf(a, dx, xx);
			xy = fmaf(a, dy, xy);
			xz = fmaf(a, dz, xz);
			yy = fmaf(b, dy, yy);
			yz = fmaf(b, dz, yz);
			zz = fmaf(cz, dz, zz);
		}
		// A Gaussian whose support reaches none of the tile's points adds nothing: skip the
		// atomics.
		if (!hit)
			continue;

		// Off-diagonal conic gradients stay undoubled; prepareBackwardCUDA symmetrizes them.
		const float h = -0.5f * w;
		float* gr = grad_rows + 12 * id;
		atomicAdd(gr + 0, h * xx);
		atomicAdd(gr + 1, h * xy);
		atomicAdd(gr + 2, h * xz);
		atomicAdd(gr + 3, h * yy);
		atomicAdd(gr + 4, h * yz);
		atomicAdd(gr + 5, h * zz);
		atomicAdd(gr + 6, w * (con_xx * Gx + con_xy * Gy + con_xz * Gz));
		atomicAdd(gr + 7, w * (con_xy * Gx + con_yy * Gy + con_yz * Gz));
		atomicAdd(gr + 8, w * (con_xz * Gx + con_yz * Gy + con_zz * Gz));
		atomicAdd(gr + 9, w * s_v);
		atomicAdd(gr + 10, s_q);
	}
}

void BACKWARD::sample(dim3 grid, const uint2* ranges, const uint32_t* gaussian_ids,
	float3 bounds_min, uint3 cells, float3 cell_size, const float* jitter, const float4* packed,
	const float* out_values, const float* out_influence, const float* dL_dout_values,
	const float* dL_dout_influence, float* grad_rows, float threshold, float floor)
{
	sampleBackwardCUDA<<<grid, BWD_THREADS>>>(grid, ranges, gaussian_ids, bounds_min, cells,
		cell_size, jitter, packed, out_values, out_influence, dL_dout_values, dL_dout_influence,
		grad_rows, threshold, floor);
}

void BACKWARD::prepare(int P, const float* raw_scales, const float* raw_rotations,
	const float* raw_values, const float* raw_weights, float max_scale, const float4* packed,
	const float* grad_rows, float* dL_dmeans, float* dL_dscales, float* dL_drotations,
	float* dL_dvalues, float* dL_dweights)
{
	prepareBackwardCUDA<<<(P + 255) / 256, 256>>>(P, raw_scales, raw_rotations, raw_values,
		raw_weights, max_scale, (const float*)packed, grad_rows, dL_dmeans, (glm::vec3*)dL_dscales,
		(glm::vec4*)dL_drotations, dL_dvalues, dL_dweights);
}
