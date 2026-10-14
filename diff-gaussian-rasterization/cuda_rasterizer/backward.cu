#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float sq(float x) { return x * x; }

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float* conics,
	const float scale_modifier,
	const float* dL_dconics,
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	auto scale = scales[idx];
	auto rot = rotations[idx];
	
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale_modifier * scale.x;
	S[1][1] = scale_modifier * scale.y;
	S[2][2] = scale_modifier * scale.z;

	// Normalize quaternion to get valid rotation (commented out for some reason?)
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	float conic[6] = {
		conics[6 * idx],
		conics[6 * idx + 1],
		conics[6 * idx + 2],
		conics[6 * idx + 3],
		conics[6 * idx + 4],
		conics[6 * idx + 5],
	};
	float dL_dconic[6] = {
		dL_dconics[6 * idx],
		dL_dconics[6 * idx + 1],
		dL_dconics[6 * idx + 2],
		dL_dconics[6 * idx + 3],
		dL_dconics[6 * idx + 4],
		dL_dconics[6 * idx + 5],
	};

	// // Compute dL_dcov as -conic * dL_dconic * conic
	// // since conic is inverse of cov
	// const float dL_dconic_conic[6] = {
	// 	dL_dconic[0]*conic[0] + dL_dconic[1]*conic[1] + dL_dconic[2]*conic[2],
    // 	dL_dconic[0]*conic[1] + dL_dconic[1]*conic[3] + dL_dconic[2]*conic[4],
    // 	dL_dconic[0]*conic[2] + dL_dconic[1]*conic[4] + dL_dconic[2]*conic[5],
    // 	dL_dconic[1]*conic[1] + dL_dconic[3]*conic[3] + dL_dconic[4]*conic[4],
    // 	dL_dconic[1]*conic[2] + dL_dconic[3]*conic[4] + dL_dconic[4]*conic[5],
    // 	dL_dconic[2]*conic[2] + dL_dconic[4]*conic[4] + dL_dconic[5]*conic[5]
	// };
	// const float dL_dcov[6] = {
	// 	-1.0f * (conic[0]*dL_dconic_conic[0] + conic[1]*dL_dconic_conic[1] + conic[2]*dL_dconic_conic[2]),
    // 	-1.0f * (conic[0]*dL_dconic_conic[1] + conic[1]*dL_dconic_conic[3] + conic[2]*dL_dconic_conic[4]),
    // 	-1.0f * (conic[0]*dL_dconic_conic[2] + conic[1]*dL_dconic_conic[4] + conic[2]*dL_dconic_conic[5]),
    // 	-1.0f * (conic[1]*dL_dconic_conic[1] + conic[3]*dL_dconic_conic[3] + conic[4]*dL_dconic_conic[4]),
    // 	-1.0f * (conic[1]*dL_dconic_conic[2] + conic[3]*dL_dconic_conic[4] + conic[4]*dL_dconic_conic[5]),
    // 	-1.0f * (conic[2]*dL_dconic_conic[2] + conic[4]*dL_dconic_conic[4] + conic[5]*dL_dconic_conic[5])
	// };
	// Unpack 'conic' and dL_dconic into full 3x3 matrices
	glm::mat3 Co(
		conic[0], conic[1], conic[2],
		conic[1], conic[3], conic[4],
		conic[2], conic[4], conic[5]
	);

	glm::mat3 G(
		dL_dconic[0], dL_dconic[1], dL_dconic[2],
		dL_dconic[1], dL_dconic[3], dL_dconic[4],
		dL_dconic[2], dL_dconic[4], dL_dconic[5]
	);

	// dL/dSigma = -C * G * C
	glm::mat3 dL_dSigma_mat = -Co * G * Co;

	// Repack to 6 parameters (xx, xy, xz, yy, yz, zz)
	float dL_dcov[6] = {
		dL_dSigma_mat[0][0],
		dL_dSigma_mat[0][1],
		dL_dSigma_mat[0][2],
		dL_dSigma_mat[1][1],
		dL_dSigma_mat[1][2],
		dL_dSigma_mat[2][2]
	};

	float abs_Sigma00 = abs(Sigma[0][0]);
	float abs_Sigma11 = abs(Sigma[1][1]);
	float abs_Sigma22 = abs(Sigma[2][2]);
	float depsilon_dSigma00 = (abs_Sigma00 >= abs_Sigma11 && abs_Sigma00 >= abs_Sigma22) ? 1e-5 * glm::sign(Sigma[0][0]) : 0.0f;
	float depsilon_dSigma11 = (abs_Sigma11 >= abs_Sigma00 && abs_Sigma11 >= abs_Sigma22) ? 1e-5 * glm::sign(Sigma[1][1]) : 0.0f;
	float depsilon_dSigma22 = (abs_Sigma22 >= abs_Sigma00 && abs_Sigma22 >= abs_Sigma11) ? 1e-5 * glm::sign(Sigma[2][2]) : 0.0f;
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov[0] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma00, dL_dcov[1], dL_dcov[2],
		dL_dcov[1], dL_dcov[3] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma11, dL_dcov[4],
		dL_dcov[2], dL_dcov[4], dL_dcov[5] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma22
	);
	// Sigma = M^T M  =>  dL_dM = 2 * M * dL_dSigma  (NOT dL_dSigma * M: dL_dSigma is
	// symmetric but M is not, and the two orders only agree when R == I, which is why
	// the old form was correct at init and drifted wrong as rotations moved.)
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = scale_modifier * glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = scale_modifier * glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = scale_modifier * glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= scale_modifier * scale.x;
	dL_dMt[1] *= scale_modifier * scale.y;
	dL_dMt[2] *= scale_modifier * scale.z;
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);
	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward version of the rendering procedure.
// Each thread handles one Gaussian that intersects this block,
// accumulating gradient contributions from all cells in the block
// before writing to global memory with a single set of atomicAdds.
template <uint32_t C>
__global__ void renderCUDA(
	const dim3 grid,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float3 volume_mins,
	const uint3 num_cells,
	const float3 cell_size,
	const float* __restrict__ jitter,
	const float4* __restrict__ packed,
	const float* __restrict__ out_cells,
	const float* __restrict__ accumulated_weights,
	const float* __restrict__ dL_dcells,
	const float* __restrict__ dL_dcell_weights,
	float3* __restrict__ dL_dmeans,
	float* __restrict__ dL_dconic,
	float* __restrict__ dL_dvalues,
	float* __restrict__ dL_dweights
)
{
	// Block tile boundaries in cell space
	uint3 cell_min = { blockIdx.x * BLOCK_X, blockIdx.y * BLOCK_Y, blockIdx.z * BLOCK_Z };
	uint3 cell_max = { min(cell_min.x + BLOCK_X, num_cells.x), min(cell_min.y + BLOCK_Y, num_cells.y), min(cell_min.z + BLOCK_Z, num_cells.z) };
	uint32_t bx = cell_max.x - cell_min.x;
	uint32_t by = cell_max.y - cell_min.y;
	uint32_t bz = cell_max.z - cell_min.z;
	uint32_t num_cells_in_block = bx * by * bz;

	// Per cell: position and dvo = dL/dR * 1/D, plus fc = dL/dweight_cell - dvo * R.
	// Every per-(Gaussian, cell) term below is built from these two, and uncovered
	// cells are stored as dvo = fc = 0, so they contribute exactly zero without a branch.
	__shared__ float4 s_cell[BLOCK_X * BLOCK_Y * BLOCK_Z];  // (x, y, z, dvo)
	__shared__ float  s_fc[BLOCK_X * BLOCK_Y * BLOCK_Z];

	for (uint32_t idx = threadIdx.x; idx < num_cells_in_block; idx += blockDim.x)
	{
		uint32_t lz = idx / (bx * by);
		uint32_t ly = (idx % (bx * by)) / bx;
		uint32_t lx = idx % bx;

		uint3 cell = { cell_min.x + lx, cell_min.y + ly, cell_min.z + lz };
		uint32_t cell_id = cell.z * num_cells.x * num_cells.y + cell.y * num_cells.x + cell.x;

		float aw = accumulated_weights[cell_id];
#if SOFT_CUTOFF
		// Bounded at 1/WEIGHT_CUTOFF instead of cut to zero, so weakly covered cells emit a
		// gradient without the 1/aw amplification the threshold was there to prevent.
		const bool covered = aw > 0.0f;
		float inv_aw = covered ? 1.0f / SOFT_DENOM(aw) : 0.0f;
#else
		const bool covered = aw > WEIGHT_CUTOFF;
		float inv_aw = covered ? 1.0f / aw : 0.0f;
#endif
		// dR/dw_i = (v_i - R * dD/daw) / D. Above the cutoff dD/daw == 1 and this is exact.
		// Below it D = max(aw, c) is constant, so the true adjoint is v_i / D; keeping the
		// -R term anyway is deliberate and measured better (see SOFT_DENOM in config.h):
		// it pushes w_i up where v_i > R and down where v_i < R, instead of pushing every
		// weight up uniformly.
		const float r_term = out_cells[cell_id];
		const float dvo = covered ? dL_dcells[cell_id] * inv_aw : 0.0f;
		s_cell[idx] = make_float4(
			static_cast<float>(cell.x) * cell_size.x + volume_mins.x + jitter[cell_id * 3],
			static_cast<float>(cell.y) * cell_size.y + volume_mins.y + jitter[cell_id * 3 + 1],
			static_cast<float>(cell.z) * cell_size.z + volume_mins.z + jitter[cell_id * 3 + 2],
			dvo);
		s_fc[idx] = covered ? dL_dcell_weights[cell_id] - dvo * r_term : 0.0f;
	}
	__syncthreads();

	// Load range of Gaussians for this block
	uint2 range = ranges[blockIdx.z * grid.y * grid.x + blockIdx.y * grid.x + blockIdx.x];
	uint32_t num_gaussians = range.y - range.x;

	// Each thread processes one (or more) Gaussians, striding by blockDim.x
	for (uint32_t g = threadIdx.x; g < num_gaussians; g += blockDim.x)
	{
		int point_idx = point_list[range.x + g];

		const float4 p0 = packed[3 * point_idx], p1 = packed[3 * point_idx + 1], p2 = packed[3 * point_idx + 2];
		const float3 mean = make_float3(p0.x, p0.y, p0.z);
		const float value = p0.w, w = p1.x;
		const float con_xx = p1.y, con_xy = p1.z, con_xz = p1.w;
		const float con_yy = p2.x, con_yz = p2.y, con_zz = p2.z;
		const float c2xy = 2.0f * con_xy, c2xz = 2.0f * con_xz, c2yz = 2.0f * con_yz;

		// With e = exp(-q/2), F = dvo*(v - R) + dL/dw_cell = dvo*v + fc and
		// dL/dquad = -w/2 * F * e, the per-cell sums need only e and qe = F * e; w and the
		// -1/2 are applied once after the loop.
		float s_v = 0.0f, s_q = 0.0f, Gx = 0.0f, Gy = 0.0f, Gz = 0.0f;
		float xx = 0.0f, xy = 0.0f, xz = 0.0f, yy = 0.0f, yz = 0.0f, zz = 0.0f;
		for (uint32_t c = 0; c < num_cells_in_block; c++)
		{
			const float4 cp = s_cell[c];
			const float dx = cp.x - mean.x, dy = cp.y - mean.y, dz = cp.z - mean.z;
			const float t0 = fmaf(c2xz, dz, fmaf(c2xy, dy, con_xx * dx));
			const float t1 = fmaf(c2yz, dz, con_yy * dy);
			const float q  = fmaf(dx, t0, fmaf(dy, t1, con_zz * dz * dz));
			if (!(q >= 0.0f && q <= 28.0f)) continue;
			const float e = __expf(-0.5f * q);
			s_v = fmaf(cp.w, e, s_v);
			const float qe = fmaf(cp.w, value, s_fc[c]) * e;
			s_q += qe;
			const float a = qe * dx, b = qe * dy, cz = qe * dz;
			Gx += a; Gy += b; Gz += cz;
			xx = fmaf(a, dx, xx); xy = fmaf(a, dy, xy); xz = fmaf(a, dz, xz);
			yy = fmaf(b, dy, yy); yz = fmaf(b, dz, yz); zz = fmaf(cz, dz, zz);
		}

		const float h = -0.5f * w;
		const float acc_dL_dvalue = w * s_v;
		// = -2 * sum(dL/dquad) / w, without the division.
		const float acc_dL_dw = s_q;
		// Off-diagonals keep the undoubled convention sum(dL/dquad * dx * dy); backward
		// preprocessing applies the symmetric doubling.
		const float acc_dL_dxx = h * xx, acc_dL_dxy = h * xy, acc_dL_dxz = h * xz;
		const float acc_dL_dyy = h * yy, acc_dL_dyz = h * yz, acc_dL_dzz = h * zz;
		// dquad/dmean = -2 conic d, so dL/dmean = -2 conic sum(dL/dquad * d) = w conic G.
		const float dL_dmean_x = w * (con_xx * Gx + con_xy * Gy + con_xz * Gz);
		const float dL_dmean_y = w * (con_xy * Gx + con_yy * Gy + con_yz * Gz);
		const float dL_dmean_z = w * (con_xz * Gx + con_yz * Gy + con_zz * Gz);

		// Single atomic write per Gaussian — all cell contributions already accumulated
		atomicAdd(&dL_dvalues[point_idx], acc_dL_dvalue);
		atomicAdd(&dL_dweights[point_idx], acc_dL_dw);

		atomicAdd(&dL_dmeans[point_idx].x, dL_dmean_x);
		atomicAdd(&dL_dmeans[point_idx].y, dL_dmean_y);
		atomicAdd(&dL_dmeans[point_idx].z, dL_dmean_z);

		atomicAdd(&dL_dconic[point_idx * 6 + 0], acc_dL_dxx);
		atomicAdd(&dL_dconic[point_idx * 6 + 1], acc_dL_dxy);
		atomicAdd(&dL_dconic[point_idx * 6 + 2], acc_dL_dxz);
		atomicAdd(&dL_dconic[point_idx * 6 + 3], acc_dL_dyy);
		atomicAdd(&dL_dconic[point_idx * 6 + 4], acc_dL_dyz);
		atomicAdd(&dL_dconic[point_idx * 6 + 5], acc_dL_dzz);
	}
}

void BACKWARD::preprocess(
	int P,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float* conics,
	const float scale_modifier,
	const float* dL_dconic,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for remaining steps: using dL_dconics
	// propagate back to scales and rotations
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P,
		radii,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		conics,
		scale_modifier,
		dL_dconic,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const float3 volume_mins,
	const uint3 num_cells,
	const float3 cell_size,
	const float* jitter,
	const bool* clamped,
	const float4* packed,
	const float* out_cells,
	const float* volumes,
	const float* accumulated_weights,
	const uint32_t* n_contrib,
	const float* dL_dcells,
	const float* dL_dcell_weights,
	float3* dL_dmean3D,
	float* dL_dconic,
	float* dL_dvalue,
	float* dL_dweights
)
{
	renderCUDA<NUM_CHANNELS> << <grid, 128 >> >(
		grid,
		ranges,
		point_list,
		volume_mins,
		num_cells,
		cell_size,
		jitter,
		packed,
		out_cells,
		accumulated_weights,
		dL_dcells,
		dL_dcell_weights,
		dL_dmean3D,
		dL_dconic,
		dL_dvalue,
		dL_dweights
	);
}
