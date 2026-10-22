#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void preprocessCUDA(int P, const float* means3D, const float* rs, const float* rq,
	const float* values, const float* weights, const float max_scale, const float3 volume_mins,
	const float3 cell_size, const dim3 grid, float4* packed, uint* aabbs, uint32_t* blocks_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Activations: exp + soft norm cap on the scales, normalized quaternion, sigmoids.
	const float s0 = expf(rs[3 * idx]), s1 = expf(rs[3 * idx + 1]), s2 = expf(rs[3 * idx + 2]);
	const float sr = sqrtf(s0 * s0 + s1 * s1 + s2 * s2) + ACT_CAP_EPS;
	const float sf = max_scale * tanhf(sr / max_scale) / sr;
	const glm::vec3 scale(s0 * sf, s1 * sf, s2 * sf);
	const float q0 = rq[4 * idx], q1 = rq[4 * idx + 1], q2 = rq[4 * idx + 2], q3 = rq[4 * idx + 3];
	const float qinv = 1.0f / fmaxf(sqrtf(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3), ACT_NORM_EPS);
	const float r = q0 * qinv, x = q1 * qinv, y = q2 * qinv, z = q3 * qinv;
	const float act_v = act_sigmoid(values[idx]), act_w = act_sigmoid(weights[idx]);

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

	const float epsilon = max(max(abs(Sigma[0][0]), abs(Sigma[1][1])), abs(Sigma[2][2])) * 1e-5;
	const float a = Sigma[0][0] + epsilon, b = Sigma[0][1], c = Sigma[0][2];
	const float d = Sigma[1][1] + epsilon, e = Sigma[1][2], f = Sigma[2][2] + epsilon;
	const float det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
	const float det_inv = 1.0 / det;
	const float cn[6] = {(d * f - e * e) * det_inv, (c * e - b * f) * det_inv,
		(b * e - c * d) * det_inv, (a * f - c * c) * det_inv, (b * c - a * e) * det_inv,
		(a * d - b * b) * det_inv};

	// Support radius in sigmas: w * exp(-m^2 / 2) = SUPPORT_T. NaN (no support) when w < SUPPORT_T.
	const float m = sqrtf(-2 * logf(SUPPORT_T / act_w));

	const float3 pos = {means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2]};
	packed[3 * idx + 0] = make_float4(pos.x, pos.y, pos.z, act_v);
	packed[3 * idx + 1] = make_float4(act_w, cn[0], cn[1], cn[2]);
	packed[3 * idx + 2] = make_float4(cn[3], cn[4], cn[5], m * m);

	// Axis-aligned bounds of the support ellipsoid: m * sqrt(Sigma_dd) per axis.
	const float3 ext = {m * sqrtf(a), m * sqrtf(d), m * sqrtf(f)};
	const float3 lo = {pos.x - ext.x, pos.y - ext.y, pos.z - ext.z};
	const float3 hi = {pos.x + ext.x, pos.y + ext.y, pos.z + ext.z};
	const float bx = cell_size.x * BLOCK_X, by = cell_size.y * BLOCK_Y, bz = cell_size.z * BLOCK_Z;
	const uint3 start =
		make_uint3(max(0, min((int)grid.x, (int)floor((lo.x - volume_mins.x) / bx))),
			max(0, min((int)grid.y, (int)floor((lo.y - volume_mins.y) / by))),
			max(0, min((int)grid.z, (int)floor((lo.z - volume_mins.z) / bz))));
	const uint3 end = make_uint3(max(0, min((int)grid.x, (int)ceil((hi.x - volume_mins.x) / bx))),
		max(0, min((int)grid.y, (int)ceil((hi.y - volume_mins.y) / by))),
		max(0, min((int)grid.z, (int)ceil((hi.z - volume_mins.z) / bz))));
	blocks_touched[idx] = (end.x - start.x) * (end.y - start.y) * (end.z - start.z);
	aabbs[idx * 6 + 0] = start.x;
	aabbs[idx * 6 + 1] = start.y;
	aabbs[idx * 6 + 2] = start.z;
	aabbs[idx * 6 + 3] = end.x;
	aabbs[idx * 6 + 4] = end.y;
	aabbs[idx * 6 + 5] = end.z;
}

// One block per tile, one thread per cell.
__global__ void __launch_bounds__(BLOCK_SIZE)
	renderCUDA(const uint2* __restrict__ ranges, const uint32_t* __restrict__ point_list,
		const dim3 grid, const float3 volume_mins, const uint3 num_cells, const float3 cell_size,
		const float* __restrict__ jitter, const float4* __restrict__ packed,
		float* __restrict__ out_weights, float* __restrict__ out_cells, const float occ_t,
		const float floor)
{
	auto block = cg::this_thread_block();
	const uint3 cell = {block.group_index().x * BLOCK_X + block.thread_index().x,
		block.group_index().y * BLOCK_Y + block.thread_index().y,
		block.group_index().z * BLOCK_Z + block.thread_index().z};
	const uint32_t cell_id = cell.z * num_cells.x * num_cells.y + cell.y * num_cells.x + cell.x;
	const float3 cell_pos =
		make_float3(static_cast<float>(cell.x) * cell_size.x + volume_mins.x + jitter[cell_id * 3],
			static_cast<float>(cell.y) * cell_size.y + volume_mins.y + jitter[cell_id * 3 + 1],
			static_cast<float>(cell.z) * cell_size.z + volume_mins.z + jitter[cell_id * 3 + 2]);
	const bool inside = cell.x < num_cells.x && cell.y < num_cells.y && cell.z < num_cells.z;
	const bool done = !inside;

	const uint2 range = ranges[block.group_index().z * grid.y * grid.x +
							   block.group_index().y * grid.x + block.group_index().x];
	const int rounds = (range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int todo = range.y - range.x;

	// Per Gaussian: (mean, value), (w, cxx, 2cxy, 2cxz), (cyy, 2cyz, czz, qmax).
	__shared__ float4 collected[3 * BLOCK_SIZE];
	float acc_w = 0.0f, acc_v = 0.0f;
	for (int i = 0; i < rounds; i++, todo -= BLOCK_SIZE) {
		if (__syncthreads_count(done) == BLOCK_SIZE)
			break;
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y) {
			const int id = point_list[range.x + progress];
			const int rank = block.thread_rank();
			const float4 p1 = packed[3 * id + 1], p2 = packed[3 * id + 2];
			collected[3 * rank + 0] = packed[3 * id + 0];
			collected[3 * rank + 1] = make_float4(p1.x, p1.y, 2.0f * p1.z, 2.0f * p1.w);
			collected[3 * rank + 2] = make_float4(p2.x, 2.0f * p2.y, p2.z, p2.w);
		}
		block.sync();
		for (int j = 0; !done && j < min(BLOCK_SIZE, todo); j++) {
			const float4 p0 = collected[3 * j], p1 = collected[3 * j + 1],
						 p2 = collected[3 * j + 2];
			const float dx = cell_pos.x - p0.x, dy = cell_pos.y - p0.y, dz = cell_pos.z - p0.z;
			const float t0 = fmaf(p1.w, dz, fmaf(p1.z, dy, p1.y * dx));
			const float t1 = fmaf(p2.y, dz, p2.x * dy);
			const float q = fmaf(dx, t0, fmaf(dy, t1, p2.z * dz * dz));
			if (!(q >= 0.0f && q <= p2.w))
				continue;
			const float weight = p1.x * __expf(-0.5f * q);
			acc_v += p0.w * weight;
			acc_w += weight;
		}
	}

	if (inside) {
		if (acc_w > occ_t) {
			out_cells[cell_id] = acc_v / max(acc_w, floor);
			out_weights[cell_id] = acc_w;
		} else {
			out_cells[cell_id] = -1.0f;
			out_weights[cell_id] = 0.0f;
		}
	}
}

void FORWARD::preprocess(int P, const float* means3D, const float* scales, const float* rotations,
	const float* values, const float* weights, float max_scale, float3 volume_mins,
	float3 cell_size, dim3 grid, float4* packed, uint* aabbs, uint32_t* blocks_touched)
{
	preprocessCUDA<<<(P + 255) / 256, 256>>>(P, means3D, scales, rotations, values, weights,
		max_scale, volume_mins, cell_size, grid, packed, aabbs, blocks_touched);
}

void FORWARD::render(dim3 grid, dim3 block, const uint2* ranges, const uint32_t* point_list,
	float3 volume_mins, uint3 num_cells, float3 cell_size, const float* jitter,
	const float4* packed, float* out_weights, float* out_cells, float occ_t, float floor)
{
	renderCUDA<<<grid, block>>>(ranges, point_list, grid, volume_mins, num_cells, cell_size, jitter,
		packed, out_weights, out_cells, occ_t, floor);
}
