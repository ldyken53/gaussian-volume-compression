// The fused volume loss, forward and backward. Matches the torch expressions it replaces,
// including torch's gradient conventions at the edges (clamp_min passes the gradient at
// equality; abs' is 0 at 0).
//
// Loss terms:
//   data  (c-g)^2 or |c-g| (l1). With mask_data, the mean over cells with g != -1 and c != -1
//         (mesh models); else the mean over every cell, an empty cell's -1 included (grid
//         models, whose g is never -1)
//   FN    on (g != -1) & (w > 0): hinge clamp(tau - w, 0) or log barrier
//         clamp(log(tau / (w + 0.01 tau)), 0); scaled by fn_reg, or fn_reg * l_v (rel, detached)
//   FP    (fp_reg > 0) on (g == -1) & (w > 0): w (lin), clamp(log(w / fp_t), 0) (log) or
//         clamp(w - fp_t, 0) (hinge, mean over its violators)
// FN and FP are means over their violators, or over all interior / exterior cells (norm_all).
// rel_pow > 0 scales both by l_v^rel_pow (detached).
#include "ops.h"
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

namespace
{

struct LossCfg {
	int l1, fn_log, rel, fp_kind, norm_all, mask_data;
	float tau, fn_reg, fp_reg, fp_t, rel_pow;
};
__device__ __forceinline__ float hingeScale(const LossCfg& k, float lv)
{
	return k.rel_pow > 0.0f ? powf(fmaxf(lv, 0.0f), k.rel_pow) : 1.0f;
}
__device__ __forceinline__ float fpValue(const LossCfg& k, float w)
{
	return k.fp_kind == 1   ? fmaxf(logf(w / k.fp_t), 0.0f)
		   : k.fp_kind == 2 ? fmaxf(w - k.fp_t, 0.0f)
							: w;
}

__device__ __forceinline__ float fnValue(const LossCfg& k, float w, bool m)
{
	if (!m)
		return 0.0f;
	return k.fn_log ? fmaxf(logf(k.tau / (w + 0.01f * k.tau)), 0.0f) : fmaxf(k.tau - w, 0.0f);
}

// Cells the data term covers.
__device__ __forceinline__ bool dataCell(const LossCfg& k, float c, float g)
{
	return !k.mask_data || (g != -1.0f && c != -1.0f);
}

constexpr int NACC = 6; // [sum err, count data, sum fn, count fn>0, sum fp, count fp>0]

__global__ void lossReduce(long long N, const float* __restrict__ c, const float* __restrict__ g,
	const float* __restrict__ w, LossCfg k, double* __restrict__ acc)
{
	float a[NACC] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < N;
		 i += (long long)gridDim.x * blockDim.x) {
		const float ci = c[i], gi = g[i], wi = w[i];
		const float d = ci - gi;
		if (dataCell(k, ci, gi)) {
			a[0] += k.l1 ? fabsf(d) : d * d;
			a[1] += 1.0f;
		}
		const float v = fnValue(k, wi, (gi != -1.0f) && (wi > 0.0f));
		a[2] += v;
		a[3] += k.norm_all ? ((gi != -1.0f) ? 1.0f : 0.0f) : ((v > 0.0f) ? 1.0f : 0.0f);
		if (k.fp_reg > 0.0f && gi == -1.0f && (wi > 0.0f || k.norm_all)) {
			const float f = wi > 0.0f ? fpValue(k, wi) : 0.0f;
			a[4] += f;
			// linear/log: mean over covered exterior cells; hinge: over its violators;
			// norm_all: over every exterior cell.
			a[5] += (k.norm_all || k.fp_kind != 2 || f > 0.0f) ? 1.0f : 0.0f;
		}
	}
	__shared__ float sh[NACC][32];
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
	for (int j = 0; j < NACC; j++) {
		for (int o = 16; o > 0; o >>= 1)
			a[j] += __shfl_down_sync(0xffffffff, a[j], o);
		if (lane == 0)
			sh[j][warp] = a[j];
	}
	__syncthreads();
	if (warp == 0) {
		const int nw = blockDim.x >> 5;
#pragma unroll
		for (int j = 0; j < NACC; j++) {
			float x = lane < nw ? sh[j][lane] : 0.0f;
			for (int o = 16; o > 0; o >>= 1)
				x += __shfl_down_sync(0xffffffff, x, o);
			if (lane == 0)
				atomicAdd(&acc[j], (double)x);
		}
	}
}

__device__ __forceinline__ float dataDenom(const double* acc) { return fmaxf((float)acc[1], 1.0f); }

// out = [loss, l_v, fn, fp].
__global__ void lossEpilogue(
	long long N, const double* __restrict__ acc, LossCfg k, float* __restrict__ out)
{
	const float lv = (float)(acc[0] / (double)dataDenom(acc));
	const float hs = hingeScale(k, lv);
	const float scale = (k.rel ? k.fn_reg * lv : k.fn_reg) * hs;
	const float fn = scale * (float)acc[2] / ((float)acc[3] + 1e-8f);
	const float fp =
		k.fp_reg > 0.0f ? hs * k.fp_reg * (float)acc[4] / ((float)acc[5] + 1e-8f) : 0.0f;
	out[0] = lv + fn + fp;
	out[1] = lv;
	out[2] = fn;
	out[3] = fp;
}

__global__ void lossBackward(long long N, const float* __restrict__ c, const float* __restrict__ g,
	const float* __restrict__ w, const double* __restrict__ acc, LossCfg k,
	const float* __restrict__ gout, float* __restrict__ dc, float* __restrict__ dw)
{
	// out = [l_v + fn + fp, l_v, fn, fp], so each component's gradient feeds its terms.
	// rel's l_v factor is detached, as in train.py.
	const float denom = dataDenom(acc);
	const float kv = (gout[0] + gout[1]) / denom;
	const float lv = (float)(acc[0] / (double)denom);
	const float hs = hingeScale(k, lv);
	const float scale = (k.rel ? k.fn_reg * lv : k.fn_reg) * hs;
	const float kf = -(gout[0] + gout[2]) * scale / ((float)acc[3] + 1e-8f);
	const float kp = (gout[0] + gout[3]) * hs * k.fp_reg / ((float)acc[5] + 1e-8f);
	for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < N;
		 i += (long long)gridDim.x * blockDim.x) {
		const float ci = c[i], gi = g[i], wi = w[i];
		const float d = ci - gi;
		// abs'(0) = 0, as torch.
		dc[i] =
			dataCell(k, ci, gi) ? kv * (k.l1 ? (float)((d > 0.0f) - (d < 0.0f)) : 2.0f * d) : 0.0f;
		// clamp_min's gradient passes where its input >= min (torch convention).
		const bool m = (gi != -1.0f) && (wi > 0.0f);
		float gw = 0.0f;
		if (m) {
			if (k.fn_log) {
				const float dd = wi + 0.01f * k.tau; // d/dw log(tau/d) = -1/d
				if (logf(k.tau / dd) >= 0.0f)
					gw = kf / dd;
			} else if (k.tau - wi >= 0.0f) {
				gw = kf;
			}
		}
		if (k.fp_reg > 0.0f && gi == -1.0f && wi > 0.0f) {
			if (k.fp_kind == 0)
				gw += kp;
			else if (k.fp_kind == 1) {
				if (logf(wi / k.fp_t) >= 0.0f)
					gw += kp / wi;
			} else if (wi - k.fp_t >= 0.0f)
				gw += kp;
		}
		dw[i] = gw;
	}
}

void check(const torch::Tensor& t, const char* name)
{
	TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(), name,
		" must be a contiguous float32 CUDA tensor");
}

LossCfg makeCfg(int64_t l1, int64_t fn_log, int64_t rel, double tau, double fn_reg, double fp_reg,
	double fp_t, int64_t fp_kind, int64_t norm_all, double rel_pow, int64_t mask_data)
{
	return LossCfg{(int)l1, (int)fn_log, (int)rel, (int)fp_kind, (int)norm_all, (int)mask_data,
		(float)tau, (float)fn_reg, (float)fp_reg, (float)fp_t, (float)rel_pow};
}

int numBlocks(long long N, int per_sm)
{
	int dev = 0, sms = 108;
	cudaGetDevice(&dev);
	cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
	return (int)std::max<long long>(
		1, std::min<long long>((N + 255) / 256, (long long)sms * per_sm));
}

} // namespace

// Returns (out[4] = [loss, l_v, fn, fp], acc[6] doubles kept for the backward).
std::vector<torch::Tensor> fused_loss_forward(torch::Tensor c, torch::Tensor g, torch::Tensor w,
	int64_t l1, int64_t fn_log, int64_t rel, double tau, double fn_reg, double fp_reg, double fp_t,
	int64_t fp_kind, int64_t norm_all, double rel_pow, int64_t mask_data)
{
	check(c, "cells");
	check(g, "gt");
	check(w, "weights");
	TORCH_CHECK(c.numel() == g.numel() && c.numel() == w.numel(), "fused_loss: size mismatch");
	const long long N = c.numel();
	const LossCfg k =
		makeCfg(l1, fn_log, rel, tau, fn_reg, fp_reg, fp_t, fp_kind, norm_all, rel_pow, mask_data);
	auto acc = torch::zeros({NACC}, c.options().dtype(torch::kFloat64));
	auto out = torch::empty({4}, c.options());
	lossReduce<<<numBlocks(N, 8), 256>>>(N, c.data_ptr<float>(), g.data_ptr<float>(),
		w.data_ptr<float>(), k, acc.data_ptr<double>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	lossEpilogue<<<1, 1>>>(N, acc.data_ptr<double>(), k, out.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return {out, acc};
}

std::vector<torch::Tensor> fused_loss_backward(torch::Tensor c, torch::Tensor g, torch::Tensor w,
	torch::Tensor acc, int64_t l1, int64_t fn_log, int64_t rel, double tau, double fn_reg,
	double fp_reg, double fp_t, int64_t fp_kind, int64_t norm_all, double rel_pow,
	int64_t mask_data, torch::Tensor gout)
{
	check(c, "cells");
	check(g, "gt");
	check(w, "weights");
	check(gout, "grad");
	const long long N = c.numel();
	const LossCfg k =
		makeCfg(l1, fn_log, rel, tau, fn_reg, fp_reg, fp_t, fp_kind, norm_all, rel_pow, mask_data);
	auto dc = torch::empty_like(c), dw = torch::empty_like(w);
	lossBackward<<<numBlocks(N, 16), 256>>>(N, c.data_ptr<float>(), g.data_ptr<float>(),
		w.data_ptr<float>(), acc.data_ptr<double>(), k, gout.data_ptr<float>(),
		dc.data_ptr<float>(), dw.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return {dc, dw};
}
