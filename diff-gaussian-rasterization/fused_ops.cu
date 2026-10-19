// The volume loss in three launches: mean (c - g)^2 plus fn_reg times the mean log-barrier
// false-negative term over the cells it is active on.
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace
{

// clamp(log(tau / (w + 0.01 tau)), 0) on cells with ground truth and some coverage.
__device__ __forceinline__ float fnValue(float tau, float w, bool m)
{
	if (!m)
		return 0.0f;
	return fmaxf(logf(tau / (w + 0.01f * tau)), 0.0f);
}

// acc = [sum (c-g)^2, sum fn, count(fn > 0)].
__global__ void lossReduce(long long N, const float* __restrict__ c, const float* __restrict__ g,
	const float* __restrict__ w, float tau, double* __restrict__ acc)
{
	float e = 0.0f, fv = 0.0f, cnt = 0.0f;
	for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < N;
		 i += (long long)gridDim.x * blockDim.x) {
		const float d = c[i] - g[i];
		e += d * d;
		const float v = fnValue(tau, w[i], (g[i] != -1.0f) && (w[i] > 0.0f));
		fv += v;
		cnt += (v > 0.0f) ? 1.0f : 0.0f;
	}
	for (int o = 16; o > 0; o >>= 1) {
		e += __shfl_down_sync(0xffffffff, e, o);
		fv += __shfl_down_sync(0xffffffff, fv, o);
		cnt += __shfl_down_sync(0xffffffff, cnt, o);
	}
	__shared__ float se[32], sf[32], sc[32];
	const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	if (lane == 0) {
		se[warp] = e;
		sf[warp] = fv;
		sc[warp] = cnt;
	}
	__syncthreads();
	if (warp == 0) {
		const int nw = blockDim.x >> 5;
		e = lane < nw ? se[lane] : 0.0f;
		fv = lane < nw ? sf[lane] : 0.0f;
		cnt = lane < nw ? sc[lane] : 0.0f;
		for (int o = 16; o > 0; o >>= 1) {
			e += __shfl_down_sync(0xffffffff, e, o);
			fv += __shfl_down_sync(0xffffffff, fv, o);
			cnt += __shfl_down_sync(0xffffffff, cnt, o);
		}
		if (lane == 0) {
			atomicAdd(&acc[0], (double)e);
			atomicAdd(&acc[1], (double)fv);
			atomicAdd(&acc[2], (double)cnt);
		}
	}
}

// out = [loss, l_v, fn] with l_v = acc0 / N and fn = fn_reg * acc1 / (acc2 + 1e-8).
__global__ void lossEpilogue(
	long long N, const double* __restrict__ acc, float fn_reg, float* __restrict__ out)
{
	const float lv = (float)(acc[0] / (double)N);
	const float fn = fn_reg * (float)acc[1] / ((float)acc[2] + 1e-8f);
	out[0] = lv + fn;
	out[1] = lv;
	out[2] = fn;
}

__global__ void lossBackward(long long N, const float* __restrict__ c, const float* __restrict__ g,
	const float* __restrict__ w, const double* __restrict__ acc, float fn_reg, float tau,
	const float* __restrict__ gout, float* __restrict__ dc, float* __restrict__ dw)
{
	const float kv = (gout[0] + gout[1]) * 2.0f / (float)N;
	const float kf = -(gout[0] + gout[2]) * fn_reg / ((float)acc[2] + 1e-8f);
	for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < N;
		 i += (long long)gridDim.x * blockDim.x) {
		dc[i] = kv * (c[i] - g[i]);
		const bool m = (g[i] != -1.0f) && (w[i] > 0.0f);
		const float d = w[i] + 0.01f * tau;
		// The clamp passes its gradient at equality, as torch's clamp_min does.
		dw[i] = (m && logf(tau / d) >= 0.0f) ? kf / d : 0.0f;
	}
}

void check(const torch::Tensor& t, const char* name)
{
	TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(), name,
		" must be a contiguous float32 CUDA tensor");
}

} // namespace

// Returns out = [loss, l_v, fn] and the double accumulators the backward needs.
std::vector<torch::Tensor> fused_loss_forward(
	torch::Tensor c, torch::Tensor g, torch::Tensor w, double fn_reg, double tau)
{
	check(c, "cells");
	check(g, "gt");
	check(w, "weights");
	TORCH_CHECK(c.numel() == g.numel() && c.numel() == w.numel(), "fused_loss: size mismatch");
	const long long N = c.numel();
	auto acc = torch::zeros({3}, c.options().dtype(torch::kFloat64));
	auto out = torch::empty({3}, c.options());
	const int blocks = (int)std::min<long long>((N + 255) / 256, 108 * 8);
	lossReduce<<<blocks, 256>>>(N, c.data_ptr<float>(), g.data_ptr<float>(), w.data_ptr<float>(),
		(float)tau, acc.data_ptr<double>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	lossEpilogue<<<1, 1>>>(N, acc.data_ptr<double>(), (float)fn_reg, out.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return {out, acc};
}

std::vector<torch::Tensor> fused_loss_backward(torch::Tensor c, torch::Tensor g, torch::Tensor w,
	torch::Tensor acc, double fn_reg, double tau, torch::Tensor gout)
{
	check(c, "cells");
	check(g, "gt");
	check(w, "weights");
	check(gout, "grad");
	const long long N = c.numel();
	auto dc = torch::empty_like(c), dw = torch::empty_like(w);
	const int blocks = (int)std::min<long long>((N + 255) / 256, 108 * 16);
	lossBackward<<<blocks, 256>>>(N, c.data_ptr<float>(), g.data_ptr<float>(), w.data_ptr<float>(),
		acc.data_ptr<double>(), (float)fn_reg, (float)tau, gout.data_ptr<float>(),
		dc.data_ptr<float>(), dw.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return {dc, dw};
}
