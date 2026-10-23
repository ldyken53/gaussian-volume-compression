// The training loss, forward and backward. It matches the torch expressions it replaces,
// including torch's gradient conventions at the edges (clamp_min passes the gradient at equality;
// abs' is 0 at 0). Per sample, with value c, ground truth g and influence I (-1 marks undefined):
//   data  (c - g)^2, or |c - g| with l1. With mask_data, the mean over the samples where c and g
//         are defined (meshes); else over every sample, an undefined c's -1 included (grids,
//         whose g is always defined)
//   FN    where g is defined and I > 0: the hinge max(fn_tau - I, 0) or the log barrier
//         max(log(fn_tau / (I + 0.01 fn_tau)), 0), times lambda_fn, or lambda_fn * data (rel)
//   FP    (lambda_fp > 0) where g is undefined and I > 0: I (lin), max(log(I / fp_tau), 0) (log)
//         or max(I - fp_tau, 0) (hinge), times lambda_fp
// FN is the mean over its violators and FP over every sample it covers (hinge: its violators),
// or over every sample where g is defined / undefined (norm_all). rel_pow > 0 scales both by
// data^rel_pow. The data factors are detached.
#include "ops.h"
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

namespace
{

struct LossCfg {
	int l1, fn_log, rel, fp_mode, norm_all, mask_data;
	float fn_tau, lambda_fn, lambda_fp, fp_tau, rel_pow;
};
__device__ __forceinline__ float hingeScale(const LossCfg& k, float lv)
{
	return k.rel_pow > 0.0f ? powf(fmaxf(lv, 0.0f), k.rel_pow) : 1.0f;
}
__device__ __forceinline__ float fpValue(const LossCfg& k, float I)
{
	return k.fp_mode == 1   ? fmaxf(logf(I / k.fp_tau), 0.0f)
		   : k.fp_mode == 2 ? fmaxf(I - k.fp_tau, 0.0f)
							: I;
}

__device__ __forceinline__ float fnValue(const LossCfg& k, float I, bool m)
{
	if (!m)
		return 0.0f;
	return k.fn_log ? fmaxf(logf(k.fn_tau / (I + 0.01f * k.fn_tau)), 0.0f)
					: fmaxf(k.fn_tau - I, 0.0f);
}

// The samples the data term covers.
__device__ __forceinline__ bool dataSample(const LossCfg& k, float c, float g)
{
	return !k.mask_data || (g != -1.0f && c != -1.0f);
}

constexpr int NACC = 6; // [sum err, count data, sum fn, count fn, sum fp, count fp]

__global__ void lossReduce(long long N, const float* __restrict__ c, const float* __restrict__ g,
	const float* __restrict__ I, LossCfg k, double* __restrict__ acc)
{
	float a[NACC] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < N;
		 i += (long long)gridDim.x * blockDim.x) {
		const float ci = c[i], gi = g[i], Ii = I[i];
		const float d = ci - gi;
		if (dataSample(k, ci, gi)) {
			a[0] += k.l1 ? fabsf(d) : d * d;
			a[1] += 1.0f;
		}
		const float v = fnValue(k, Ii, (gi != -1.0f) && (Ii > 0.0f));
		a[2] += v;
		a[3] += k.norm_all ? ((gi != -1.0f) ? 1.0f : 0.0f) : ((v > 0.0f) ? 1.0f : 0.0f);
		if (k.lambda_fp > 0.0f && gi == -1.0f && (Ii > 0.0f || k.norm_all)) {
			const float f = Ii > 0.0f ? fpValue(k, Ii) : 0.0f;
			a[4] += f;
			a[5] += (k.norm_all || k.fp_mode != 2 || f > 0.0f) ? 1.0f : 0.0f;
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

// out = [loss, data, fn, fp].
__global__ void lossEpilogue(
	long long N, const double* __restrict__ acc, LossCfg k, float* __restrict__ out)
{
	const float lv = (float)(acc[0] / (double)dataDenom(acc));
	const float hs = hingeScale(k, lv);
	const float scale = (k.rel ? k.lambda_fn * lv : k.lambda_fn) * hs;
	const float fn = scale * (float)acc[2] / ((float)acc[3] + 1e-8f);
	const float fp =
		k.lambda_fp > 0.0f ? hs * k.lambda_fp * (float)acc[4] / ((float)acc[5] + 1e-8f) : 0.0f;
	out[0] = lv + fn + fp;
	out[1] = lv;
	out[2] = fn;
	out[3] = fp;
}

__global__ void lossBackward(long long N, const float* __restrict__ c, const float* __restrict__ g,
	const float* __restrict__ I, const double* __restrict__ acc, LossCfg k,
	const float* __restrict__ gout, float* __restrict__ dc, float* __restrict__ dI)
{
	// Each output's gradient feeds its own terms; out[0] = data + fn + fp feeds all three.
	const float denom = dataDenom(acc);
	const float kv = (gout[0] + gout[1]) / denom;
	const float lv = (float)(acc[0] / (double)denom);
	const float hs = hingeScale(k, lv);
	const float scale = (k.rel ? k.lambda_fn * lv : k.lambda_fn) * hs;
	const float kf = -(gout[0] + gout[2]) * scale / ((float)acc[3] + 1e-8f);
	const float kp = (gout[0] + gout[3]) * hs * k.lambda_fp / ((float)acc[5] + 1e-8f);
	for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < N;
		 i += (long long)gridDim.x * blockDim.x) {
		const float ci = c[i], gi = g[i], Ii = I[i];
		const float d = ci - gi;
		dc[i] = dataSample(k, ci, gi) ? kv * (k.l1 ? (float)((d > 0.0f) - (d < 0.0f)) : 2.0f * d)
									  : 0.0f;
		const bool m = (gi != -1.0f) && (Ii > 0.0f);
		float gI = 0.0f;
		if (m) {
			if (k.fn_log) {
				const float dd = Ii + 0.01f * k.fn_tau; // d/dI log(fn_tau / dd) = -1 / dd
				if (logf(k.fn_tau / dd) >= 0.0f)
					gI = kf / dd;
			} else if (k.fn_tau - Ii >= 0.0f) {
				gI = kf;
			}
		}
		if (k.lambda_fp > 0.0f && gi == -1.0f && Ii > 0.0f) {
			if (k.fp_mode == 0)
				gI += kp;
			else if (k.fp_mode == 1) {
				if (logf(Ii / k.fp_tau) >= 0.0f)
					gI += kp / Ii;
			} else if (Ii - k.fp_tau >= 0.0f)
				gI += kp;
		}
		dI[i] = gI;
	}
}

void check(const torch::Tensor& t, const char* name)
{
	TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(), name,
		" must be a contiguous float32 CUDA tensor");
}

LossCfg makeCfg(int64_t l1, int64_t fn_log, int64_t rel, double fn_tau, double lambda_fn,
	double lambda_fp, double fp_tau, int64_t fp_mode, int64_t norm_all, double rel_pow,
	int64_t mask_data)
{
	return LossCfg{(int)l1, (int)fn_log, (int)rel, (int)fp_mode, (int)norm_all, (int)mask_data,
		(float)fn_tau, (float)lambda_fn, (float)lambda_fp, (float)fp_tau, (float)rel_pow};
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

// (out[4] = [loss, data, fn, fp], acc[6], the double accumulators the backward needs).
std::vector<torch::Tensor> volumeLossForward(torch::Tensor values, torch::Tensor gt,
	torch::Tensor influence, int64_t l1, int64_t fn_log, int64_t rel, double fn_tau,
	double lambda_fn, double lambda_fp, double fp_tau, int64_t fp_mode, int64_t norm_all,
	double rel_pow, int64_t mask_data)
{
	check(values, "values");
	check(gt, "gt");
	check(influence, "influence");
	TORCH_CHECK(values.numel() == gt.numel() && values.numel() == influence.numel(),
		"volume_loss: size mismatch");
	const long long N = values.numel();
	const LossCfg k = makeCfg(l1, fn_log, rel, fn_tau, lambda_fn, lambda_fp, fp_tau, fp_mode,
		norm_all, rel_pow, mask_data);
	auto acc = torch::zeros({NACC}, values.options().dtype(torch::kFloat64));
	auto out = torch::empty({4}, values.options());
	lossReduce<<<numBlocks(N, 8), 256>>>(N, values.data_ptr<float>(), gt.data_ptr<float>(),
		influence.data_ptr<float>(), k, acc.data_ptr<double>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	lossEpilogue<<<1, 1>>>(N, acc.data_ptr<double>(), k, out.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return {out, acc};
}

std::vector<torch::Tensor> volumeLossBackward(torch::Tensor values, torch::Tensor gt,
	torch::Tensor influence, torch::Tensor acc, int64_t l1, int64_t fn_log, int64_t rel,
	double fn_tau, double lambda_fn, double lambda_fp, double fp_tau, int64_t fp_mode,
	int64_t norm_all, double rel_pow, int64_t mask_data, torch::Tensor grad_out)
{
	check(values, "values");
	check(gt, "gt");
	check(influence, "influence");
	check(grad_out, "grad_out");
	const long long N = values.numel();
	const LossCfg k = makeCfg(l1, fn_log, rel, fn_tau, lambda_fn, lambda_fp, fp_tau, fp_mode,
		norm_all, rel_pow, mask_data);
	auto d_values = torch::empty_like(values), d_influence = torch::empty_like(influence);
	lossBackward<<<numBlocks(N, 16), 256>>>(N, values.data_ptr<float>(), gt.data_ptr<float>(),
		influence.data_ptr<float>(), acc.data_ptr<double>(), k, grad_out.data_ptr<float>(),
		d_values.data_ptr<float>(), d_influence.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return {d_values, d_influence};
}
