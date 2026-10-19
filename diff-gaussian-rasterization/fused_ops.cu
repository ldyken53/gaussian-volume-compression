// Fused per-Gaussian activations and the volume loss, forward and backward. Matches the
// torch expressions they replace, including torch's gradient conventions at the edges
// (clamp_min passes the gradient at equality; abs' is 0 at 0; normalize's eps floor blocks it).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

namespace
{

constexpr float CAP_EPS = 1e-8f;   // _apply_cap: r = ||s|| + 1e-8
constexpr float NORM_EPS = 1e-12f; // F.normalize default eps

__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// scale = s * M tanh(r/M) / r,  s = exp(raw),  r = ||s|| + eps
__global__ void actForward(int N, const float* __restrict__ rs, const float* __restrict__ rq,
	const float* __restrict__ rv, const float* __restrict__ rw, float M, float* __restrict__ os,
	float* __restrict__ oq, float* __restrict__ ov, float* __restrict__ ow)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N)
		return;
	const float s0 = expf(rs[3 * i]), s1 = expf(rs[3 * i + 1]), s2 = expf(rs[3 * i + 2]);
	const float r = sqrtf(s0 * s0 + s1 * s1 + s2 * s2) + CAP_EPS;
	const float f = M * tanhf(r / M) / r;
	os[3 * i] = s0 * f;
	os[3 * i + 1] = s1 * f;
	os[3 * i + 2] = s2 * f;

	const float q0 = rq[4 * i], q1 = rq[4 * i + 1], q2 = rq[4 * i + 2], q3 = rq[4 * i + 3];
	const float inv = 1.0f / fmaxf(sqrtf(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3), NORM_EPS);
	oq[4 * i] = q0 * inv;
	oq[4 * i + 1] = q1 * inv;
	oq[4 * i + 2] = q2 * inv;
	oq[4 * i + 3] = q3 * inv;

	ov[i] = sigmoidf(rv[i]);
	ow[i] = sigmoidf(rw[i]);
}

__global__ void actBackward(int N, const float* __restrict__ rs, const float* __restrict__ rq,
	const float* __restrict__ ov, const float* __restrict__ ow, float M,
	const float* __restrict__ gs, const float* __restrict__ gq, const float* __restrict__ gv,
	const float* __restrict__ gw, float* __restrict__ ds, float* __restrict__ dq,
	float* __restrict__ dv, float* __restrict__ dw)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N)
		return;
	// Soft cap. f(r) = M tanh(r/M)/r, f'(r) = ((1 - t^2) r - M t) / r^2,
	// d scale_j/d s_k = delta_jk f + s_j f'(r) s_k / ||s||, then * s_k for the exp.
	const float s0 = expf(rs[3 * i]), s1 = expf(rs[3 * i + 1]), s2 = expf(rs[3 * i + 2]);
	const float nrm = sqrtf(s0 * s0 + s1 * s1 + s2 * s2);
	const float r = nrm + CAP_EPS;
	const float t = tanhf(r / M);
	const float f = M * t / r;
	const float fp = ((1.0f - t * t) * r - M * t) / (r * r);
	const float g0 = gs[3 * i], g1 = gs[3 * i + 1], g2 = gs[3 * i + 2];
	const float c = (nrm > 0.0f) ? (g0 * s0 + g1 * s1 + g2 * s2) * fp / nrm : 0.0f;
	ds[3 * i] = (g0 * f + c * s0) * s0;
	ds[3 * i + 1] = (g1 * f + c * s1) * s1;
	ds[3 * i + 2] = (g2 * f + c * s2) * s2;

	// Normalize: y = x / max(||x||, eps). Above the floor dx = (g - y (y.g)) / ||x||;
	// at the floor the denominator is constant, so dx = g / eps.
	const float x0 = rq[4 * i], x1 = rq[4 * i + 1], x2 = rq[4 * i + 2], x3 = rq[4 * i + 3];
	const float n = sqrtf(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3);
	const float h0 = gq[4 * i], h1 = gq[4 * i + 1], h2 = gq[4 * i + 2], h3 = gq[4 * i + 3];
	if (n >= NORM_EPS) {
		const float inv = 1.0f / n;
		const float y0 = x0 * inv, y1 = x1 * inv, y2 = x2 * inv, y3 = x3 * inv;
		const float yg = y0 * h0 + y1 * h1 + y2 * h2 + y3 * h3;
		dq[4 * i] = (h0 - y0 * yg) * inv;
		dq[4 * i + 1] = (h1 - y1 * yg) * inv;
		dq[4 * i + 2] = (h2 - y2 * yg) * inv;
		dq[4 * i + 3] = (h3 - y3 * yg) * inv;
	} else {
		const float inv = 1.0f / NORM_EPS;
		dq[4 * i] = h0 * inv;
		dq[4 * i + 1] = h1 * inv;
		dq[4 * i + 2] = h2 * inv;
		dq[4 * i + 3] = h3 * inv;
	}

	const float v = ov[i], w = ow[i];
	dv[i] = gv[i] * v * (1.0f - v);
	dw[i] = gw[i] * w * (1.0f - w);
}

// Loss terms:
//   data  (c-g)^2 or |c-g| (l1), mean over cells with g != -1 and c != -1
//   FN    on (g != -1) & (w > 0): hinge clamp(tau - w, 0) or log barrier
//         clamp(log(tau / (w + 0.01 tau)), 0); scaled by fn_reg, or fn_reg * l_v (rel, detached)
//   FP    (fp_reg > 0) on (g == -1) & (w > 0): w (lin), clamp(log(w / fp_t), 0) (log) or
//         clamp(w - fp_t, 0) (hinge, mean over its violators)
// FN and FP are means over their violators, or over all interior / exterior cells (norm_all).
// rel_pow > 0 scales both by l_v^rel_pow (detached).
struct LossCfg {
	int l1, fn_log, rel, fp_kind, norm_all;
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

constexpr int NACC = 6; // [sum err, count data, sum fn, count fn>0, sum fp, count fp>0]

__global__ void lossReduce(long long N, const float* __restrict__ c, const float* __restrict__ g,
	const float* __restrict__ w, LossCfg k, double* __restrict__ acc)
{
	float a[NACC] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < N;
		 i += (long long)gridDim.x * blockDim.x) {
		const float ci = c[i], gi = g[i], wi = w[i];
		const float d = ci - gi;
		const bool md = gi != -1.0f && ci != -1.0f;
		if (md) {
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
		const bool md = gi != -1.0f && ci != -1.0f;
		// abs'(0) = 0, as torch.
		dc[i] = md ? kv * (k.l1 ? (float)((d > 0.0f) - (d < 0.0f)) : 2.0f * d) : 0.0f;
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
	double fp_t, int64_t fp_kind, int64_t norm_all, double rel_pow)
{
	return LossCfg{(int)l1, (int)fn_log, (int)rel, (int)fp_kind, (int)norm_all, (float)tau,
		(float)fn_reg, (float)fp_reg, (float)fp_t, (float)rel_pow};
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

std::vector<torch::Tensor> fused_act_forward(
	torch::Tensor rs, torch::Tensor rq, torch::Tensor rv, torch::Tensor rw, double max_scale)
{
	check(rs, "scaling");
	check(rq, "rotation");
	check(rv, "values");
	check(rw, "weight");
	const int N = rs.size(0);
	auto os = torch::empty_like(rs), oq = torch::empty_like(rq), ov = torch::empty_like(rv),
		 ow = torch::empty_like(rw);
	if (N > 0)
		actForward<<<(N + 255) / 256, 256>>>(N, rs.data_ptr<float>(), rq.data_ptr<float>(),
			rv.data_ptr<float>(), rw.data_ptr<float>(), (float)max_scale, os.data_ptr<float>(),
			oq.data_ptr<float>(), ov.data_ptr<float>(), ow.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return {os, oq, ov, ow};
}

std::vector<torch::Tensor> fused_act_backward(torch::Tensor rs, torch::Tensor rq, torch::Tensor ov,
	torch::Tensor ow, double max_scale, torch::Tensor gs, torch::Tensor gq, torch::Tensor gv,
	torch::Tensor gw)
{
	for (auto* t : {&rs, &rq, &ov, &ow, &gs, &gq, &gv, &gw})
		check(*t, "fused_act_backward input");
	const int N = rs.size(0);
	auto ds = torch::empty_like(rs), dq = torch::empty_like(rq), dv = torch::empty_like(ov),
		 dw = torch::empty_like(ow);
	if (N > 0)
		actBackward<<<(N + 255) / 256, 256>>>(N, rs.data_ptr<float>(), rq.data_ptr<float>(),
			ov.data_ptr<float>(), ow.data_ptr<float>(), (float)max_scale, gs.data_ptr<float>(),
			gq.data_ptr<float>(), gv.data_ptr<float>(), gw.data_ptr<float>(), ds.data_ptr<float>(),
			dq.data_ptr<float>(), dv.data_ptr<float>(), dw.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return {ds, dq, dv, dw};
}

// Returns (out[4] = [loss, l_v, fn, fp], acc[6] doubles kept for the backward).
std::vector<torch::Tensor> fused_loss_forward(torch::Tensor c, torch::Tensor g, torch::Tensor w,
	int64_t l1, int64_t fn_log, int64_t rel, double tau, double fn_reg, double fp_reg, double fp_t,
	int64_t fp_kind, int64_t norm_all, double rel_pow)
{
	check(c, "cells");
	check(g, "gt");
	check(w, "weights");
	TORCH_CHECK(c.numel() == g.numel() && c.numel() == w.numel(), "fused_loss: size mismatch");
	const long long N = c.numel();
	const LossCfg k =
		makeCfg(l1, fn_log, rel, tau, fn_reg, fp_reg, fp_t, fp_kind, norm_all, rel_pow);
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
	torch::Tensor gout)
{
	check(c, "cells");
	check(g, "gt");
	check(w, "weights");
	check(gout, "grad");
	const long long N = c.numel();
	const LossCfg k =
		makeCfg(l1, fn_log, rel, tau, fn_reg, fp_reg, fp_t, fp_kind, norm_all, rel_pow);
	auto dc = torch::empty_like(c), dw = torch::empty_like(w);
	lossBackward<<<numBlocks(N, 16), 256>>>(N, c.data_ptr<float>(), g.data_ptr<float>(),
		w.data_ptr<float>(), acc.data_ptr<double>(), k, gout.data_ptr<float>(),
		dc.data_ptr<float>(), dw.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return {dc, dw};
}
