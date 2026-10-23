// The raw parameters' activations, forward and backward. They match the torch expressions they
// replace, including torch's gradient conventions at the edges (normalize's eps floor blocks the
// gradient).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include "bvh.h"

namespace
{

void check(const torch::Tensor& t, const char* name)
{
	TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(), name,
		" must be a contiguous float32 CUDA tensor");
}

constexpr float CAP_EPS = 1e-8f;   // r = ||s|| + 1e-8 in the soft cap
constexpr float NORM_EPS = 1e-12f; // F.normalize's default eps

__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// scale = s M tanh(r / M) / r with s = exp(raw), r = ||s|| + eps
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
	// Soft cap: f(r) = M tanh(r/M) / r, f'(r) = ((1 - t^2) r - M t) / r^2 and
	// d scale_j / d s_k = delta_jk f + s_j f'(r) s_k / ||s||, then * s_k for the exp.
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

	// Normalize: y = x / max(||x||, eps). Above the floor dx = (g - y (y.g)) / ||x||; at the
	// floor the denominator is constant, so dx = g / eps.
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

} // namespace

std::vector<torch::Tensor> activateForward(torch::Tensor raw_scales, torch::Tensor raw_rotations,
	torch::Tensor raw_values, torch::Tensor raw_weights, double max_scale)
{
	check(raw_scales, "raw_scales");
	check(raw_rotations, "raw_rotations");
	check(raw_values, "raw_values");
	check(raw_weights, "raw_weights");
	const int N = raw_scales.size(0);
	auto scales = torch::empty_like(raw_scales), rotations = torch::empty_like(raw_rotations),
		 values = torch::empty_like(raw_values), weights = torch::empty_like(raw_weights);
	if (N > 0)
		actForward<<<(N + 255) / 256, 256>>>(N, raw_scales.data_ptr<float>(),
			raw_rotations.data_ptr<float>(), raw_values.data_ptr<float>(),
			raw_weights.data_ptr<float>(), (float)max_scale, scales.data_ptr<float>(),
			rotations.data_ptr<float>(), values.data_ptr<float>(), weights.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return {scales, rotations, values, weights};
}

std::vector<torch::Tensor> activateBackward(torch::Tensor raw_scales, torch::Tensor raw_rotations,
	torch::Tensor values, torch::Tensor weights, double max_scale, torch::Tensor dL_dscales,
	torch::Tensor dL_drotations, torch::Tensor dL_dvalues, torch::Tensor dL_dweights)
{
	for (auto* t : {&raw_scales, &raw_rotations, &values, &weights, &dL_dscales, &dL_drotations,
			 &dL_dvalues, &dL_dweights})
		check(*t, "activate_backward input");
	const int N = raw_scales.size(0);
	auto ds = torch::empty_like(raw_scales), dq = torch::empty_like(raw_rotations),
		 dv = torch::empty_like(values), dw = torch::empty_like(weights);
	if (N > 0)
		actBackward<<<(N + 255) / 256, 256>>>(N, raw_scales.data_ptr<float>(),
			raw_rotations.data_ptr<float>(), values.data_ptr<float>(), weights.data_ptr<float>(),
			(float)max_scale, dL_dscales.data_ptr<float>(), dL_drotations.data_ptr<float>(),
			dL_dvalues.data_ptr<float>(), dL_dweights.data_ptr<float>(), ds.data_ptr<float>(),
			dq.data_ptr<float>(), dv.data_ptr<float>(), dw.data_ptr<float>());
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	return {ds, dq, dv, dw};
}
