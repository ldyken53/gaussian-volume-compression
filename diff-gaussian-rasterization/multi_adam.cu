// One-launch Adam over several parameter tensors with per-tensor learning rates.
// Ported unchanged from gaussian-volume-struct (d0b03fb, EMA fusion f3cc97d).
//
// torch.optim.Adam(fused=True) launches one multi_tensor_apply kernel per param group,
// one block per 65536 elements: a 44k-Gaussian model's groups get 1-3 blocks each on a
// 108-SM GPU, so the 5 launches are latency-bound (~50 us apiece for ~1 us of traffic).
// This covers every element of every tensor in a single grid-stride launch. Same update
// as torch's fused Adam (no weight decay, no amsgrad, no maximize):
//   m = b1 m + (1-b1) g;  v = b2 v + (1-b2) g^2;
//   p -= (lr / (1-b1^t)) * m / (sqrt(v) / sqrt(1-b2^t) + eps)
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <vector>

#define MAX_TENSORS 8

struct AdamArgs {
	float* p[MAX_TENSORS];
	const float* g[MAX_TENSORS];
	float* m[MAX_TENSORS];
	float* v[MAX_TENSORS];
	long long end[MAX_TENSORS];   // inclusive prefix sum of element counts
	float step_size[MAX_TENSORS]; // lr / (1 - b1^t)
	float lr[MAX_TENSORS];
	float* e[MAX_TENSORS];        // optional Polyak average, updated from the new p
	float eps[MAX_TENSORS];       // per-tensor Adam eps
	float gate[MAX_TENSORS];      // per-tensor SNR-gate exponent p (0 = plain Adam)
	int n;
};

template <bool EMA>
__global__ void multiAdamKernel(AdamArgs a, long long total, float b1, float b2,
                                float bc2_sqrt, float eps, float ema_w)
{
	for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
	     i += (long long)gridDim.x * blockDim.x) {
		int t = 0;
		while (i >= a.end[t]) t++;
		const long long j = i - (t ? a.end[t - 1] : 0);
		const float g = a.g[t][j];
		const float m = b1 * a.m[t][j] + (1.0f - b1) * g;
		const float v = b2 * a.v[t][j] + (1.0f - b2) * g * g;
		a.m[t][j] = m;
		a.v[t][j] = v;
		const float den = sqrtf(v) / bc2_sqrt + a.eps[t];
		float step = a.step_size[t] * m / den;
		// SNR gate: |m_hat| / (sqrt(v_hat) + eps) is ~1 for a gradient with a consistent
		// sign and ~sqrt((1-b1)/(1+b1)) for zero-mean noise, whatever its magnitude, so
		// raising it to p damps signal-free random walks without an absolute threshold.
		if (a.gate[t] > 0.0f) step *= powf(fminf(fabsf(a.step_size[t] * m / den) / a.lr[t], 1.0f), a.gate[t]);
		const float p = a.p[t][j] - step;
		a.p[t][j] = p;
		// torch.lerp_(ema, p, w) for w < 0.5: ema + w * (p - ema)
		if (EMA) { const float e = a.e[t][j]; a.e[t][j] = e + ema_w * (p - e); }
	}
}

static void multi_adam_impl(std::vector<torch::Tensor>& params, std::vector<torch::Tensor>& grads,
                 std::vector<torch::Tensor>& exp_avgs, std::vector<torch::Tensor>& exp_avg_sqs,
                 std::vector<double>& lrs, int64_t step, double beta1, double beta2, double eps,
                 std::vector<torch::Tensor>* emas, double ema_w,
                 const std::vector<double>* epss = nullptr, const std::vector<double>* gates = nullptr)
{
	const int n = params.size();
	TORCH_CHECK(n > 0 && n <= MAX_TENSORS, "multi_adam_: 1..8 tensors");
	TORCH_CHECK(grads.size() == n && exp_avgs.size() == n && exp_avg_sqs.size() == n &&
	            lrs.size() == n, "multi_adam_: list length mismatch");
	AdamArgs a;
	a.n = n;
	long long total = 0;
	const double bc1 = 1.0 - std::pow(beta1, (double)step);
	for (int t = 0; t < n; t++) {
		for (auto* x : {&params[t], &grads[t], &exp_avgs[t], &exp_avg_sqs[t]}) {
			TORCH_CHECK(x->is_cuda() && x->scalar_type() == torch::kFloat32 && x->is_contiguous(),
			            "multi_adam_: tensors must be contiguous float32 CUDA");
			TORCH_CHECK(x->numel() == params[t].numel(), "multi_adam_: size mismatch");
		}
		a.p[t] = params[t].data_ptr<float>();
		a.g[t] = grads[t].data_ptr<float>();
		a.m[t] = exp_avgs[t].data_ptr<float>();
		a.v[t] = exp_avg_sqs[t].data_ptr<float>();
		total += params[t].numel();
		a.end[t] = total;
		a.step_size[t] = (float)(lrs[t] / bc1);
		a.lr[t] = (float)lrs[t] > 0.0f ? (float)lrs[t] : 1.0f;
		a.eps[t] = (float)(epss ? (*epss)[t] : eps);
		a.gate[t] = (float)(gates ? (*gates)[t] : 0.0);
		a.e[t] = nullptr;
		if (emas) {
			auto& x = (*emas)[t];
			TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.is_contiguous() &&
			            x.numel() == params[t].numel(), "multi_adam_ema_: bad EMA buffer");
			a.e[t] = x.data_ptr<float>();
		}
	}
	if (total == 0) return;
	const float bc2_sqrt = (float)std::sqrt(1.0 - std::pow(beta2, (double)step));
	const int threads = 256;
	const long long blocks = std::min<long long>((total + threads - 1) / threads, 108 * 16);
	if (emas)
		multiAdamKernel<true><<<(int)blocks, threads>>>(a, total, (float)beta1, (float)beta2, bc2_sqrt, (float)eps, (float)ema_w);
	else
		multiAdamKernel<false><<<(int)blocks, threads>>>(a, total, (float)beta1, (float)beta2, bc2_sqrt, (float)eps, 0.0f);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void multi_adam_(std::vector<torch::Tensor> params, std::vector<torch::Tensor> grads,
                 std::vector<torch::Tensor> exp_avgs, std::vector<torch::Tensor> exp_avg_sqs,
                 std::vector<double> lrs, int64_t step, double beta1, double beta2, double eps)
{
	multi_adam_impl(params, grads, exp_avgs, exp_avg_sqs, lrs, step, beta1, beta2, eps, nullptr, 0.0);
}

// Same step, plus emas[t] = lerp(emas[t], new params[t], ema_w) in the same pass.
void multi_adam_ema_(std::vector<torch::Tensor> params, std::vector<torch::Tensor> grads,
                     std::vector<torch::Tensor> exp_avgs, std::vector<torch::Tensor> exp_avg_sqs,
                     std::vector<double> lrs, int64_t step, double beta1, double beta2, double eps,
                     std::vector<torch::Tensor> emas, double ema_w)
{
	TORCH_CHECK(emas.size() == params.size(), "multi_adam_ema_: list length mismatch");
	multi_adam_impl(params, grads, exp_avgs, exp_avg_sqs, lrs, step, beta1, beta2, eps, &emas, ema_w);
}

// General form: per-tensor eps and SNR-gate exponents; emas may be empty (no Polyak update).
void multi_adam_ex_(std::vector<torch::Tensor> params, std::vector<torch::Tensor> grads,
                    std::vector<torch::Tensor> exp_avgs, std::vector<torch::Tensor> exp_avg_sqs,
                    std::vector<double> lrs, int64_t step, double beta1, double beta2,
                    std::vector<double> epss, std::vector<double> gates,
                    std::vector<torch::Tensor> emas, double ema_w)
{
	TORCH_CHECK(epss.size() == params.size() && gates.size() == params.size(), "multi_adam_ex_: list length mismatch");
	TORCH_CHECK(emas.empty() || emas.size() == params.size(), "multi_adam_ex_: list length mismatch");
	multi_adam_impl(params, grads, exp_avgs, exp_avg_sqs, lrs, step, beta1, beta2, 0.0,
	                emas.empty() ? nullptr : &emas, ema_w, &epss, &gates);
}
