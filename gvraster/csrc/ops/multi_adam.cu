// One-launch Adam over several parameter tensors with per-tensor learning rates, optionally
// updating each tensor's parameter average in the same pass. It is torch's fused Adam without
// weight decay, amsgrad or maximize:
//   m = b1 m + (1-b1) g;  v = b2 v + (1-b2) g^2;
//   p -= (lr / (1-b1^t)) * m / (sqrt(v) / sqrt(1-b2^t) + eps)
#include "ops.h"
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#define MAX_TENSORS 8

struct AdamArgs {
	float* p[MAX_TENSORS];
	const float* g[MAX_TENSORS];
	float* m[MAX_TENSORS];
	float* v[MAX_TENSORS];
	float* e[MAX_TENSORS];        // the parameter averages, updated from the new p
	long long end[MAX_TENSORS];   // inclusive prefix sum of the element counts
	float step_size[MAX_TENSORS]; // lr / (1 - b1^t)
};

template <bool EMA>
__global__ void multiAdamKernel(
	AdamArgs a, long long total, float b1, float b2, float bc2_sqrt, float eps, float ema_w)
{
	for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
		 i += (long long)gridDim.x * blockDim.x) {
		int t = 0;
		while (i >= a.end[t])
			t++;
		const long long j = i - (t ? a.end[t - 1] : 0);
		const float g = a.g[t][j];
		const float m = b1 * a.m[t][j] + (1.0f - b1) * g;
		const float v = b2 * a.v[t][j] + (1.0f - b2) * g * g;
		a.m[t][j] = m;
		a.v[t][j] = v;
		const float p = a.p[t][j] - a.step_size[t] * m / (sqrtf(v) / bc2_sqrt + eps);
		a.p[t][j] = p;
		// torch.lerp_(ema, p, w) for w < 0.5.
		if (EMA) {
			const float e = a.e[t][j];
			a.e[t][j] = e + ema_w * (p - e);
		}
	}
}

// emas is empty, or one average per parameter, updated to lerp(average, new parameter, ema_w).
void multiAdam(std::vector<torch::Tensor> params, std::vector<torch::Tensor> grads,
	std::vector<torch::Tensor> exp_avgs, std::vector<torch::Tensor> exp_avg_sqs,
	std::vector<double> lrs, int64_t step, double beta1, double beta2, double eps,
	std::vector<torch::Tensor> emas, double ema_w)
{
	const int n = params.size();
	const bool ema = !emas.empty();
	TORCH_CHECK(n > 0 && n <= MAX_TENSORS, "multi_adam_: 1..8 tensors");
	TORCH_CHECK(grads.size() == n && exp_avgs.size() == n && exp_avg_sqs.size() == n &&
					lrs.size() == n && (!ema || emas.size() == n),
		"multi_adam_: list length mismatch");
	AdamArgs a;
	long long total = 0;
	const double bc1 = 1.0 - std::pow(beta1, (double)step);
	for (int t = 0; t < n; t++) {
		std::vector<torch::Tensor*> ts = {&params[t], &grads[t], &exp_avgs[t], &exp_avg_sqs[t]};
		if (ema)
			ts.push_back(&emas[t]);
		for (auto* x : ts)
			TORCH_CHECK(x->is_cuda() && x->scalar_type() == torch::kFloat32 && x->is_contiguous() &&
							x->numel() == params[t].numel(),
				"multi_adam_: tensors must be contiguous float32 CUDA of matching size");
		a.p[t] = params[t].data_ptr<float>();
		a.g[t] = grads[t].data_ptr<float>();
		a.m[t] = exp_avgs[t].data_ptr<float>();
		a.v[t] = exp_avg_sqs[t].data_ptr<float>();
		a.e[t] = ema ? emas[t].data_ptr<float>() : nullptr;
		total += params[t].numel();
		a.end[t] = total;
		a.step_size[t] = (float)(lrs[t] / bc1);
	}
	if (total == 0)
		return;
	const float bc2_sqrt = (float)std::sqrt(1.0 - std::pow(beta2, (double)step));
	const int threads = 256;
	const long long blocks = std::min<long long>((total + threads - 1) / threads, 108 * 16);
	if (ema)
		multiAdamKernel<true><<<(int)blocks, threads>>>(
			a, total, (float)beta1, (float)beta2, bc2_sqrt, (float)eps, (float)ema_w);
	else
		multiAdamKernel<false><<<(int)blocks, threads>>>(
			a, total, (float)beta1, (float)beta2, bc2_sqrt, (float)eps, 0.0f);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
}
