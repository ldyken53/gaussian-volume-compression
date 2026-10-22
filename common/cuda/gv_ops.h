// Host entry points of the kernels both extensions compile from common/cuda (multi_adam.cuh,
// volume_loss.cuh); each extension's bindings include this.
#pragma once
#include <torch/extension.h>
#include <vector>

void multi_adam_(std::vector<torch::Tensor> params, std::vector<torch::Tensor> grads,
	std::vector<torch::Tensor> exp_avgs, std::vector<torch::Tensor> exp_avg_sqs,
	std::vector<double> lrs, int64_t step, double beta1, double beta2, double eps,
	std::vector<torch::Tensor> emas, double ema_w);
std::vector<torch::Tensor> fused_loss_forward(torch::Tensor c, torch::Tensor g, torch::Tensor w,
	int64_t l1, int64_t fn_log, int64_t rel, double tau, double fn_reg, double fp_reg, double fp_t,
	int64_t fp_kind, int64_t norm_all, double rel_pow, int64_t mask_data);
std::vector<torch::Tensor> fused_loss_backward(torch::Tensor c, torch::Tensor g, torch::Tensor w,
	torch::Tensor acc, int64_t l1, int64_t fn_log, int64_t rel, double tau, double fn_reg,
	double fp_reg, double fp_t, int64_t fp_kind, int64_t norm_all, double rel_pow,
	int64_t mask_data, torch::Tensor gout);
