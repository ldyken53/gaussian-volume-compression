// Host entry points of gvraster._ops.
#pragma once
#include <torch/extension.h>
#include <vector>

void multiAdam(std::vector<torch::Tensor> params, std::vector<torch::Tensor> grads,
	std::vector<torch::Tensor> exp_avgs, std::vector<torch::Tensor> exp_avg_sqs,
	std::vector<double> lrs, int64_t step, double beta1, double beta2, double eps,
	std::vector<torch::Tensor> emas, double ema_w);
std::vector<torch::Tensor> volumeLossForward(torch::Tensor values, torch::Tensor gt,
	torch::Tensor influence, int64_t l1, int64_t fn_log, int64_t rel, double fn_tau,
	double lambda_fn, double lambda_fp, double fp_tau, int64_t fp_mode, int64_t norm_all,
	double rel_pow, int64_t mask_data);
std::vector<torch::Tensor> volumeLossBackward(torch::Tensor values, torch::Tensor gt,
	torch::Tensor influence, torch::Tensor acc, int64_t l1, int64_t fn_log, int64_t rel,
	double fn_tau, double lambda_fn, double lambda_fp, double fp_tau, int64_t fp_mode,
	int64_t norm_all, double rel_pow, int64_t mask_data, torch::Tensor grad_out);
