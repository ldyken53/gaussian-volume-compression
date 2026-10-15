/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include <vector>
void multi_adam_(std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
                 std::vector<torch::Tensor>, std::vector<double>, int64_t, double, double, double);
std::vector<torch::Tensor> fused_act_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
std::vector<torch::Tensor> fused_act_backward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double,
                                              torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> fused_loss_forward(torch::Tensor, torch::Tensor, torch::Tensor, double);
std::vector<torch::Tensor> fused_loss_backward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, torch::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("compute_relocation", &ComputeRelocationCUDA);
  m.def("multi_adam_", &multi_adam_);
  m.def("fused_act_forward", &fused_act_forward);
  m.def("fused_act_backward", &fused_act_backward);
  m.def("fused_loss_forward", &fused_loss_forward);
  m.def("fused_loss_backward", &fused_loss_backward);
}