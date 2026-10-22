#include "ops.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("multi_adam_", &multi_adam_);
	m.def("fused_loss_forward", &fused_loss_forward);
	m.def("fused_loss_backward", &fused_loss_backward);
}
