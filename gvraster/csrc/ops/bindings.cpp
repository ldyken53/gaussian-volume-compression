#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("multi_adam_", &multiAdam);
	m.def("volume_loss_forward", &volumeLossForward);
	m.def("volume_loss_backward", &volumeLossBackward);
}
