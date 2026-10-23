#include "tile.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("sample_forward", &sampleForward);
	m.def("sample_backward", &sampleBackward);
}
