#include "bvh.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("set_points", &setPoints);
	m.def("set_points_cached", &setPointsCached);
	m.def("leaf_order", &leafOrder);
	m.def("set_point_flags", &setPointFlags);
	m.def("sample_forward", &sampleForward);
	m.def("sample_backward", &sampleBackward);
	m.def("activate_forward", &activateForward);
	m.def("activate_backward", &activateBackward);
	m.def("margin_set_exterior", &marginSetExterior);
	m.def("margin_fp", &marginFP);
}
