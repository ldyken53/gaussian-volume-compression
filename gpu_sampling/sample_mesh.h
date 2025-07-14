#include <iostream>
#include <viskores/cont/Initialize.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::array_t<double> sample_mesh(
    py::array_t<float> pts_arr,
    py::array_t<int64_t> conn_arr,
    py::array_t<double> val_arr,
    py::array_t<float> samp_arr
);