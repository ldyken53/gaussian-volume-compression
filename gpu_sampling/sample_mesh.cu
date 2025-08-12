// sample_mesh_uniform.cpp

#include <iostream>
#include <numeric>
#include <cstring>

#include <viskores/cont/Initialize.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/DataSetBuilderExplicit.h>
#include <viskores/io/VTKDataSetReader.h>
#include <viskores/filter/resampling/Probe.h>
#include <viskores/rendering/Actor.h>
#include <viskores/rendering/CanvasRayTracer.h>
#include <viskores/rendering/MapperRayTracer.h>
#include <viskores/rendering/Scene.h>
#include <viskores/rendering/View3D.h>
#include <viskores/cont/Timer.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::array_t<double> sample_mesh(
    py::array_t<int64_t> dims_arr,
    py::array_t<float>   origin_arr,
    py::array_t<float>   spacing_arr,
    py::array_t<double>  val_arr,
    py::array_t<float>   samp_arr)
{
  // 0) Pick CUDA device for timing async kernels
  auto &tracker = viskores::cont::GetRuntimeDeviceTracker();
  tracker.ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  auto cudaTag = viskores::cont::DeviceAdapterTagCuda();
  viskores::cont::Timer timer(cudaTag);

  // 1) Initialization
  timer.Start();
  viskores::cont::Initialize();
  timer.Stop();
  std::cout << "Initialize: " 
            << timer.GetElapsedTime() << " s\n";

  // 2) Read and wrap input arrays
  timer.Reset();
  timer.Start();

  // dims
  auto dims_buf = dims_arr.request();
  auto dims_ptr = static_cast<int64_t*>(dims_buf.ptr);
  viskores::Id nx = dims_ptr[0],
                   ny = dims_ptr[1],
                   nz = dims_ptr[2];

  // origin
  auto orig_buf = origin_arr.request();
  auto o_ptr = static_cast<float*>(orig_buf.ptr);

  // spacing
  auto sp_buf = spacing_arr.request();
  auto s_ptr = static_cast<float*>(sp_buf.ptr);

  // values
  auto val_buf = val_arr.request();
  auto v_ptr   = static_cast<double*>(val_buf.ptr);
  std::size_t totalPts = static_cast<std::size_t>(nx) * ny * nz;
  std::vector<viskores::Float64> val_vec(v_ptr, v_ptr + totalPts);
  auto valHandle = viskores::cont::make_ArrayHandleMove(std::move(val_vec));

  // build uniform DataSet
  viskores::Id3 dims3{nx, ny, nz};
  viskores::Vec3f origin{ o_ptr[0], o_ptr[1], o_ptr[2] };
  viskores::Vec3f spacing{ s_ptr[0], s_ptr[1], s_ptr[2] };
  auto inData = viskores::cont::DataSetBuilderUniform::Create(
    dims3, origin, spacing, "coords"
  );
  inData.AddPointField("value", valHandle);

  timer.Stop();
  std::cout << "ReadDataSet: " 
            << timer.GetElapsedTime() << " s\n";

  // 3) Build explicit point‐vertex grid for sampling locations
  timer.Reset();
  timer.Start();

  auto samp_buf = samp_arr.request();
  std::size_t n_samples = samp_buf.shape[0];
  auto samp_ptr = static_cast<float*>(samp_buf.ptr);

  std::vector<viskores::Vec<float,3>> sample_coords(n_samples);
  std::memcpy(
    sample_coords.data(),
    samp_ptr,
    n_samples * 3 * sizeof(float)
  );

  std::vector<viskores::Id> sample_conn(n_samples);
  std::iota(sample_conn.begin(), sample_conn.end(), 0);

  auto explicitGrid = viskores::cont::DataSetBuilderExplicit::Create(
    sample_coords,
    viskores::CellShapeTagVertex{},
    static_cast<viskores::IdComponent>(1),
    sample_conn,
    "sample_coords"
  );

  timer.Stop();
  std::cout << "Build explicit grid: " 
            << timer.GetElapsedTime() << " s\n";

  // 4) Probe filter setup
  timer.Reset();
  timer.Start();

  viskores::filter::resampling::Probe probe;
  probe.SetGeometry(explicitGrid);
  probe.SetInvalidValue(-1.0);

  timer.Stop();
  std::cout << "Probe setup: " 
            << timer.GetElapsedTime() << " s\n";

  // 5) Probe execution
  timer.Reset();
  timer.Start();

  viskores::cont::DataSet sampled = probe.Execute(inData);

  timer.Stop();
  std::cout << "Probe execute: " 
            << timer.GetElapsedTime() << " s\n";

  // 6) Retrieve and return result
  timer.Reset();
  timer.Start();

  const auto array = sampled.GetPointField("value").GetData();
  auto concrete = array.AsArrayHandle<viskores::cont::ArrayHandle<viskores::Float64>>();
  concrete.SyncControlArray();
  auto readPortal = concrete.ReadPortal();

  std::size_t n = readPortal.GetNumberOfValues();
  py::array_t<double> result(n);
  auto out_ptr = result.mutable_data();
  for (std::size_t i = 0; i < n; ++i)
  {
    out_ptr[i] = readPortal.Get(i);
  }

  timer.Stop();
  std::cout << "Data retrieval: " 
            << timer.GetElapsedTime() << " s\n";

  return result;
}