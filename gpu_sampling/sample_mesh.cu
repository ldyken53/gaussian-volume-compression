#include <viskores/cont/Initialize.h>
#include <viskores/cont/DataSetBuilderExplicit.h>
#include <viskores/io/VTKDataSetReader.h>
#include <viskores/filter/resampling/Probe.h>
#include <viskores/filter/mesh_info/MeshQuality.h>
#include <viskores/rendering/Actor.h>
#include <viskores/rendering/CanvasRayTracer.h>
#include <viskores/rendering/MapperRayTracer.h>
#include <viskores/rendering/Scene.h>
#include <viskores/rendering/View3D.h>
#include <viskores/cont/Timer.h> // for timing :contentReference[oaicite:1]{index=1}
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::array_t<double> sample_mesh(
  py::array_t<float> pts_arr,
  py::array_t<int64_t> conn_arr,
  py::array_t<double> val_arr,
  py::array_t<float> samp_arr
)
{
  // Pick the CUDA device for timing async kernels
  auto &tracker = viskores::cont::GetRuntimeDeviceTracker();
  tracker.ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  auto cudaTag = viskores::cont::DeviceAdapterTagCuda();
  viskores::cont::Timer timer(cudaTag); // GPU timer :contentReference[oaicite:2]{index=2}

  // 1) Initialization
  timer.Start();                      // begin timing :contentReference[oaicite:3]{index=3}
  viskores::cont::Initialize();
  timer.Stop();                       // end timing :contentReference[oaicite:4]{index=4}
  std::cout << "Initialize: " 
            << timer.GetElapsedTime() << " s\n"; // elapsed :contentReference[oaicite:5]{index=5}

  // 2) Read the VTK dataset
  timer.Reset();                      // clear previous time :contentReference[oaicite:6]{index=6}
  timer.Start();
  auto pts_buf   = pts_arr.request();   // get ptr, shape, strides
  auto conn_buf  = conn_arr.request();
  auto val_buf = val_arr.request();

  size_t n_pts = pts_buf.shape[0];
  auto pts_ptr = static_cast<float*>(pts_buf.ptr);
  std::vector<viskores::Vec<float,3>> coords;
  coords.reserve(n_pts);
  for (size_t i = 0; i < n_pts; ++i) {
    coords.emplace_back(
      pts_ptr[3*i + 0],
      pts_ptr[3*i + 1],
      pts_ptr[3*i + 2]
    );
  }
  auto conn_ptr = static_cast<int64_t*>(conn_buf.ptr);
  std::vector<viskores::Id> conn_vec(conn_ptr, conn_ptr + conn_buf.shape[0]);
  auto val_ptr = static_cast<double*>(val_buf.ptr);
  std::vector<viskores::Float64> val_vec(val_ptr, val_ptr + val_buf.shape[0]);
  viskores::cont::ArrayHandle<viskores::Float64> valHandle = viskores::cont::make_ArrayHandleMove(std::move(val_vec));

  auto inData = viskores::cont::DataSetBuilderExplicit::Create(
    coords, viskores::CellShapeTagTetra{}, static_cast<viskores::IdComponent>(4), conn_vec, "coords");
  
  inData.AddPointField(
    "value",
    valHandle
  );
  timer.Stop();
  std::cout << "ReadDataSet: " 
            << timer.GetElapsedTime() << " s\n";

  // 3) Build an explicit point‐vertex grid
  timer.Reset();
  timer.Start();
  auto samp_buf = samp_arr.request();
  size_t n_samples = samp_buf.shape[0];
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
    viskores::CellShapeTagVertex{},                // each cell is a single vertex
    static_cast<viskores::IdComponent>(1),          // 1 point per cell
    sample_conn,                                    // connectivity [0,1,2,…]
    "sample_coords"                                 // name for the coordinate field
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

  // 5) Probe filter execution
  timer.Reset();
  timer.Start();
  viskores::cont::DataSet sampled = probe.Execute(inData);
  timer.Stop();
  std::cout << "Probe execute: " 
            << timer.GetElapsedTime() << " s\n";

  // 6) Retrieve and synchronize field data
  timer.Reset();
  timer.Start();
  const auto array = sampled.GetPointField("value").GetData();
  auto concrete = array.AsArrayHandle<viskores::cont::ArrayHandle<viskores::Float64>>();
  concrete.SyncControlArray();       // pull data back to host
  auto readPortal = concrete.ReadPortal();
  std::size_t n = readPortal.GetNumberOfValues();
  py::array_t<double> result(n);
  auto buf = result.mutable_data();
  for (std::size_t i = 0; i < n; ++i)
  {
    buf[i] = readPortal.Get(i);
  }
  timer.Stop();
  std::cout << "Data retrieval: " 
            << timer.GetElapsedTime() << " s\n";

  return result;
}