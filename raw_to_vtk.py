#!/usr/bin/env python3
"""
Convert a 2048×2048×1920 uint8 raw volume (NRRD-style) into a
float32, normalized [0,1] voxel dataset on the unit cube, and save
as a structured VTK file using PyVista (ImageData).
"""

import numpy as np
import pyvista as pv
import os


def read_raw_volume(filename, shape, dtype=np.uint8, order="C"):
    """
    Reads a raw binary file into a NumPy array.
    """
    count = np.prod(shape)
    data = np.fromfile(filename, dtype=dtype, count=count)
    if data.size != count:
        raise IOError(f"Expected {count} elements, got {data.size}")
    return data.reshape(shape, order=order)


def normalize_volume(vol):
    """
    Linearly scales vol so its values lie in [0,1].
    """
    vmin, vmax = vol.min(), vol.max()
    if vmax == vmin:
        return np.zeros_like(vol, dtype=np.float32)
    return (vol - vmin) / (vmax - vmin)


def build_image_data(volume):
    """
    Builds a PyVista ImageData whose points span [0,1]^3 and whose
    point-data is 'volume'.
    """
    nx, ny, nz = volume.shape
    m = max(nx, ny, nz)
    spacing = (1.0 / (m), 1.0 / (m), 1.0 / (m))
    origin = (0.0, 0.0, 0.0)

    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=spacing, origin=origin)
    print(grid.bounds)
    grid.point_data["value"] = volume.ravel(order="C")
    return grid


def main():
    # --- Update these to match your header ---
    raw_file = "/lus/eagle/projects/dist_relational_alg/ldyken53/miranda_1024x1024x1024_float32.raw"
    vtk_file = "miran.vtk"
    shape = (1024, 1024, 1024)  # as per sizes: X, Y, Z
    dtype = np.float32  # input type is uint8
    order = "C"  # use 'F' if needed for Fortran-order

    if not os.path.isfile(raw_file):
        raise FileNotFoundError(f"Could not locate raw file: {raw_file}")

    print("1) Reading raw uint8 volume…")
    vol_u8 = read_raw_volume(raw_file, shape, dtype=dtype, order=order)

    print("2) Converting to float32…")
    vol_f32 = vol_u8.astype(np.float32)

    print("3) Normalizing to [0,1]…")
    vol_norm = normalize_volume(vol_f32)

    print("4) Building ImageData in [0,1]^3…")
    grid = build_image_data(vol_norm)

    print(f"5) Saving to VTK: {vtk_file} …")
    grid.save(vtk_file)

    print("Done!")
    mesh = pv.read(vtk_file)
    print(mesh.bounds)


if __name__ == "__main__":
    main()
