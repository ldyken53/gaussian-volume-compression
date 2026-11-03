"""Convert a raw volume (values only, no header) into a .vti grid that train.py reads.

    python tools/raw_to_grid.py volume.raw volume.vti --shape 512 512 512 --dtype uint8

The raw file holds X*Y*Z values with x varying fastest, then y, then z (the usual layout of raw
scientific volumes); give a byte order with the dtype if it is not the machine's (e.g. '>f4').

By default the grid is written already scaled the way train.py would scale it: values into
[0, 1] and the longest axis into [0, 1]. The transform is recorded in the grid's field data, so
train.py and eval.py skip their own scaling (as --is_scaled does), and models sample the grid in
its scaled units. --no_scale writes the raw values and node coordinates instead.
"""

import os
import sys
from argparse import ArgumentParser

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veggies.volume import SCALED_KEY  # noqa: E402


def main():
    p = ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("raw")
    p.add_argument("out", help="output .vti")
    p.add_argument("--shape", type=int, nargs=3, required=True, metavar=("X", "Y", "Z"))
    p.add_argument(
        "--dtype",
        default="float32",
        help="numpy dtype of the values, e.g. uint8, float32, >f4",
    )
    p.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("DX", "DY", "DZ"),
    )
    p.add_argument("--array", default="value", help="name of the scalar array")
    p.add_argument(
        "--no_scale",
        action="store_true",
        help="keep the raw values and coordinates (train.py then scales the grid at every load)",
    )
    a = p.parse_args()

    x, y, z = a.shape
    values = np.fromfile(a.raw, dtype=a.dtype, count=x * y * z)
    if values.size != x * y * z:
        raise SystemExit(f"{a.raw}: expected {x * y * z} values, found {values.size}")
    values = values.astype(np.float32)
    vmin, vmax = float(values.min()), float(values.max())
    spacing = np.array(a.spacing, dtype=np.float64)
    if not a.no_scale:
        if vmax == vmin:
            raise SystemExit(f"{a.raw}: every value is {vmin}; nothing to scale")
        # The same transform as veggies.volume.normalize_grid on the raw grid (origin 0).
        scale = float(((np.array(a.shape) - 1) * spacing).max())
        values -= np.float32(vmin)
        values /= np.float32(vmax - vmin)
        spacing = spacing / scale
    grid = pv.ImageData(dimensions=(x, y, z), spacing=tuple(spacing))
    grid.point_data[a.array] = values
    if not a.no_scale:
        grid.field_data[SCALED_KEY] = np.array([0.0, scale, vmin, vmax])
    grid.save(a.out)
    print(f"{a.out}: {x}x{y}x{z} grid, raw values in [{vmin}, {vmax}]")
    if not a.no_scale:
        print(
            f"  scaled: values (v - {vmin}) / {vmax - vmin:g}, coordinates / {scale:g} "
            f"(spacing {tuple(np.round(spacing, 8))}); train.py skips its own scaling"
        )


if __name__ == "__main__":
    main()
