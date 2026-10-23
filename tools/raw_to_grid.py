"""Convert a raw volume (values only, no header) into a .vti grid that train.py reads.

    python tools/raw_to_grid.py volume.raw volume.vti --shape 512 512 512 --dtype uint8

The raw file holds X*Y*Z values with x varying fastest, then y, then z (the usual layout of raw
scientific volumes). train.py scales the grid into the unit cube and the values into [0, 1].
"""

from argparse import ArgumentParser

import numpy as np
import pyvista as pv


def main():
    p = ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("raw")
    p.add_argument("out", help="output .vti")
    p.add_argument("--shape", type=int, nargs=3, required=True, metavar=("X", "Y", "Z"))
    p.add_argument(
        "--dtype",
        default="float32",
        help="numpy dtype of the values, e.g. uint8, float32",
    )
    p.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("DX", "DY", "DZ"),
    )
    p.add_argument("--array", default="value", help="name of the scalar array")
    a = p.parse_args()

    x, y, z = a.shape
    values = np.fromfile(a.raw, dtype=a.dtype, count=x * y * z)
    if values.size != x * y * z:
        raise SystemExit(f"{a.raw}: expected {x * y * z} values, found {values.size}")
    grid = pv.ImageData(dimensions=(x, y, z), spacing=a.spacing)
    grid.point_data[a.array] = values.astype(np.float32)
    grid.save(a.out)
    print(f"{a.out}: {x}x{y}x{z} grid, values in [{values.min()}, {values.max()}]")


if __name__ == "__main__":
    main()
