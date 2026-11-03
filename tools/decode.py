"""Decode a trained model over its whole source volume: the model's value at every node of the source
grid or mesh, written as a copy of the source with the data array replaced. NaN marks nodes where the
model is undefined.

    python tools/decode.py output/mito_1024 mito.vtu mito_decoded.vtu
    python tools/decode.py output/vertebra_1024 vertebra.vtk vertebra_decoded.vti --compare

Meshes are decoded at their nodes: training moves a mesh's cell data to its points, so the model
represents node values, and the output stores point data even when the source stored cell data.
--compare also prints the PSNR against the source over the defined nodes and the undefined share.
Surface models leave some nodes that lie exactly on the mesh boundary undefined (rbl 1024x: 0.8% of
boundary nodes, no interior ones): their reconstructed boundary sits within about 0.001 of the true one,
on either side.
"""

import os
import sys
import time
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veggies import api, volume  # noqa: E402


def node_chunks(ds, chunk):
    """(start, positions) over every node in the dataset's own order, about chunk nodes at a time."""
    if isinstance(ds, pv.ImageData):
        # One z slab at a time, x fastest: the grid's node order, without building every position.
        nx, ny, nz = ds.dimensions
        (ox, oy, oz), (sx, sy, sz) = ds.origin, ds.spacing
        X, Y = np.meshgrid(ox + sx * np.arange(nx), oy + sy * np.arange(ny), indexing="xy")
        xy = np.column_stack([X.ravel(), Y.ravel()])
        per = max(1, chunk // len(xy))
        for z0 in range(0, nz, per):
            zs = oz + sz * np.arange(z0, min(nz, z0 + per))
            yield z0 * len(xy), np.concatenate(
                [np.column_stack([xy, np.full(len(xy), z)]) for z in zs]
            )
    else:
        points = np.asarray(ds.points)
        for s in range(0, len(points), chunk):
            yield s, points[s : s + chunk]


def main(argv=None):
    p = ArgumentParser(
        description=__doc__.split("\n")[0], epilog=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    p.add_argument("model", help="trained model directory")
    p.add_argument("source", help="the volume the model was trained on")
    p.add_argument("out", help="decoded copy of the source (.vti / .vtu / .vtk)")
    p.add_argument("--array", default=None, help="the array the model was trained on (default: the first)")
    p.add_argument("--iteration", type=int, default=-1, help="checkpoint to load (default: the last)")
    p.add_argument("--chunk", type=int, default=4_000_000, help="nodes decoded per call")
    p.add_argument("--compare", action="store_true", help="report PSNR against the source")
    a = p.parse_args(argv)

    t0 = time.time()
    model = api.load_model(a.model, a.iteration)
    ds = volume.select_array(pv.read(a.source), a.array)
    name = ds.array_names[0]
    if name not in ds.point_data:
        ds = ds.cell_data_to_point_data()
    source = np.asarray(ds.point_data[name])
    if source.ndim > 1 and source.shape[1] > 1:
        raise SystemExit(f"{name!r} has {source.shape[1]} components; models fit scalar volumes")
    print(f"{ds.n_points} nodes, {name!r}; load {time.time() - t0:.0f}s", flush=True)

    t1 = time.time()
    decoded = np.empty(ds.n_points, dtype=np.float32)
    for start, positions in node_chunks(ds, a.chunk):
        decoded[start : start + len(positions)] = model.sample(positions).cpu().numpy()
    print(f"decoded in {time.time() - t1:.1f}s", flush=True)

    if a.compare:
        ref = source.reshape(-1).astype(np.float64)
        defined = ~np.isnan(decoded)
        mse = np.mean((decoded[defined] - ref[defined]) ** 2)
        span = ref.max() - ref.min()
        print(
            f"psnr over defined nodes {20 * np.log10(span) - 10 * np.log10(mse):.3f} dB; "
            f"undefined {np.count_nonzero(~defined)} of {len(ref)} nodes "
            f"({100 * np.mean(~defined):.4f}%)",
            flush=True,
        )

    ds.point_data[name] = decoded
    ds.save(a.out)
    print(f"wrote {a.out}; {time.time() - t0:.0f}s total", flush=True)


if __name__ == "__main__":
    main()
