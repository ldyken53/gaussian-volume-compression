"""Evaluation points for a mesh too large to triangulate at once (eval.py --eval_set).

    python tools/precompute_tet.py mesh.vtu out/mesh_eval --n 1000000

Writes <out>_samples.npy (n, 3) and <out>_gt.npy (n,): uniform points in uniform tets, as
eval.py draws them, but from the tets of a random subset of the cells, which is statistically
the same and needs only that subset triangulated. Points and values are scaled the way train.py
scales the mesh. On meshes with non-tet cells the values are probed from the cells, as eval.py
takes them (--probe), rather than interpolated linearly within the split tets.
"""

import argparse

import numpy as np

from tets import (
    has_non_tets,
    points_in_tets,
    probe,
    read_mesh,
    scale_for_probe,
    triangulate,
)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("src", help="the mesh")
    p.add_argument("out", help="output prefix")
    p.add_argument("--array", help="the scalar array (default: the first)")
    p.add_argument("--n", type=int, default=1_000_000, help="points")
    p.add_argument(
        "--subset",
        type=int,
        default=4_000_000,
        help="cells to triangulate (0: all); must be >> n / 4 to keep the tets varied",
    )
    p.add_argument(
        "--probe",
        action=argparse.BooleanOptionalAction,
        help="probe the values from the cells (default: when the mesh has non-tet cells)",
    )
    a = p.parse_args()

    m, name, vals, pts = read_mesh(a.src, a.array)
    rng = np.random.default_rng(0)
    cells = None
    if a.subset and a.subset < m.n_cells:
        cells = rng.choice(m.n_cells, size=a.subset, replace=False)
    tets, _ = triangulate(m, cells)
    samples, gt = points_in_tets(
        tets[rng.integers(0, tets.shape[0], size=a.n)], pts, vals, rng
    )
    if a.probe is None:
        a.probe = has_non_tets(m)
    if a.probe:
        # The same points; only their values change.
        scale_for_probe(m, name, vals, pts)
        gt = probe(m, name, samples)
        # A point the probe misses lies on a seam of a non-conforming mesh, not outside it.
        found = gt != -1
        print(f"probed: dropping {int((~found).sum())} points not found", flush=True)
        samples, gt = samples[found], gt[found]
    np.save(a.out + "_samples.npy", samples.astype(np.float32))
    np.save(a.out + "_gt.npy", gt.astype(np.float32))
    print(
        f"wrote {a.out}_{{samples,gt}}.npy  n={a.n}  "
        f"gt[{gt.min():.5f},{gt.max():.5f}] mean {gt.mean():.5f}  "
        f"xyz bounds {np.round(samples.min(0), 4)}..{np.round(samples.max(0), 4)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
