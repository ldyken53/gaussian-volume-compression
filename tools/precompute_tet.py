"""Evaluation points for a mesh too large to triangulate at once (eval.py --eval_set).

    python tools/precompute_tet.py mesh.vtu out/mesh_eval --n 1000000

Writes <out>_samples.npy (n, 3) and <out>_gt.npy (n,): uniform points in uniform tets, as
eval.py draws them, but from the tets of a random subset of the cells, which is statistically
the same and needs only that subset triangulated. Points and values are scaled the way train.py
scales the mesh.
"""

import argparse

import numpy as np

from tets import points_in_tets, read_mesh, triangulate


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
    a = p.parse_args()

    m, _, vals, pts = read_mesh(a.src, a.array)
    rng = np.random.default_rng(0)
    cells = None
    if a.subset and a.subset < m.n_cells:
        cells = rng.choice(m.n_cells, size=a.subset, replace=False)
    tets, _ = triangulate(m, cells)
    samples, gt = points_in_tets(
        tets[rng.integers(0, tets.shape[0], size=a.n)], pts, vals, rng
    )
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
