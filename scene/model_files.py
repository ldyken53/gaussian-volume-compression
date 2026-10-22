"""Files kept beside each model's ply. Rendering a model needs max_scale, the size cap
get_scaling applies, and render_rule, the (occ_t, floor) it renders with: a point is
empty where aw <= occ_t, else S / max(aw, floor). train_seconds.txt is its training
time."""

import os

import numpy as np


def size_cap(max_scale, k, mins, maxes, n):
    """The size cap max(max_scale, k * (V / n)^(1/3)), V the bounding-box volume: it grows with
    the Gaussian spacing. k <= 0 keeps max_scale."""
    if k <= 0:
        return max_scale
    vol = float(np.prod(np.array(maxes) - np.array(mins)))
    cap = max(max_scale, k * (vol / n) ** (1.0 / 3.0))
    print(f"max_scale {cap:.4f} (k={k}, N={n}, V={vol:.3f})")
    return cap


def write_max_scale(model_dir, max_scale):
    with open(os.path.join(model_dir, "max_scale"), "w") as f:
        f.write(f"{max_scale}\n")


def read_max_scale(model_dir, default=0.02):
    path = os.path.join(model_dir, "max_scale")
    return float(open(path).read()) if os.path.exists(path) else default


def write_render_rule(model_dir, occ_t, floor):
    with open(os.path.join(model_dir, "render_rule"), "w") as f:
        f.write(f"{occ_t} {floor}\n")


def write_train_seconds(model_dir, seconds, save_seconds):
    """The training time to quote (the iteration loop's wall clock minus saves): printed as
    TRAIN_SECONDS and written to train_seconds.txt."""
    print(f"TRAIN_SECONDS: {seconds:.2f} (save {save_seconds:.2f})", flush=True)
    with open(os.path.join(model_dir, "train_seconds.txt"), "w") as f:
        f.write(f"{seconds:.3f}\n")


def read_render_rule(model_dir, legacy):
    """The model's (occ_t, floor); models from before render_rule existed use `legacy`."""
    path = os.path.join(model_dir, "render_rule")
    if not os.path.exists(path):
        return legacy
    occ_t, floor = (float(x) for x in open(path).read().split()[:2])
    return occ_t, floor
