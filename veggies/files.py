"""The files of a model directory. Sampling a model needs its max_scale (the scale cap) and its
render_rule; veggies.json records its type and how the source volume was normalized."""

import glob
import json
import os


def write_cfg(model_dir, args):
    with open(os.path.join(model_dir, "cfg_args"), "w") as f:
        f.write(str(args))


def write_max_scale(model_dir, max_scale):
    with open(os.path.join(model_dir, "max_scale"), "w") as f:
        f.write(f"{max_scale}\n")


def _read(model_dir, name):
    """The text of a model file, or None if it is missing."""
    path = os.path.join(model_dir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read()


def read_max_scale(model_dir, default=0.02):
    text = _read(model_dir, "max_scale")
    return float(text) if text is not None else default


def write_render_rule(model_dir, threshold, floor):
    with open(os.path.join(model_dir, "render_rule"), "w") as f:
        f.write(f"{threshold} {floor}\n")


def read_render_rule(model_dir, legacy):
    text = _read(model_dir, "render_rule")
    if text is None:
        return legacy
    threshold, floor = (float(x) for x in text.split()[:2])
    return threshold, floor


def write_info(model_dir, kind, normalization):
    with open(os.path.join(model_dir, "veggies.json"), "w") as f:
        json.dump({"kind": kind, "normalization": normalization._asdict()}, f, indent=1)


def read_info(model_dir):
    text = _read(model_dir, "veggies.json")
    return json.loads(text) if text is not None else {}


def write_train_seconds(model_dir, seconds, save_seconds):
    print(f"TRAIN_SECONDS: {seconds:.2f} (save {save_seconds:.2f})", flush=True)
    with open(os.path.join(model_dir, "train_seconds.txt"), "w") as f:
        f.write(f"{seconds:.3f}\n")


def checkpoint(model_dir, iteration=-1):
    """The ply saved at iteration (-1: the highest past 0), or None."""
    if iteration != -1:
        path = os.path.join(
            model_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"
        )
        return path if os.path.exists(path) else None
    plys = sorted(
        glob.glob(f"{model_dir}/point_cloud/iteration_*/point_cloud.ply"),
        key=lambda x: int(x.split("iteration_")[1].split("/")[0]),
    )
    plys = [p for p in plys if "iteration_0/" not in p]
    return plys[-1] if plys else None
