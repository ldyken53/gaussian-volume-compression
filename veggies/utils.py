import random
import sys
from datetime import datetime

import numpy as np
import torch


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def exponential_lr(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    def lr(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        return delay_rate * np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)

    return lr


def build_rotation(q):
    """Rotation matrices of quaternions (w, x, y, z), normalized first."""
    norm = torch.sqrt(
        q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    )
    q = q / norm[:, None]
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros((q.size(0), 3, 3), device="cuda")
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(scales, rotations):
    """R diag(s): its outer product with itself is the covariance."""
    L = torch.zeros((scales.shape[0], 3, 3), dtype=torch.float, device="cuda")
    L[:, 0, 0] = scales[:, 0]
    L[:, 1, 1] = scales[:, 1]
    L[:, 2, 2] = scales[:, 2]
    return build_rotation(rotations) @ L


def morton_codes(xyz):
    """Morton codes of points in the unit cube."""
    norm = (xyz * ((1 << 21) - 1)).long()

    def spread(n):
        n = n & 0x1FFFFF
        n = (n | (n << 32)) & 0x1F00000000FFFF
        n = (n | (n << 16)) & 0x1F0000FF0000FF
        n = (n | (n << 8)) & 0x100F00F00F00F00F
        n = (n | (n << 4)) & 0x10C30C30C30C30C3
        n = (n | (n << 2)) & 0x1249249249249249
        return n

    return spread(norm[:, 0]) | (spread(norm[:, 1]) << 1) | (spread(norm[:, 2]) << 2)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(torch.device("cuda:0"))


def timestamp_stdout(quiet):
    out = sys.stdout

    class Timestamped:
        def write(self, text):
            if quiet:
                return
            if text.endswith("\n"):
                stamp = datetime.now().strftime("%d/%m %H:%M:%S")
                text = text.replace("\n", f" [{stamp}]\n")
            out.write(text)

        def flush(self):
            out.flush()

    sys.stdout = Timestamped()
