from typing import NamedTuple

import numpy as np


class BasicPointCloud(NamedTuple):
    points: np.array
    values: np.array
