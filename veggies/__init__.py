"""VEGGIES: Volume Encoding Gaussians with Geometry-Integration and Explicit Sampling.

from veggies import train, load_model
train("volume.vti", "out/model", ratio=256)
values = load_model("out/model").sample(points)
"""

from veggies.api import TrainedModel, load_model
from veggies.model import VeggiesModel
from veggies.train import train

__all__ = ["TrainedModel", "VeggiesModel", "load_model", "train"]
