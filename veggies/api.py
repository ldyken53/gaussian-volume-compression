"""Using a trained model: load_model(model_dir).sample(points)."""

import torch

from veggies import files, sampling
from veggies.model import VeggiesModel


class TrainedModel:
    def __init__(self, model_dir, iteration=-1):
        ply = files.checkpoint(model_dir, iteration)
        if ply is None:
            raise FileNotFoundError(f"no saved model in {model_dir}")
        info = files.read_info(model_dir)
        self.kind = info.get("kind")
        self.normalization = info.get("normalization")
        self.rule = files.read_render_rule(
            model_dir, sampling.legacy_rule(self.kind == "surface")
        )
        self.model = VeggiesModel()
        self.model.max_scale = files.read_max_scale(model_dir)
        self.model.load_ply(ply)

    def sample(self, points):
        """Values at points (N, 3) in the source volume's coordinates, as an (N,) CUDA tensor, NaN
        where the model is undefined. Interior models (VEGGIES+S) are defined beyond the volume's
        domain; clip their samples with the volume's exterior surface."""
        points = torch.as_tensor(points, dtype=torch.float32, device="cuda")
        norm = self.normalization
        if norm:
            points = (points - norm["offset"]) / norm["scale"]
        with torch.no_grad():
            sampling.set_points(points.contiguous())
            values = sampling.sample_points(self.model, self.rule)[0]
        values = torch.where(
            values == -1, torch.full_like(values, float("nan")), values
        )
        if norm:
            values = (
                values * (norm["value_max"] - norm["value_min"]) + norm["value_min"]
            )
        return values


def load_model(model_dir, iteration=-1):
    return TrainedModel(model_dir, iteration)
