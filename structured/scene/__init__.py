import os

from scene.dataset_readers import readData
from scene.gaussian_model import GaussianModel


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args,
        gaussians: GaussianModel,
        normalized=False,
        fraction=0.01,
        max_init=None,
    ):
        self.model_path = args.model_path
        self.gaussians = gaussians
        if not (
            os.path.exists(args.source_path)
            and args.source_path.lower().endswith((".vtk", ".vtu"))
        ):
            raise SystemExit(f"{args.source_path}: expected a .vtk or .vtu volume")
        mesh, pcd = readData(args.source_path, fraction, normalized, max_init)
        self.gaussians.create_from_pcd(pcd, mesh)

    def save(self, iteration):
        self.gaussians.save_ply(
            os.path.join(
                self.model_path, f"point_cloud/iteration_{iteration}", "point_cloud.ply"
            )
        )
