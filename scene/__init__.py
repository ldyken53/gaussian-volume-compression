import os

from arguments import ModelParams
from scene.dataset_readers import readData
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration


class Scene:

    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        normalized=False,
        fraction=0.01
    ):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # Only treat model_path as a resume when the checkpoint is actually there.
        # train.py passes load_iteration=0 whenever --model_path is set, which would
        # otherwise make every named fresh run try to load a ply that does not exist.
        if load_iteration is not None and load_iteration >= 0:
            if not os.path.exists(os.path.join(
                    self.model_path, "point_cloud", f"iteration_{load_iteration}",
                    "point_cloud.ply")):
                load_iteration = None
        if load_iteration is not None:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        mesh = None
        if self.loaded_iter is None or load_iteration == -1 or load_iteration > 0:
            if os.path.exists(args.source_path) and args.source_path.lower().endswith(('.vtk', '.vtu')):
                mesh, pcd = readData(args.source_path, fraction, normalized)
            else:
                assert False, "Could not recognize scene type!"

        if self.loaded_iter is not None:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                ),
                mesh,
                args.train_test_exp,
            )
        else:
            self.gaussians.create_from_pcd(
                pcd,
                mesh,
            )

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, f"point_cloud/iteration_{iteration}"
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))