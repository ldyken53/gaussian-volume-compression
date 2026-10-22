from gvcommon.gaussian_model import GaussianModel as _GaussianModel
from gvraster import ops


class GaussianModel(_GaussianModel):
    """The shared Gaussian model (gvcommon), stepped by gvraster's Adam kernel, with what the
    tile renderer reads from it: the volume bounds and the render rule."""

    multi_adam_ = staticmethod(ops.multi_adam_)

    def __init__(self):
        super().__init__()
        # Render rule: a cell is empty where aw <= occ_t, else S / max(aw, floor). Training uses
        # occ_t = 0; a trained model's rule is in its render_rule file.
        self.occ_t = 0.0
        self.floor = 0.01

    def _set_bounds(self, mesh):
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds if mesh else (0, 1, 0, 1, 0, 1)
        self.mins = [xmin, ymin, zmin]
        self.maxes = [xmax, ymax, zmax]

    def load_ply(self, path, mesh=None):
        super().load_ply(path)
        print(f"Number of points at initialisation : {self._xyz.shape[0]}")
        self._set_bounds(mesh)
        self.mesh = mesh

    def densify_and_prune(
        self,
        max_grad,
        min_weight,
        new_weight,
        empty_points,
        empty_values,
        prune_only=False,
        num_densify=None,
        use_3dgs=False,
    ):
        if not prune_only:
            if not use_3dgs:
                self.densify_in_place(
                    empty_points, empty_values, min_weight, new_weight
                )
                return
            # Called before the Adam step, so .grad is this iteration's.
            if self._xyz.grad is not None:
                grads = self._xyz.grad.detach().clone()
                remaining = num_densify
                added = self.densify_and_clone(
                    grads, max_grad, 1.0, max_new_points=remaining
                )
                if remaining is not None:
                    remaining -= added
                if remaining is None or remaining > 0:
                    self.densify_and_split(
                        grads, max_grad, 1.0, max_new_points=remaining
                    )
        self.prune_points((self.get_weight < min_weight).squeeze())
