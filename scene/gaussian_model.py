from bvh_diff_gaussian_rasterization import _C
from gvcommon.gaussian_model import GaussianModel as _GaussianModel


class GaussianModel(_GaussianModel):
    """The shared Gaussian model (gvcommon), stepped by this pipeline's extension."""

    multi_adam_ = staticmethod(_C.multi_adam_)
