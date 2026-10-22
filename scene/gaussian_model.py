from gvcommon.gaussian_model import GaussianModel as _GaussianModel
from gvraster import ops


class GaussianModel(_GaussianModel):
    """The shared Gaussian model (gvcommon), stepped by gvraster's Adam kernel."""

    multi_adam_ = staticmethod(ops.multi_adam_)
