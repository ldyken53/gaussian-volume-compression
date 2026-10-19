import os
from argparse import ArgumentParser


class GroupParams:
    pass


class ParamGroup:
    """Exposes every attribute as --name; a leading underscore also adds the -n shorthand."""

    def __init__(self, parser: ArgumentParser, name: str):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            flags = ["--" + key.lstrip("_")]
            if key.startswith("_"):
                flags.append("-" + key[1])
            if type(value) == bool:
                group.add_argument(*flags, default=value, action="store_true")
            else:
                group.add_argument(*flags, default=value, type=type(value))

    def extract(self, args):
        group = GroupParams()
        for k, v in vars(args).items():
            if k in vars(self) or ("_" + k) in vars(self):
                setattr(group, k, v)
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser):
        self._source_path = ""
        self._model_path = ""
        # Final Gaussian count.
        self.cap_max = -1
        super().__init__(parser, "Loading Parameters")

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 5_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.values_lr = 0.005
        self.weight_lr = 0.025
        self.scaling_lr = 0.008
        self.rotation_lr = 0.005
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 16_000
        # --densify_3dgs
        self.percent_dense = 0.01
        # --use_mcmc
        self.noise_lr = 5e5
        self.scale_reg = 0.01
        self.weight_reg = 0.00001
        super().__init__(parser, "Optimization Parameters")
