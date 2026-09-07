#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from argparse import ArgumentParser, Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument(
                        "--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self._depths = ""
        self._resolution = -1
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        self.cap_max = -1
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 16000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.values_lr = 0.0025
        self.weight_lr = 0.025
        # 3DGS's 0.001 is tuned for a 30k-iteration schedule; this project trains
        # ~1k, which leaves the shape parameters frozen mid-descent. 0.005 gained
        # +0.39 to +1.38 dB in struct (ddc4dee); rotation_lr deliberately stays 0.001.
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.densification_interval = 100
        self.weight_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 16_000
        self.densify_grad_threshold = 0.000002
        # Single constant fn_reg, as in struct's 8fc0f36 -- but calibrated for this repo.
        # The FN penalty is absolute while mito's data loss is ~25x smaller than the
        # structured datasets', so struct's 0.5 over-regularises badly here. 0.02 is the
        # smallest weight that still holds zero false negatives at 1024x, and is worth
        # +10.4 / +3.1 / +0.5 dB over the old 0.5 -> 0.02 schedule at 1024x / 256x / 64x.
        self.fn_reg = 0.02
        self.fn_reg2 = -1.0  # fn_reg after densify_from_iter; <0 keeps fn_reg constant.
        self.fp_reg = 0.5
        self.scale_reg = 0.01
        self.weight_reg = 0.00001
        self.noise_lr = 5e5
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
