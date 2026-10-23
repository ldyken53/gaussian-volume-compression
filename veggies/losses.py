"""The training loss: L2 reconstruction plus the false negative (FN) and false positive (FP)
terms. Structured and interior models keep samples defined with a log barrier on the influence;
surface models classify samples around the defined threshold T with an FN hinge inside the mesh
and an FP hinge outside it, both scaled by the data loss."""

import torch

from gvraster.ops import volume_loss
from veggies import sampling

SUPPORT_T = 1e-4  # a Gaussian reaches a sample only where w G >= SUPPORT_T


class TrainingLoss:
    def __init__(self, args, threshold):
        self.args = args
        self.surface = args.kind == "surface"
        if args.fn_mode == "log":
            self.fn_tau = args.fn_thresh
        else:
            self.fn_tau = ((1.0 + args.fn_margin) if self.surface else 1.1) * threshold
        self.fp_tau = (
            (1.0 - args.fp_margin) * threshold if args.fp_mode == "hinge" else SUPPORT_T
        )
        self.fp_on = self.surface and args.fp_mode != "margin"
        # Grid models fit every lattice cell, an undefined one's -1 included; mesh models skip
        # samples without ground truth or influence.
        self.mask_data = args.kind != "structured"
        self.frozen, self.scale_sum, self.scale_n = None, 0.0, 0

    def __call__(self, values, gt, influence, model, data, iteration):
        """(loss, data loss, fn, fp)."""
        a = self.args
        lambda_fn, lambda_fp, rel_pow = a.lambda_fn, a.lambda_fp, a.hinge_rel_pow
        if self.frozen is not None:
            lambda_fn, lambda_fp, rel_pow = (
                a.lambda_fn * self.frozen,
                a.lambda_fp * self.frozen,
                0.0,
            )
        loss, data_loss, fn, fp = volume_loss(
            values,
            gt,
            influence,
            l1=self.surface,
            fn_mode=a.fn_mode,
            fn_tau=self.fn_tau,
            lambda_fn=lambda_fn,
            lambda_fp=lambda_fp if self.fp_on else 0.0,
            fp_tau=self.fp_tau,
            fp_mode=a.fp_mode if self.fp_on else "lin",
            norm_all=a.fnfp_norm == "all",
            rel_pow=rel_pow,
            mask_data=self.mask_data,
        )
        hinge_scale = data_loss  # the L1 data loss of surface models
        if self.surface:
            # Values are fit with L2 and a light log barrier; the call above supplies the hinges.
            l2 = volume_loss(
                values,
                gt,
                influence,
                l1=False,
                fn_mode="log",
                fn_tau=a.fn_thresh,
                lambda_fn=0.0001,
            )
            data_loss = l2[1]
            loss = l2[0] + fn + fp
        if a.hinge_freeze > 0 and a.hinge_rel_pow > 0 and self.frozen is None:
            if iteration > a.hinge_freeze - 50:
                self.scale_sum = self.scale_sum + hinge_scale.detach()
                self.scale_n += 1
            if iteration >= a.hinge_freeze:
                self.frozen = (
                    float(self.scale_sum / max(self.scale_n, 1)) ** a.hinge_rel_pow
                )
                print(
                    f"hinge weights frozen at iteration {iteration}: scale {self.frozen:.3e}"
                )
        if self.surface and data.margin_n_ext is not None:
            fp = (
                a.lambda_fp
                * sampling.margin_fp(model, data.batch, self.fp_tau)
                / data.margin_n_ext[data.batch]
            )
            loss = loss + fp
        if a.densify == "mcmc":
            loss = loss + a.weight_reg * torch.abs(model.weights).mean()
            loss = loss + a.scale_reg * torch.abs(model.scales).mean()
        return loss, data_loss, fn, fp
