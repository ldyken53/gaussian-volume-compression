#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define OPACITY 0.01

// Two independent thresholds (they used to be one WEIGHT_CUTOFF constant, with the
// support expressed as TRUNC_FRAC * WEIGHT_CUTOFF):
//
// DENOM_FLOOR: a cell renders accumulated_value / max(aw, DENOM_FLOOR) (SOFT_DENOM below),
// so a weakly covered cell fades toward 0 instead of dividing by a tiny aw. It does NOT
// decide coverage under SOFT_CUTOFF 1: any aw > 0 is covered, only aw == 0 is -1. (Under
// SOFT_CUTOFF 0 it is also the hard coverage threshold.) Only 0.01 and no floor have been
// tested: no floor is +0.01..+0.24 dB on 5 cells but loses richtmyer 1024x coverage without
// the log barrier; rendering a floor-trained model without it is free (+-0.01 dB).
#define DENOM_FLOOR 1e-2   // runtime override: DENOM_FLOOR env

// SUPPORT_T: a Gaussian contributes to a cell only where w * exp(-q/2) >= SUPPORT_T
// (GAUSS_CUTOFF 1), i.e. within its own m = sqrt(2 ln(w / SUPPORT_T)) sigma ellipsoid, and
// is binned into the tiles that ellipsoid's AABB touches. Runtime override: SUPPORT_T env
// (legacy: the old TRUNC_FRAC env, read as SUPPORT_T = TRUNC_FRAC * 0.01). A model must be
// rendered at the value it was trained with; train.py records it in <model>/raster_env.
// 1e-4 since the per-Gaussian cutoff: 1e-3 cost 7.6 dB summed over the 14 benchmark cells
// (miranda 64x -5.4: its points carry little accumulated weight, so an absolute 1e-3 cut
// drops a large share of it) and 1e-4 matched the old q <= 28-per-tile rule's speed at
// +3.0 dB. Models with no raster_env predate it: full_psnr.py evaluates them at 1e-3 with
// GAUSS_CUTOFF=0.
#define SUPPORT_T 1e-4

// 0 = hard cutoff (original): a cell whose accumulated weight is under DENOM_FLOOR is
// declared empty, written as -1.0, reports weight 0, and receives no gradient. The FN
// barrier in train.py exists purely to compensate for that missing gradient.
// 1 = clamped denominator: any cell reached by at least one Gaussian renders
// accumulated_value / max(aw, DENOM_FLOOR). The 1/aw blow-up that the threshold was
// guarding against is bounded at 1/DENOM_FLOOR instead of being cut off, so weakly
// covered cells stay differentiable and render a small bounded value rather than the -1
// sentinel. -1.0 then means only "no Gaussian reaches this cell", where no gradient
// exists to give anyway.
// NOT metric-compatible with 0: cells that used to contribute a full-magnitude (-1 - gt)
// error now contribute a small one, so PSNR rises partly because the sentinel is gone.
// It also changes the weights output (true small aw instead of 0), which feeds the FN mask.
#define SOFT_CUTOFF 1
// A/B against SOFT_CUTOFF 0, 2026-09-04, 4000 iterations, -1 empty sentinel:
// at the stock fn_reg 0.02 the two are indistinguishable (chame 64x 49.446 vs 49.448,
// 256x 48.022 vs 48.050, 1024x 46.620 vs 46.604, vert 64x 42.381 vs 42.380, miran
// 1024x 33.664 vs 33.783). What the soft cutoff actually buys is the LOW-fn_reg
// regime: miranda 1024x at fn_reg 0.0005 reads 35.210 soft and **8.32** hard, with
// 990,790 dead cells. Per-dataset fn_reg tuning is worth +0.6 to +1.6 dB and it is
// only reachable from here -- see gaussian-volume-fn-reg-per-dataset.

// Denominator floor under SOFT_CUTOFF. Two alternatives were built and measured worse
// (2026-09-04, 4000 iterations), so neither is kept:
// - aw + DENOM_FLOOR (smooth, no kink at the floor): enlarges EVERY cell's
//   denominator, so every value is biased low.
// - The exact adjoint of max(): below the cutoff D is constant, so dR/dw_i = v_i/D and
//   the backward's -R term should be dropped there (see backward.cu). It costs 0.2-0.8 dB:
//   m64 52.615 vs 53.371, m1k 34.765 vs 35.297, r1k 29.726 vs 29.935, c64 49.386 vs 49.396.
// The shipped backward is therefore deliberately inexact below the cutoff. Do not "fix" it.
// The floor is a kernel argument: DENOM_FLOOR is the default, the DENOM_FLOOR env overrides
// it (runtime_denom_floor() in auxiliary.h, read once per process; 0 = no floor).
#define SOFT_DENOM(aw, fl) (max((aw), (fl)))

// Backward-render speed switches (results unchanged up to float summation order).
// SKIP_EMPTY: a (Gaussian, tile) pair whose support reaches no cell of the tile skips its
// 11 global atomics -- the tile-overlap test is conservative, so at high Gaussian density
// many binned pairs contribute exactly zero.
#ifndef SKIP_EMPTY
#define SKIP_EMPTY 1
#endif
// PACKED_GRAD: accumulate each Gaussian's 11 render gradients into one 12-float row
// (conic[6], mean[3], value, weight, pad) so a pair's atomics hit one cache line instead
// of four arrays; preprocess backward reads the conic part at stride 12.
// FUSE_ACT: the rasterizer takes the RAW parameters (log-scales, unnormalised quaternions,
// pre-sigmoid value and weight) and applies the activations itself -- exp + soft cap
// M tanh(||s||/M)/||s|| with M passed in the scale_modifier slot, normalise, sigmoid -- in
// forward preprocess, and their chain rule in backward preprocess, which also writes the
// mean / value / weight gradients straight from the packed row. Same float expressions as
// fused_ops.cu's actForward / actBackward; removes those two kernels and their tensors.
// GAUSS_CUTOFF (runtime env, default 1): each Gaussian contributes to a cell only where
// w * exp(-q/2) >= T = SUPPORT_T, i.e. q <= qmax = 2 ln(w / T) -- exactly the ellipsoid the
// tile AABB is built from. 2 = weight-independent, G >= T. 0 = the old rule. Without it
// the per-cell test is the blanket q <= 28, so a Gaussian reaches out to 5.3 sigma inside
// any tile its 1e-3 ellipsoid touches and its support depends on tile alignment; with it
// the support is a per-Gaussian ellipsoid a renderer can reproduce exactly.
#ifndef GAUSS_CUTOFF
#define GAUSS_CUTOFF 1
#endif
#ifndef FUSE_ACT
#define FUSE_ACT 1
#endif
#ifndef BWD_THREADS
#define BWD_THREADS 256
#endif
#ifndef PACKED_GRAD
#define PACKED_GRAD 1
#endif

#endif
