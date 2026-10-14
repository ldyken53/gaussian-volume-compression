#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define OPACITY 0.01

// Cells whose total accumulated weight falls below this are treated as empty
// (written as -1.0) and contribute no gradient. This is a semantic threshold:
// it defines what the model considers covered, so changing it changes results.
#define WEIGHT_CUTOFF 1e-2

// Where a single Gaussian's own contribution is considered to have died out,
// as a fraction of WEIGHT_CUTOFF. Sets the truncation radius m in forward.cu:
// m = sqrt(-2 ln(TRUNC_FRAC * WEIGHT_CUTOFF / w)), and per-Gaussian cost goes
// as m^3. Under SOFT_CUTOFF this is NOT only a speed/quality dial: a cell is
// written -1 exactly when no Gaussian's truncated support reaches it, so raising
// TRUNC_FRAC shrinks coverage too. Overridable at runtime via the TRUNC_FRAC env
// var; a model must be rendered at the value it was trained with.
#define TRUNC_FRAC 0.1

// 0 = hard cutoff (original): a cell whose accumulated weight is under WEIGHT_CUTOFF is
// declared empty, written as -1.0, reports weight 0, and receives no gradient. The FN
// barrier in train.py exists purely to compensate for that missing gradient.
// 1 = clamped denominator: any cell reached by at least one Gaussian renders
// accumulated_value / max(aw, WEIGHT_CUTOFF). The 1/aw blow-up that the threshold was
// guarding against is bounded at 1/WEIGHT_CUTOFF instead of being cut off, so weakly
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
// - aw + WEIGHT_CUTOFF (smooth, no kink at the cutoff): enlarges EVERY cell's
//   denominator, so every value is biased low.
// - The exact adjoint of max(): below the cutoff D is constant, so dR/dw_i = v_i/D and
//   the backward's -R term should be dropped there (see backward.cu). It costs 0.2-0.8 dB:
//   m64 52.615 vs 53.371, m1k 34.765 vs 35.297, r1k 29.726 vs 29.935, c64 49.386 vs 49.396.
// The shipped backward is therefore deliberately inexact below the cutoff. Do not "fix" it.
#define SOFT_DENOM(aw) (max((aw), (float)WEIGHT_CUTOFF))

#endif
