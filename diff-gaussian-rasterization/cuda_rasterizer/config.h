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
// as m^3. Purely a speed/quality dial -- unlike WEIGHT_CUTOFF it does not
// change which cells count as covered.
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

// Shape of the denominator floor under SOFT_CUTOFF.
// 0 = max(aw, WEIGHT_CUTOFF): kinked -- the derivative jumps at aw == WEIGHT_CUTOFF.
// 1 = aw + WEIGHT_CUTOFF: smooth everywhere, monotone, same asymptotics (-> aw when
//     aw >> cutoff, -> aw/cutoff when aw << cutoff), and no derivative discontinuity for
//     the optimiser to sit on. Both still bias weakly covered cells low.
#define SOFT_SMOOTH 0

// Below the cutoff the forward computes the DAMPED render S/c, whose true adjoint is
// v_i/D. Keeping r_term (SOFT_TRUE_ADJOINT 0) instead applies the adjoint of the
// NORMALISED render S/aw, which is deliberately not the derivative of what the forward
// computes. Measured, that mismatch is worth 0.2-0.8 dB: the true adjoint pushes every
// weight the same way ("someone cover this cell"), while the normalised one pushes w_i up
// where v_i > R and down where v_i < R ("these Gaussians should cover it").
// max+corrected: m64 52.615, m1k 34.765, r1k 29.726, c64 49.386
// max+mismatched: m64 53.371, m1k 35.297, r1k 29.935, c64 49.396
// Set to 1 for the mathematically consistent version; it is worse. Do not "fix" this.
#define SOFT_TRUE_ADJOINT 0

#if SOFT_SMOOTH
#define SOFT_DENOM(aw) ((aw) + (float)WEIGHT_CUTOFF)
#else
#define SOFT_DENOM(aw) (max((aw), (float)WEIGHT_CUTOFF))
#endif

#endif
