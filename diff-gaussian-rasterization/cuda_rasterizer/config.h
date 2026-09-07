#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 4
#define BLOCK_Y 4
#define BLOCK_Z 2
#define OPACITY 0.01
#define WEIGHT_CUTOFF 1e-2

// Truncation radius as a fraction of WEIGHT_CUTOFF: m = sqrt(-2 ln(TRUNC_FRAC *
// WEIGHT_CUTOFF / w)). Purely a speed/quality dial; per-Gaussian cost goes as m^3.
#define TRUNC_FRAC 0.1

// 0 = hard cutoff (original): a cell whose accumulated weight is under WEIGHT_CUTOFF is
// declared empty, written as -1.0, reports weight 0, and receives no gradient. The FN
// barrier in train.py exists purely to compensate for that missing gradient.
// 1 = clamped denominator (ported from struct 8ba3ea2): any cell reached by at least one
// Gaussian renders accumulated_value / max(aw, WEIGHT_CUTOFF), reports its true aw, and
// stays differentiable; the 1/aw blow-up is bounded at 1/WEIGHT_CUTOFF instead of cut to
// zero. -1.0 then means only "no Gaussian reaches this cell". NOT metric-compatible with
// 0: sentinel cells that contributed full-magnitude error now contribute a small one.
// The weights output also changes (true small aw instead of 0), which feeds the FN mask.
// Measured in unstructured (2026-09-04, all new defaults, 2 reps): soft at its best
// fn_reg (0.005) vs hard at 0.02 -- mito 1024x +0.38, mito 256x/64x and valley within
// noise, sf1 1024x -0.45 (the FN barrier gets more aggressive under soft because weak
// cells enter fn_mask). Unlike struct, no config here NEEDS the low-fn_reg regime, so
// the default stays hard. The installed .so is built with 0.
#define SOFT_CUTOFF 0

// Keep struct's measured choices: max() denominator (aw + c biases every cell low), and
// the deliberately mismatched adjoint below the cutoff -- the "consistent" v_i/D form
// measured 0.2-0.8 dB WORSE in struct (see gaussian-volume-soft-cutoff). Do not "fix".
#define SOFT_DENOM(aw) (max((aw), (float)WEIGHT_CUTOFF))

#endif