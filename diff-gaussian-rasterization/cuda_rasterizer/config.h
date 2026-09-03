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

#endif
