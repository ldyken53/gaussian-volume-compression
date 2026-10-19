#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

// A Gaussian reaches a sample only where w * exp(-q/2) >= SUPPORT_T.
#define SUPPORT_T 1e-4f
// Hard cutoff (surface models): aw <= WEIGHT_CUTOFF is empty, else S / aw.
// Soft cutoff (interior models): empty only at aw == 0, else S / max(aw, WEIGHT_CUTOFF).
#define WEIGHT_CUTOFF 1e-2f

#endif
