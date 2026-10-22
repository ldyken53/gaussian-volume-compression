#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

// A Gaussian reaches a sample only where w * exp(-q/2) >= SUPPORT_T.
#define SUPPORT_T 1e-4f
// The render rule (occ_t, floor) is a runtime argument, stored per model in its render_rule file:
// a sample is empty where aw <= occ_t, else it renders S / max(aw, floor). Interior models use
// (0.002, 0.01), surface models (0.01, 0.01).

#endif
