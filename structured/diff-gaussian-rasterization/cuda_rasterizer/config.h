#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

// A Gaussian reaches a cell only where w * exp(-q/2) >= SUPPORT_T.
#define SUPPORT_T 1e-4f
// The render rule (occ_t, floor) is a runtime argument, stored per model in its render_rule file:
// a cell is empty (-1) where aw <= occ_t, else it renders S / max(aw, floor). Models train with
// (0, 0.01) and are deployed at (0.002, 0.01).

#define BLOCK_X 4
#define BLOCK_Y 4
#define BLOCK_Z 2
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y * BLOCK_Z)
#define BWD_THREADS 256

#endif
