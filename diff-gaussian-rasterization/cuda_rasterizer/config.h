#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

// A cell renders S / max(aw, DENOM_FLOOR); aw == 0 renders the -1 empty sentinel.
#define DENOM_FLOOR 1e-2f
// A Gaussian reaches a cell only where w * exp(-q/2) >= SUPPORT_T.
#define SUPPORT_T 1e-4f

#define BLOCK_X 4
#define BLOCK_Y 4
#define BLOCK_Z 2
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y * BLOCK_Z)
#define BWD_THREADS 256

#endif
