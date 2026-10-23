#pragma once

// A Gaussian reaches a lattice point only where w G >= SUPPORT_T.
#define SUPPORT_T 1e-4f

// Tiles of 4 x 4 x 2 lattice points.
#define BLOCK_X 4
#define BLOCK_Y 4
#define BLOCK_Z 2
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y * BLOCK_Z)
#define BWD_THREADS 256
