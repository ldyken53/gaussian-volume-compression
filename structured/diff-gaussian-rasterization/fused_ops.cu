// The volume loss both extensions compile: common/cuda/volume_loss.cuh. This pipeline's data
// term covers every cell (mask_data = 0, see diff_gaussian_rasterization/fused.py).
#include "volume_loss.cuh"
