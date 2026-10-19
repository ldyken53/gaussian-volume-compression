#!/bin/bash
# usage: build_cfg.sh <tag> <SOFT_CUTOFF 0|1>  -- build the working tree with config.h SOFT_CUTOFF set
set -e
source /home/ldyken53/fnexp/env.sh >/dev/null 2>&1
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.6.3
GCC13=/soft/spack/base/0.7.1/install/linux-sles15-x86_64/gcc-12.3.0/gcc-13.2.0-mqxcg3ctxvwlxbwpuhbegxptlpfa2u74
export PATH=$CUDA_HOME/bin:$GCC13/bin:$PATH CC=$GCC13/bin/gcc CXX=$GCC13/bin/g++ LD_LIBRARY_PATH=$GCC13/lib64:$LD_LIBRARY_PATH
T=$1; W=/lus/eagle/projects/UIC-HPC/ldyken53/rastbuild/$T; O=/home/ldyken53/gaussian-volume/rast_$T
rm -rf $W $O; mkdir -p $W; cp -r /home/ldyken53/gaussian-volume/diff-gaussian-rasterization $W/src
rm -rf $W/src/build $W/src/build_fast $W/src/*.egg-info; cd $W/src
sed -i "s/^#define SOFT_CUTOFF [01]/#define SOFT_CUTOFF $2/" cuda_rasterizer/config.h
for d in "${@:3}"; do sed -i "0,/^#define SOFT_CUTOFF/s//#define $d\n#define SOFT_CUTOFF/" cuda_rasterizer/config.h; done; grep -n "MIN_BLOCKS" cuda_rasterizer/config.h || true
grep -n "^#define SOFT_CUTOFF\|^#define SUPPORT_T" cuda_rasterizer/config.h
MAX_JOBS=8 pip install --no-build-isolation --no-deps --target $O . > $W/build.log 2>&1
echo "BUILD $T OK"
