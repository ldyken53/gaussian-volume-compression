#!/bin/bash
# Final models: interior and surface models of each mesh and structured models of each grid, at
# every final compression ratio. One GPU, serial.
# Build once: (cd gvraster && python setup.py build_ext --inplace)
# usage: run_final.sh <gpu> [interior|surface|structured|all] [dataset ...]
#   meshes: mito sf1 valley rbl eq; grids: vertebra miranda chameleon richtmyer
# Outputs output/IN_<d>_<r>, output/SU_<d>_s<r> and output/ST_<d>_<r>, evaluations in
# bench_logs/eval_<tag>.json.
cd "$(dirname "$0")"
G=${1:-0}; MODE=${2:-all}; shift $(( $# < 2 ? $# : 2 ))
DS=${*:-"mito sf1 valley rbl eq vertebra miranda chameleon richtmyer"}
export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$PWD/gvraster:$PYTHONPATH
A=/lus/eagle/projects/dist_relational_alg/ldyken53
U=/lus/eagle/projects/UIC-HPC/ldyken53
declare -A SRC=([mito]=$A/mito.vtu [sf1]=$A/sf1.vtu [valley]=$U/valley.vtu [rbl]=$U/rbl.vtu
                [eq]=$A/earthquakenorm.vtu [vertebra]=/home/ldyken53/vertebra.vtk
                [miranda]=$A/miran.vtk [chameleon]=$A/chame.vtk [richtmyer]=$A/richt.vtk)
# Meshes: final Gaussian count per ratio 1024x / 256x / 64x: storage / (ratio x 48 bytes).
declare -A CAP=([mito]="5015 20058 80232" [sf1]="12663 50653 202611" [valley]="23925 95700 382800"
                [rbl]="3539 14157 56628" [eq]="87659 350636 1402545")
# Grids: the ratios trained (--ratio sets the count to nodes / (12 x ratio)).
declare -A RATIOS=([vertebra]="1024 256 64 4096 16384" [miranda]="1024 256 64"
                   [chameleon]="1024 256 64" [richtmyer]="1024 256 64")
mkdir -p bench_logs

run() {  # tag src [train flags...]
  local t=$1 src=$2; shift 2
  python train.py -s $src -m output/$t "$@" > bench_logs/$t.log 2>&1
  echo "EXIT=$?" >> bench_logs/$t.log
  local ev=""
  [[ " $* " == *" --encode_surface "* ]] && ev="--surface output/$t"
  # Grids report the continuous PSNR, the structured metric of record.
  [[ $src == *.vtk ]] && ev="--is_scaled --continuous 2"
  python eval.py -s $src --models output/$t $ev --out bench_logs/eval_$t.json \
    > bench_logs/eval_$t.log 2>&1
}

for d in $DS; do
  if [ -n "${RATIOS[$d]}" ]; then
    if [ $MODE = all ] || [ $MODE = structured ]; then
      for r in ${RATIOS[$d]}; do run ST_${d}_$r ${SRC[$d]} --is_scaled --ratio $r; done
    fi
    continue
  fi
  caps=(${CAP[$d]}); ei=""; es=""
  # Earthquake trains on precomputed sample sets.
  [ $d = eq ] && ei="--precomputed_samples $U/earthquakebig" && es="--precomputed_samples $A/earthquake2big"
  for k in 0 1 2; do
    r=$((1024 >> (2 * k)))
    if [ $MODE = all ] || [ $MODE = interior ]; then
      run IN_${d}_$r ${SRC[$d]} --cap_max ${caps[$k]} $ei
    fi
    if [ $MODE = all ] || [ $MODE = surface ]; then
      run SU_${d}_s$r ${SRC[$d]} --cap_max ${caps[$k]} --encode_surface $es
    fi
  done
done
echo "FINAL RUNS DONE"
