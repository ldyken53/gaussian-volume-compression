#!/bin/bash
# The paper's final models: interior (VEGGIES+S) and surface (VEGGIES) models of each mesh and
# structured models of each grid, at every compression ratio. One GPU, serial.
# usage: run_final.sh <gpu> [interior|surface|structured|all] [dataset ...]
#   meshes: mito sf1 valley rbl eq; grids: vertebra miranda chameleon richtmyer
# Models go to output/<kind>_<dataset>_<ratio>, evaluations to bench_logs/eval_<tag>.json.
cd "$(dirname "$0")"
G=${1:-0}; MODE=${2:-all}; shift $(( $# < 2 ? $# : 2 ))
DS=${*:-"mito sf1 valley rbl eq vertebra miranda chameleon richtmyer"}
export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$PWD/gvraster:$PYTHONPATH
A=/lus/eagle/projects/dist_relational_alg/ldyken53
U=/lus/eagle/projects/UIC-HPC/ldyken53
declare -A SRC=([mito]=$A/mito.vtu [sf1]=$A/sf1.vtu [valley]=$U/valley.vtu [rbl]=$U/rbl.vtu
                [eq]=$A/earthquakenorm.vtu [vertebra]=/home/ldyken53/vertebra.vtk
                [miranda]=$A/miran.vtk [chameleon]=$A/chame.vtk [richtmyer]=$A/richt.vtk)
declare -A RATIOS=([vertebra]="1024 256 64 4096 16384")
# Earthquake is too large to triangulate here, so it trains on precomputed sample sets.
declare -A SETS=([eq_interior]=$U/earthquakebig [eq_surface]=$A/earthquake2big)
mkdir -p bench_logs

run() {  # kind dataset ratio
  local t=$1_$2_$3 src=${SRC[$2]} flags="--kind $1 --ratio $3"
  [ $1 = structured ] && flags="$flags --is_scaled"
  [ -n "${SETS[$2_$1]}" ] && flags="$flags --precomputed_samples ${SETS[$2_$1]}"
  python train.py -s $src -m output/$t $flags > bench_logs/$t.log 2>&1
  echo "EXIT=$?" >> bench_logs/$t.log
  # Grids report the continuous PSNR, the structured metric of record.
  local ev=""; [ $1 = structured ] && ev="--is_scaled --continuous 2"
  python eval.py -s $src --models output/$t $ev --out bench_logs/eval_$t.json > bench_logs/eval_$t.log 2>&1
}

for d in $DS; do
  if [[ ${SRC[$d]} == *.vtk ]]; then
    kinds="structured"
  else
    kinds="interior surface"
  fi
  for kind in $kinds; do
    [ $MODE = all ] || [ $MODE = $kind ] || continue
    for r in ${RATIOS[$d]:-1024 256 64}; do run $kind $d $r; done
  done
done
echo "FINAL RUNS DONE"
