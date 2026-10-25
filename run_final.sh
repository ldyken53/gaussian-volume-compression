#!/bin/bash
# The paper's final models: interior (VEGGIES+S) and surface (VEGGIES) models of each mesh and
# structured models of each grid, at every compression ratio. One GPU, serial.
# usage: run_final.sh <gpu> [interior|surface|structured|all] [dataset ...]
#   meshes: mito sf1 valley rbl eq impact; grids: vertebra miranda chameleon richtmyer
# Models go to $OUT/<kind>_<dataset>_<ratio> (default output/), logs and evaluations to $LOGS
# (default bench_logs/). With PRINT=1 nothing runs: each run prints "tag<TAB>train<TAB>eval".
cd "$(dirname "$0")"
G=${1:-0}; MODE=${2:-all}; shift $(( $# < 2 ? $# : 2 ))
DS=${*:-"mito sf1 valley rbl eq impact vertebra miranda chameleon richtmyer"}
OUT=${OUT:-output} LOGS=${LOGS:-bench_logs}
export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$PWD/gvraster:$PYTHONPATH
A=/lus/eagle/projects/dist_relational_alg/ldyken53
U=/lus/eagle/projects/UIC-HPC/ldyken53
declare -A SRC=([mito]=$A/mito.vtu [sf1]=$A/sf1.vtu [valley]=$U/valley.vtu [rbl]=$U/rbl.vtu
                [eq]=$A/earthquakenorm.vtu [impact]=$A/impactnorm.vtu
                [vertebra]=/home/ldyken53/vertebra.vtk [miranda]=$A/miran.vtk
                [chameleon]=$A/chame.vtk [richtmyer]=$A/richt.vtk)
declare -A RATIOS=([vertebra]="1024 256 64 16 4096 16384" [valley]="1024 256 64 16 4096 16384")
# Earthquake and impact are too large to triangulate here, so they train on precomputed sample
# sets (tools/gen_samples.py defaults; --surface for the surface sets):
#   gen_samples.py earthquakenorm.vtu $U/earthquake_i50 --batches 100 (_i50s with --surface)
#   gen_samples.py impactnorm.vtu $U/impact_i50p --batches 200 (_i50ps with --surface)
# impact also evaluates on precomputed points (tools/precompute_tet.py impactnorm.vtu
# eval_sets/impactprobe) and trains for 16,000 iterations: at 5,000 it is 6-8 dB lower.
declare -A SETS=([eq_interior]=$U/earthquake_i50 [eq_surface]=$U/earthquake_i50s
                 [impact_interior]=$U/impact_i50p [impact_surface]=$U/impact_i50ps)
declare -A EVAL_SET=([impact]=eval_sets/impactprobe)
declare -A ITERS=([impact]=16000)
[ -n "$PRINT" ] || mkdir -p $LOGS

run() {  # kind dataset ratio
  local t=$1_$2_$3 src=${SRC[$2]} flags="--kind $1 --ratio $3"
  [ $1 = structured ] && flags="$flags --is_scaled"
  [ -n "${SETS[$2_$1]}" ] && flags="$flags --precomputed_samples ${SETS[$2_$1]}"
  [ -n "${ITERS[$2]}" ] && flags="$flags --iterations ${ITERS[$2]}"
  # Grids report the continuous PSNR, the structured metric of record.
  local ev=""; [ $1 = structured ] && ev="--is_scaled --continuous 2"
  [ -n "${EVAL_SET[$2]}" ] && ev="--eval_set ${EVAL_SET[$2]}"
  local train="python train.py -s $src -m $OUT/$t $flags"
  local eval="python eval.py -s $src --models $OUT/$t $ev --out $LOGS/eval_$t.json"
  if [ -n "$PRINT" ]; then printf '%s\t%s\t%s\n' "$t" "$train" "$eval"; return; fi
  $train > $LOGS/$t.log 2>&1
  echo "EXIT=$?" >> $LOGS/$t.log
  $eval > $LOGS/eval_$t.log 2>&1
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
[ -n "$PRINT" ] || echo "FINAL RUNS DONE"
