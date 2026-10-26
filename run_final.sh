#!/bin/bash
# The paper's final models: interior (VEGGIES+S) and surface (VEGGIES) models of each mesh and
# structured models of each grid, at every compression ratio. One GPU, serial.
# usage: [DATA=dir] run_final.sh <gpu> [interior|surface|structured|all] [dataset ...]
#   meshes: mito sf1 valley rbl eq impact; grids: vertebra miranda chameleon richtmyer
# Models go to $OUT/<kind>_<dataset>_<ratio> (default output/), logs and evaluations to $LOGS
# (default bench_logs/). With PRINT=1 nothing runs: each run prints "tag<TAB>train<TAB>eval".
#
# Every input is read from $DATA (default data/):
#   mito.vtu sf1.vtu valley.vtu rbl.vtu earthquakenorm.vtu impactnorm.vtu
#   vertebra.vtk miran.vtk chame.vtk richt.vtk
# Earthquake and impact are too large to triangulate on the GPU, so they train on precomputed
# sample sets (tools/gen_samples.py defaults) and impact evaluates on precomputed points:
#   python tools/gen_samples.py $DATA/earthquakenorm.vtu $DATA/earthquake_i50 --batches 100
#   python tools/gen_samples.py $DATA/earthquakenorm.vtu $DATA/earthquake_i50s --batches 100 --surface
#   python tools/gen_samples.py $DATA/impactnorm.vtu $DATA/impact_i50p --batches 200
#   python tools/gen_samples.py $DATA/impactnorm.vtu $DATA/impact_i50ps --batches 200 --surface
#   python tools/precompute_tet.py $DATA/impactnorm.vtu $DATA/impactprobe
# impact trains for 16,000 iterations: at 5,000 it is 6-8 dB lower.
cd "$(dirname "$0")"
G=${1:-0}; MODE=${2:-all}; shift $(( $# < 2 ? $# : 2 ))
DS=${*:-"mito sf1 valley rbl eq impact vertebra miranda chameleon richtmyer"}
DATA=${DATA:-data} OUT=${OUT:-output} LOGS=${LOGS:-bench_logs}
export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$PWD/gvraster:$PYTHONPATH
declare -A SRC=([mito]=mito.vtu [sf1]=sf1.vtu [valley]=valley.vtu [rbl]=rbl.vtu
                [eq]=earthquakenorm.vtu [impact]=impactnorm.vtu [vertebra]=vertebra.vtk
                [miranda]=miran.vtk [chameleon]=chame.vtk [richtmyer]=richt.vtk)
declare -A RATIOS=([vertebra]="1024 256 64 16 4096 16384" [valley]="1024 256 64 16 4096 16384")
declare -A SETS=([eq_interior]=earthquake_i50 [eq_surface]=earthquake_i50s
                 [impact_interior]=impact_i50p [impact_surface]=impact_i50ps)
declare -A EVAL_SET=([impact]=impactprobe)
declare -A ITERS=([impact]=16000)

need() {  # file: stop with the list above if an input is missing
  [ -e "$1" ] && return
  echo "run_final.sh: missing $1 (set DATA to the directory holding the inputs; see the header)" >&2
  exit 1
}

run() {  # kind dataset ratio
  local t=$1_$2_$3 src=$DATA/${SRC[$2]} flags="--kind $1 --ratio $3"
  need $src
  [ $1 = structured ] && flags="$flags --is_scaled"
  if [ -n "${SETS[$2_$1]}" ]; then
    need $DATA/${SETS[$2_$1]}_samples.npy
    flags="$flags --precomputed_samples $DATA/${SETS[$2_$1]}"
  fi
  [ -n "${ITERS[$2]}" ] && flags="$flags --iterations ${ITERS[$2]}"
  # Grids report the continuous PSNR, the structured metric of record.
  local ev=""; [ $1 = structured ] && ev="--is_scaled --continuous 2"
  if [ -n "${EVAL_SET[$2]}" ]; then
    need $DATA/${EVAL_SET[$2]}_samples.npy
    ev="--eval_set $DATA/${EVAL_SET[$2]}"
  fi
  local train="python train.py -s $src -m $OUT/$t $flags"
  local eval="python eval.py -s $src --models $OUT/$t $ev --out $LOGS/eval_$t.json"
  if [ -n "$PRINT" ]; then printf '%s\t%s\t%s\n' "$t" "$train" "$eval"; return; fi
  $train > $LOGS/$t.log 2>&1
  echo "EXIT=$?" >> $LOGS/$t.log
  $eval > $LOGS/eval_$t.log 2>&1
}

[ -n "$PRINT" ] || mkdir -p $LOGS
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
