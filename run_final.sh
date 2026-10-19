#!/bin/bash
# Final models: interior and surface, three compression ratios per dataset. One GPU, serial.
# Build once: (cd diff-gaussian-rasterization && python setup.py build_ext --inplace)
# usage: run_final.sh <gpu> [interior|surface|all] [dataset ...]   (mito sf1 valley rbl eq)
# Outputs output/IN_<d>_<r> and output/SU_<d>_s<r>, evaluations in bench_logs/eval_<tag>.json.
cd "$(dirname "$0")"
G=${1:-0}; MODE=${2:-all}; shift $(( $# < 2 ? $# : 2 )); DS=${*:-"mito sf1 valley rbl eq"}
export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$PWD/diff-gaussian-rasterization:$PYTHONPATH
A=/lus/eagle/projects/dist_relational_alg/ldyken53
U=/lus/eagle/projects/UIC-HPC/ldyken53
declare -A SRC=([mito]=$A/mito.vtu [sf1]=$A/sf1.vtu [valley]=$U/valley.vtu [rbl]=$U/rbl.vtu
                [eq]=$A/earthquakenorm.vtu)
# cap_max per ratio 1024x / 256x / 64x (training ends at cap_max + 1000 Gaussians).
declare -A CAP=([mito]="5015 20058 80232" [sf1]="12663 50653 202611" [valley]="23925 95700 382800"
                [rbl]="3539 14157 56628" [eq]="87659 350636 1402545")
mkdir -p bench_logs

run() {  # tag src cap [train flags...]
  local t=$1 src=$2 cap=$3; shift 3
  python train.py -s $src --cap_max $cap -m output/$t "$@" > bench_logs/$t.log 2>&1
  echo "EXIT=$?" >> bench_logs/$t.log
  local surf=""; [[ " $* " == *" --encode_surface "* ]] && surf="--surface output/$t"
  python eval_final.py -s $src --models output/$t $surf --out bench_logs/eval_$t.json \
    > bench_logs/eval_$t.log 2>&1
}

for d in $DS; do
  caps=(${CAP[$d]}); ei=""; es=""
  # Earthquake trains on precomputed sample sets.
  [ $d = eq ] && ei="--precomputed_samples $U/earthquakebig" && es="--precomputed_samples $A/earthquake2big"
  for k in 0 1 2; do
    r=$((1024 >> (2 * k)))
    [ $MODE != surface ] && run IN_${d}_$r ${SRC[$d]} ${caps[$k]} $ei
    [ $MODE != interior ] && run SU_${d}_s$r ${SRC[$d]} ${caps[$k]} --encode_surface $es
  done
done
echo "FINAL RUNS DONE"
