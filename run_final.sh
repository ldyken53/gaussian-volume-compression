#!/bin/bash
# Final unified configs (2026-10-18), one GPU, serial. Builds: rast_fz7 (SOFT_CUTOFF 1) for interior,
# rast_fo3 (SOFT_CUTOFF 0) for surface; make them with ./build_cfg.sh <tag> <SOFT_CUTOFF>.
# usage: run_final.sh <gpu> [interior|surface|all] [dataset ...]   (datasets: mito sf1 valley rbl eq)
# Outputs: output/IN_<d>_<r>, output/SU_<d>_s<r>; evals in bench_logs/eval_<tag>.json.
cd /home/ldyken53/gaussian-volume
G=${1:-0}; MODE=${2:-all}; shift $(( $# < 2 ? $# : 2 )); DS=${*:-"mito sf1 valley rbl eq"}
M=/lus/eagle/projects/dist_relational_alg/ldyken53/mito.vtu
S=/lus/eagle/projects/dist_relational_alg/ldyken53/sf1.vtu
V=/lus/eagle/projects/UIC-HPC/ldyken53/valley.vtu
R=/lus/eagle/projects/UIC-HPC/ldyken53/rbl.vtu
E=/lus/eagle/projects/dist_relational_alg/ldyken53/earthquakenorm.vtu
EPI="--precomputed_samples --precomputed_prefix /lus/eagle/projects/UIC-HPC/ldyken53/earthquakebig"
EPS="--precomputed_samples --precomputed_prefix /lus/eagle/projects/dist_relational_alg/ldyken53/earthquake2big"
CACHE="--sample_cache_dir /lus/eagle/projects/UIC-HPC/ldyken53/sample_cache"
SHARED="--scaling_lr 0.008 --rotation_lr 0.005 --values_lr 0.005 --densify_alpha 2.5 --densify_growth 1.5
 --ema 0.995 --ema_from 1000 --max_scale_k 1.25 --init_weight 0.1 --fraction 1 --init_share_max 0.5
 --init_max 20000 --iterations 5000"
INT="$SHARED --loss l2 --fn_mode log --fn_thresh 0.002 --fn_reg 0.0001 --samples_per_node 0.25"
SURF="$CACHE $SHARED --max_scale_vol --loss l1 --fn_mode abs --fp_mode hinge --fp_margin 0.2 --fn_margin 0.05
 --fnfp_norm all --surf_mask_data --fn_reg 300 --fp_reg 3 --interior_skin 0.1 --encode_surface --sample_frac 0.5"

# cap_max (final Gaussian count) per ratio 1024/256/64. The initial count comes from --init_share_max/--init_max
# (--fraction 1 = rule over all nodes), so no per-cell fraction.
declare -A SRC=([mito]=$M [sf1]=$S [valley]=$V [rbl]=$R [eq]=$E)
declare -A CAP=([mito]="5015 20058 80232" [sf1]="12663 50653 202611" [valley]="23925 95700 382800"
                [rbl]="3539 14157 56628" [eq]="87659 350636 1402545")
for d in $DS; do
  caps=(${CAP[$d]}); ei=""; es=""
  [ $d = eq ] && ei=$EPI && es=$EPS
  for k in 0 1 2; do
    r=$((1024 >> (2 * k)))
    [ $MODE != surface ] && ./tb_train_gen.sh fz7 $G IN_${d}_$r ${SRC[$d]} 1 ${caps[$k]} $ei $INT
    [ $MODE != interior ] && env OCC_T=0.01 OCC_KEEP_AW=1 \
      ./tb_train_surf.sh fo3 $G SU_${d}_s$r ${SRC[$d]} 1 ${caps[$k]} $es $SURF
  done
done
echo "FINAL RUNS DONE"
