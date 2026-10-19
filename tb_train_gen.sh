#!/bin/bash
# usage: tb_train_gen.sh <build> <gpu> <tag> <source.vtu> <fraction> <cap_max> [extra train flags...]
source /home/ldyken53/fnexp/env.sh >/dev/null 2>&1
cd /home/ldyken53/gaussian-volume
B=$1; G=$2; T=$3; SRC=$4; FR=$5; CAP=$6; shift 6
export CUDA_VISIBLE_DEVICES=$G PYTHONPATH=/home/ldyken53/gaussian-volume/rast_$B:$PYTHONPATH
python train.py -s $SRC --fraction $FR --cap_max $CAP --iterations 4000 "$@" --log_to_file -m ./output/$T > bench_logs/$T.log 2>&1
echo "EXIT=$?" >> bench_logs/$T.log
MS=$(cat output/$T/max_scale 2>/dev/null); python eval_final.py -s $SRC --models output/$T${MS:+:$MS} --out bench_logs/eval_$T.json > bench_logs/eval_$T.log 2>&1
