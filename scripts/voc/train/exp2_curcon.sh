#!/usr/bin/env bash
set -e

CONFIG="configs/sscod/voc/exp2_curcon.py"
WORK_DIR="./checkpoints/sscod/voc/exp2_curcon"

GPUS=1
SEED=0

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
    tools/train.py $CONFIG \
    --launcher pytorch --seed $SEED --work_dir $WORK_DIR
