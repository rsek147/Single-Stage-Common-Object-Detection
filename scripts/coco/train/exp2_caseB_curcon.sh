#!/usr/bin/env bash
set -e

CONFIG="configs/sscod/coco/exp2_caseB_curcon.py"
WORK_DIR="/checkpoints/sscod/coco/exp2_caseB_curcon"

GPUS=8
SEED=0

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
    tools/train.py $CONFIG \
    --launcher pytorch --seed $SEED --options work_dir=$WORK_DIR
