#!/usr/bin/env bash
set -e

CONFIG="configs/sscod/coco/exp2_caseB_baseline.py"
WORK_DIR="/checkpoints/sscod/coco/exp2_caseB_baseline"

GPUS=2
SEED=0

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
    tools/train.py $CONFIG \
    --launcher pytorch --seed $SEED --work_dir $WORK_DIR
