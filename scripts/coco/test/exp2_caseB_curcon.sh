#!/usr/bin/env bash
set -e

PROPOSAL=100
IOU=0.5
PAIRS=6
CONFIG="configs/sscod/coco/exp2_caseB_curcon.py"
WORK_DIR="/checkpoints/sscod/coco/exp2_caseB_curcon"

GPUS=4
CKPT="${WORK_DIR}/epoch_12.pth"
PKL="${WORK_DIR}/epoch_12.pkl"

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
    tools/test.py ${CONFIG} ${CKPT} \
    --launcher pytorch --out ${PKL} --options work_dir=$WORK_DIR

python tools/eval_sscod.py ${PKL} ${CONFIG} \
    --num_proposal_pairs=$PROPOSAL --iou_thr=$IOU --k_pairs=$PAIRS
