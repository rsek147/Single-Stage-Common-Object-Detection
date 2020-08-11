#!/usr/bin/env bash
set -e

PROPOSAL=100
IOU=0.5
PAIRS=6
CONFIG="configs/sscod/voc/exp2_arccon.py"
WORK_DIR="./checkpoints/sscod/voc/exp2_arccon"

GPUS=6
CKPT="${WORK_DIR}/epoch_12.pth"
PKL="${WORK_DIR}/epoch_12.pkl"

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
    tools/test.py ${CONFIG} ${CKPT} \
    --launcher pytorch --out ${PKL} --options work_dir=$WORK_DIR

export EONC='1'
echo "Evaluating EONC..."
python tools/eval_sscod.py ${PKL} ${CONFIG} \
    --num_proposal_pairs=$PROPOSAL --iou_thr=$IOU --k_pairs=$PAIRS

export EONC='0'
echo "Evaluating ONC..."
python tools/eval_sscod.py ${PKL} ${CONFIG} \
    --num_proposal_pairs=$PROPOSAL --iou_thr=$IOU --k_pairs=$PAIRS
