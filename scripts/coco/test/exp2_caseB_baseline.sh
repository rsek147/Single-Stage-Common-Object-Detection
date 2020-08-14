#!/usr/bin/env bash
set -e

PROPOSAL=100
IOU=0.5
PAIRS=6
CONFIG="configs/sscod/coco/exp2_caseB_baseline.py"
WORK_DIR="/checkpoints/sscod/coco/exp2_caseB_baseline"

GPUS=4
CKPT="${WORK_DIR}/epoch_12.pth"
PKL="${WORK_DIR}/epoch_12.pkl"

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
    tools/test.py ${CONFIG} ${CKPT} \
    --launcher pytorch --out ${PKL} --options work_dir=$WORK_DIR

echo "Evaluating Soft-EONC..."
export EONC='1'
python tools/eval_sscod_baseline.py ${PKL} ${CONFIG} \
    --num_proposal_pairs=$PROPOSAL --iou_thr=$IOU --k_pairs=$PAIRS --feat "soft"

echo "Evaluating Soft-ONC..."
export EONC='0'
python tools/eval_sscod_baseline.py ${PKL} ${CONFIG} \
    --num_proposal_pairs=$PROPOSAL --iou_thr=$IOU --k_pairs=$PAIRS --feat "soft"

echo "Evaluating Hard-EONC..."
export EONC='1'
python tools/eval_sscod_baseline.py ${PKL} ${CONFIG} \
    --num_proposal_pairs=$PROPOSAL --iou_thr=$IOU --k_pairs=$PAIRS --feat "hard"

echo "Evaluating Hard-ONC..."
export EONC='0'
python tools/eval_sscod_baseline.py ${PKL} ${CONFIG} \
    --num_proposal_pairs=$PROPOSAL --iou_thr=$IOU --k_pairs=$PAIRS --feat "hard"
