CONFIG="configs/sscod/voc/exp2_arccon.py"
WORK_DIR="/checkpoints/sscod/voc/exp2_arccon"
CKPT="${WORK_DIR}/epoch_12.pth"
export CUDA_VISIBLE_DEVICES=1
SEED=0
PYTHON=${PYTHON:-"python"}
DATA_ROOT="/data/VOCdevkit/"
RESULT_FILE=$CHECKPOINT_FILE".pkl"
RESULT_TXT=${CHECKPOINT_FILE}".txt"
echo "CHECKPOINT="$CHECKPOINT_FILE


python tools/infer_codet_images.py --config ${CONFIG} --checkpoint ${CKPT} \
                                            --img_folder codet-debug-imgs

# python tools/test.py ${CONFIG_FILE}  ${CHECKPOINT_FILE} --show --show-dir './cache'