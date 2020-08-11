#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1
scripts/coco/train/exp2_caseA_baseline.sh
scripts/coco/train/exp2_caseB_baseline.sh
