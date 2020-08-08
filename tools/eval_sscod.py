import os
import mmcv
import numpy as np
from argparse import ArgumentParser
from multiprocessing import cpu_count

from mmdet import datasets
from mmdet.core.evaluation import eval_cod_ap


def main():
    parser = ArgumentParser(description='COD Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--gt_pairs_file', help='file containing gt pairs')
    parser.add_argument(
        '--num_proposal_pairs',
        type=int,
        default=100,
        help='Number of proposal pairs')
    parser.add_argument(
        '--iou_thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    parser.add_argument(
        '--k_pairs',
        type=int,
        default=6,
        help='Number of images that an anchor images should pair')

    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # Print test params
    print("========================================")
    print("num_proposal_pairs:", args.num_proposal_pairs)
    print("iou_thr:", args.iou_thr)
    print("k_pairs:", args.k_pairs)
    print("========================================")

    # Load detect results
    det_results = mmcv.load(args.result)

    # Load gt annotations
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    annotations = [
        test_dataset.get_ann_info(i) for i in range(len(test_dataset))]

    # Eval
    eval_cod_ap(
        det_results, annotations,
        args.iou_thr, args.num_proposal_pairs, args.k_pairs,
        dataset='voc07' if 'VOC' in cfg.dataset_type else None,
        gt_pairs_file=args.gt_pairs_file,
        save_gt_pairs_to=os.path.join(cfg.data_root, 'Test_gt_pairs.pkl'),
        nproc=cpu_count())


if __name__ == '__main__':
    main()
