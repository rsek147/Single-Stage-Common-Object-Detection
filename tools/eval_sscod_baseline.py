from argparse import ArgumentParser

import os
import mmcv
import numpy as np

from mmdet import datasets
from mmdet.core.evaluation import eval_cod_ap


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    parser = ArgumentParser(description='Codet Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--gt_pairs_file', help='file containing gt pairs')
    parser.add_argument(
        '--num_proposal_pairs',
        type=int,
        default=10,
        help='Number of proposal pairs')
    parser.add_argument(
        '--iou_thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    parser.add_argument(
        '--k_pairs',
        type=int,
        default=2,
        help='Number of images that an anchor images should pair')
    parser.add_argument(
        '--feat',
        type=str,
        default='hard',
        help='Type of features: soft/hard. Default is hard')

    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # Print test params
    assert args.feat in ['logit', 'soft', 'hard']
    print("\n========================================")
    print("num_proposal_pairs:", args.num_proposal_pairs)
    print("iou_thr:", args.iou_thr)
    print("k_pairs:", args.k_pairs)
    print("feat:", args.feat)
    print("========================================")

    # Load gt annotations
    test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    annotations = [test_dataset.get_ann_info(
        i) for i in range(len(test_dataset))]

    # Load detect results
    results = mmcv.load(args.result)

    det_results = []
    for result in results:
        # Boxes
        bboxes = result['bboxes']
        if len(bboxes) == 0:
            det_results.append(dict(
                boxes=np.zeros([0, 5]),
                embed=np.zeros([0, len(cfg.total_classes)]),
            ))
            continue

        # Soft features
        logits = result['logits']
        soft_embeddings = _sigmoid(logits)

        # Hard features
        labels = result['classes']
        hard_embeddings = np.zeros(
            [len(labels), len(cfg.total_classes)], dtype=int)
        hard_embeddings[np.arange(len(labels)), labels] = 1

        # Construct a dict
        if args.feat == 'soft':
            embed = soft_embeddings
        elif args.feat == 'hard':
            embed = hard_embeddings

        det_results.append(dict(
            boxes=bboxes,
            embed=embed,
        ))

    # Eval
    eval_cod_ap(
        det_results, annotations,
        args.iou_thr, args.num_proposal_pairs, args.k_pairs,
        dataset='voc07' if 'VOC' in cfg.dataset_type else None,
        gt_pairs_file=args.gt_pairs_file,
        save_gt_pairs_to=os.path.join(cfg.data_root, 'Test_gt_pairs.pkl'))


if __name__ == '__main__':
    main()
