import numpy as np
import os
import pickle
from tqdm import tqdm
from terminaltables import AsciiTable
from multiprocessing import Pool
from .bbox_overlaps import bbox_overlaps
from .mean_ap import average_precision
import random


def create_gt_pairs(annotations, cls_gts_ignore=None, area_ranges=None,
                    k_pairs=10, gt_pairs_file=None, max_tries=1000):
    num_imgs = len(annotations)

    def get_ignore_idx(cls_gts_ignore_, num_boxes, bboxes_,
                       num_areas=1, area_ranges=None):
        if len(cls_gts_ignore_) == 0 and area_ranges is None:
            return None

        if len(cls_gts_ignore_) > 0:
            ignore_idx = cls_gts_ignore_
        else:
            ignore_idx = np.zeros((num_areas, num_boxes)).astype(bool)
        if area_ranges is not None:
            for j, bbox in enumerate(bboxes_):
                gt_areas = (bbox[:, 2] - bbox[:, 0] + 1) * (
                    bbox[:, 3] - bbox[:, 1] + 1)
                for k, (min_area, max_area) in enumerate(area_ranges):
                    ignore_idx[k, :] = (gt_areas < min_area) | (
                        gt_areas > max_area)
        return ignore_idx

    if isinstance(gt_pairs_file, str) and os.path.isfile(gt_pairs_file):
        print("loading gt pairs")
        with open(gt_pairs_file, 'rb') as f:
            data = pickle.load(f)
        image_pairs_indexes = data['image_pairs_indexes']
        image_pairs_num_pos_pairs = data['image_pairs_num_pos_pairs']
        assert len(image_pairs_indexes) == num_imgs and \
            len(image_pairs_indexes[0]) == k_pairs+1, \
            "the gt_pairs_file is incorrect. " \
            "Please use another one or pass None to recompute"

        # random pickup one sample in the gt and double check.
        k = random.randint(0, num_imgs-1)
        pairs_indexes_k = image_pairs_indexes[k]
        num_pos_pairs_k = image_pairs_num_pos_pairs[k]
        assert pairs_indexes_k[0] == k and num_pos_pairs_k[0] == 0, \
            "the first element must be idx of the anchor sample"

    else:
        print("Create gt pairs")
        image_pairs_indexes = []
        image_pairs_num_pos_pairs = []
        num_areas = len(area_ranges) if area_ranges is not None else 1
        for idx1 in tqdm(range(num_imgs), total=num_imgs):
            ann_1 = annotations[idx1]['labels']
            n1 = len(ann_1)
            ignore_boxes_1 = get_ignore_idx(cls_gts_ignore[idx1],
                                            n1, annotations[idx1]['bboxes'],
                                            num_areas, area_ranges)
            # Choose randomly k_pairs
            kp = 0
            tries = 0
            idx_pairs = [idx1]
            num_pos_pairs = [0]
            while kp < k_pairs:
                idx2 = random.randint(0, num_imgs-1)
                ann_2 = annotations[idx2]['labels']
                # we requires a pair image has at least one common class
                if idx2 not in idx_pairs and \
                        len(set(ann_1).intersection(set(ann_2))) > 0:
                    kp += 1
                    idx_pairs.append(idx2)

                    n2 = len(ann_2)
                    labels_1 = np.repeat(ann_1[:, np.newaxis], n2, axis=1)
                    labels_2 = np.repeat(ann_2[np.newaxis, :], n1, axis=0)
                    pos_pairs = (labels_1 == labels_2)

                    # ignored_gts/gts beyond the specific scale are not counted
                    ignore_boxes_2 = get_ignore_idx(
                        cls_gts_ignore[idx2], n2, annotations[idx2]['bboxes'],
                        num_areas, area_ranges)
                    num_pos_pairs.append(pos_pairs.sum())

                else:
                    tries += 1
                    if max_tries != -1 and tries > max_tries:
                        break

            if len(num_pos_pairs) == 1:
                continue
            elif len(num_pos_pairs) != (k_pairs+1):
                num_pos_pairs += [num_pos_pairs[-1]] * \
                    (k_pairs + 1 - len(num_pos_pairs))

            image_pairs_indexes.append(idx_pairs)
            image_pairs_num_pos_pairs.append(num_pos_pairs)
    return image_pairs_indexes, image_pairs_num_pos_pairs


def tpfp_iou(det_bboxes,
             gt_bboxes,
             gt_labels,
             gt_bboxes_ignore=None,
             iou_thr=0.5,
             area_ranges=None):
    """Check if detected bboxes are true positive or false positive, and get
    the corresponding labels of GT box. Copied from mean_ap.tpfp_default, just
    add gt_labels for input and return additional output.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_labels (ndarray): Label of GT boxes of this image, of shape (n,1).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
                of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
                Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
                in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
                each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)

    if num_dets == 0:
        # If no detection was found
        fake_num_det = max(1, num_gts)
        tp = np.zeros((num_scales, fake_num_det), dtype=np.int8)
        fp = np.ones((num_scales, fake_num_det), dtype=np.int8)
        tp_gt_labels = np.zeros((num_scales, fake_num_det), dtype=np.int8)
        return tp, fp, tp_gt_labels

    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.int8)
    fp = np.zeros((num_scales, num_dets), dtype=np.int8)
    tp_gt_labels = np.zeros((num_scales, num_dets), dtype=np.int8)
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if num_gts == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + 1) * (
                det_bboxes[:, 3] - det_bboxes[:, 1] + 1)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp, tp_gt_labels

    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)

    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)

        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * \
                (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)

        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt] or
                        gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                        tp_gt_labels[k, i] = gt_labels[matched_gt]
                    else:
                        fp[k, i] = 1

                # otherwise ignore this detected bbox, tp = 0, fp = 0
                elif min_area is None:
                    fp[k, i] = 1
                else:
                    bbox = det_bboxes[i, :4]
                    area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
                    if area >= min_area and area < max_area:
                        fp[k, i] = 1

    return tp, fp, tp_gt_labels


def tpfp_cod(det_results_1, det_results_2,
             tp_boxes_1, tp_boxes_2,
             fp_boxes_1, fp_boxes_2,
             gt_labels_1, gt_labels_2, num_proposal_pairs):

    # A pair of predicted boxes is considered postive if:
    # 1. Their embed vector has highest cosine similarity
    if len(gt_labels_1) == 0 or len(gt_labels_2) == 0:
        # Ignore this prediction pairs
        return (np.zeros(1, dtype=np.int8), np.zeros(1, dtype=np.int8),
                np.zeros(1, dtype=np.int8))

    embeddings1 = det_results_1['embed']
    scores1 = det_results_1['boxes'][:, -1]
    embeddings2 = det_results_2['embed']
    scores2 = det_results_2['boxes'][:, -1]
    cosine = np.matmul(embeddings1, embeddings2.transpose())
    cosine[cosine < 0] = 0
    cosine = scores1[:, None] * np.sqrt(cosine) * scores2[None, :]

    # Find the top_proposal_pairs that have highest cosine score
    cosine = cosine.reshape(-1)
    num_proposals = min(len(cosine), num_proposal_pairs)
    top_proposal_idxes = np.argpartition(
        cosine, -num_proposals)[-num_proposals:]
    top_proposal_idxes = top_proposal_idxes[np.argsort(
        cosine[top_proposal_idxes])[::-1]]

    # 2. Both boxes have iou with gt > iou_thr
    pred_pair_tp_iou = np.matmul(
        tp_boxes_1.transpose(), tp_boxes_2).astype(bool)

    # 3. We also need the labels of the nearest GT box(of each predicted box)
    # for evaluation
    n1, n2 = gt_labels_1.shape[1], gt_labels_2.shape[1]
    labels_1 = np.repeat(gt_labels_1.transpose(), n2, axis=1)
    labels_2 = np.repeat(gt_labels_2, n1, axis=0)
    pred_pairs_same_class = np.logical_and.reduce(
        [labels_1 == labels_2, labels_1 > 0, labels_2 > 0])

    # extract the properties of the top_proposal_pairs
    cosine = cosine[top_proposal_idxes]
    try:
        pred_pair_tp_iou = pred_pair_tp_iou.reshape(-1)[top_proposal_idxes]
    except:
        import pdb
        pdb.set_trace()
    pred_pairs_same_class = pred_pairs_same_class.reshape(-1)[
        top_proposal_idxes]

    # TP pairs are the one satisying both iou condition and matching
    # class condition
    tp = np.logical_and(pred_pair_tp_iou, pred_pairs_same_class)

    # FP pairs if either one or two above conditions are missed
    pred_pair_fp_iou = np.repeat(
        fp_boxes_1.astype(bool).transpose(), n2, axis=1) | \
        np.repeat(fp_boxes_2.astype(bool), n1, axis=0)
    pred_pair_fp_iou = pred_pair_fp_iou.reshape(-1)[top_proposal_idxes]
    fp = np.logical_or(pred_pair_fp_iou, ~pred_pairs_same_class)

    return tp.astype(np.int8), fp.astype(np.int8), cosine


def tpfp_cod_batch(det_results,
                   tp_boxes, fp_boxes,
                   gt_label_of_boxes,
                   idx_pairs,
                   num_proposal_pairs=100):
    idx1 = idx_pairs[0]
    det_results_1 = det_results[idx1]
    tp_boxes_1 = tp_boxes[idx1]
    fp_boxes_1 = fp_boxes[idx1]
    gt_labels_1 = gt_label_of_boxes[idx1]

    tp, fp, score = [], [], []
    # print(idx_pairs)
    # import pdb; pdb.set_trace()
    for idx2 in idx_pairs[1:]:
        det_results_2 = det_results[idx2]
        tp_boxes_2 = tp_boxes[idx2]
        fp_boxes_2 = fp_boxes[idx2]
        gt_labels_2 = gt_label_of_boxes[idx2]

        tp_i, fp_i, score_i = tpfp_cod(
            det_results_1, det_results_2,
            tp_boxes_1, tp_boxes_2,
            fp_boxes_1, fp_boxes_2,
            gt_labels_1, gt_labels_2, num_proposal_pairs)
        tp.append(tp_i)
        fp.append(fp_i)
        score.append(score_i)
    return np.hstack(tp), np.hstack(fp), np.hstack(score)


def eval_cod_ap(det_results,
                annotations,
                iou_thr=0.5,
                num_proposal_pairs=100,
                k_pairs=4,
                dataset=None,
                gt_pairs_file=None,
                save_gt_pairs_to=None,
                nproc=4,
                scale_ranges=None,
                seed=0):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:
                - "bboxes": numpy array of shape (n, 4)
                - "labels": numpy array of shape (n, )
                - "bboxes_ignore" (optional): numpy array of shape (k, 4)
                - "labels_ignore" (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """

    random.seed(seed)
    num_imgs = len(annotations)
    assert len(det_results) == num_imgs and num_imgs >= k_pairs
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = Pool(nproc)
    eval_results = []

    # Get TP and FP box prediction if their IoU with GT >thr
    cls_dets = [det_result['boxes'] for det_result in det_results]
    cls_gts = [ann['bboxes'] for ann in annotations]
    cls_labels = [ann['labels'] for ann in annotations]
    cls_gts_ignore = []

    for ann in annotations:
        if len(ann.get('labels_ignore', [])) > 0:
            ignore_inds = ann['labels_ignore']
            cls_gts_ignore.append(ann['bboxes_ignore'])
        else:
            cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))

    # Find the pairs index for each image
    image_pairs_indexes, image_pairs_num_pos_pairs = create_gt_pairs(
        annotations, cls_gts_ignore, area_ranges, k_pairs,  gt_pairs_file)

    if save_gt_pairs_to is not None and save_gt_pairs_to != gt_pairs_file:
        with open(save_gt_pairs_to, 'wb') as f:
            pickle.dump(dict(
                image_pairs_indexes=image_pairs_indexes,
                image_pairs_num_pos_pairs=image_pairs_num_pos_pairs), f)

    print("Finish generating gt image pairs") if gt_pairs_file is None \
        else print("Finish loading gt image pairs")

    # Evaluate if a predicted box is TP or FP based on iou with GT.
    tpfp_box_and_labels = pool.starmap(
        tpfp_iou,
        zip(cls_dets, cls_gts, cls_labels, cls_gts_ignore,
            [iou_thr for _ in range(num_imgs)],
            [area_ranges for _ in range(num_imgs)]))
    tp_boxes, fp_boxes, gt_label_of_boxes = tuple(zip(*tpfp_box_and_labels))
    print("Finish evaluating postive boxes using IOU")

    # Evaluate if a predicted pairs are TP or FP. Single Thread mode for DEBUG:
    tp, fp, scores = [], [], []
    for idx_pairs in image_pairs_indexes:
        tp_i, fp_i, score_i = tpfp_cod_batch(
            det_results, tp_boxes, fp_boxes, gt_label_of_boxes,
            idx_pairs, num_proposal_pairs)
        tp.append(tp_i)
        fp.append(fp_i)
        scores.append(score_i)

    print("Finish evaluating postive cod pairs")

    # sort all det bboxes by score, also sort tp and fp
    scores = np.hstack(scores)
    num_dets = scores.shape[0]
    sort_inds = np.argsort(-scores)

    tp = np.hstack(tp)[sort_inds].astype(np.float32)
    fp = np.hstack(fp)[sort_inds].astype(np.float32)
    num_gts = np.array(image_pairs_num_pos_pairs, dtype=np.float32).sum()

    # calculate recall and precision with tp and fp
    tp = np.cumsum(tp, axis=0)
    fp = np.cumsum(fp, axis=0)
    eps = np.finfo(np.float32).eps
    recalls = tp / np.maximum(num_gts, eps)
    precisions = tp / np.maximum((tp + fp), eps)

    # calculate AP
    mode = 'area' if dataset != 'voc07' else '11points'
    ap = average_precision(recalls, precisions, mode)
    table_data = [['num_gts', 'num_dets', 'recall', 'avg. prec(ap)'],
                  [int(num_gts), num_dets, recalls[-1], ap]
                  ]
    table = AsciiTable(table_data)
    print(table.table)
    return image_pairs_indexes, image_pairs_num_pos_pairs
