import torch
from mmdet.ops.nms import nms_wrapper


def multiclass_nms_with_logits(multi_bboxes,
                               multi_scores,
                               multi_logits,
                               score_thr,
                               nms_cfg,
                               max_num=-1,
                               score_factors=None):

    num_classes = multi_scores.shape[1]
    bboxes, labels, logits = [], [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)

    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue

        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
            _logits = multi_logits[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]

        if score_factors is not None:
            _scores *= score_factors[cls_inds]

        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, index = nms_op(cls_dets, **nms_cfg_)
        cls_logits = _logits[index]
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
        logits.append(cls_logits)

    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        logits = torch.cat(logits)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
            logits = logits[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        logits = multi_bboxes.new_zeros((0, multi_logits.shape[1]))

    return bboxes, labels, logits
