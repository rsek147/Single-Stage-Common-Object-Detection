import torch

from ..registry import HEADS
from .atss_head import ATSSHead
from mmdet.core import delta2bbox, force_fp32, multiclass_nms_with_logits


@ HEADS.register_module
class ATSS_COD_Basline_Head(ATSSHead):

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_logits = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, anchors in zip(
                cls_scores, bbox_preds, centernesses, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            logits = cls_score.permute(
                1, 2, 0).reshape(-1, self.cls_out_channels)
            scores = logits.sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                logits = logits[topk_inds, :]
                centerness = centerness[topk_inds]

            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_logits.append(logits)
            mlvl_centerness.append(centerness)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_logits = torch.cat(mlvl_logits)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        det_bboxes, det_labels, det_logits = multiclass_nms_with_logits(
            mlvl_bboxes,
            mlvl_scores,
            mlvl_logits,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels, det_logits
