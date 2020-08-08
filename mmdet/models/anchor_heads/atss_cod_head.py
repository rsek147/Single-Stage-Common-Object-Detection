import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from mmcv.cnn import normal_init

from ..registry import HEADS
from ..builder import build_loss
from .atss_head import ATSSHead, reduce_mean
from ..utils import ConvModule, Scale, bias_init_with_prob
from mmdet.core import (delta2bbox, force_fp32, multi_apply,
                        multiclass_nms_with_logits)


@HEADS.register_module
class ATSS_COD_Head(ATSSHead):

    def __init__(self,
                 stacked_obj_convs=4,
                 embed_channels=256,
                 embed_norm_cfg=None,
                 classwise_loss=dict(
                     type='ArcFaceLoss', scale=30.0,
                     margin=0.5, easy_margin=True, focal_gamma=2,
                     ignore_class0=True, loss_weight=1.0),
                 pairwise_loss=None,
                 unseen_classID=None,
                 exp_type=1,
                 **kwargs):
        # Classwise loss is the family that use softmax to separate each class.
        # Eg: ArcFace, CurricularFace, SphereLoss. Pairwise loss is the family
        # that compute distance between each pair points. Eg: Triplet,
        # Contrastive, Intra, InterLoss, CosineSimilarity
        super().__init__(**kwargs)

        self.stacked_obj_convs = stacked_obj_convs
        self.embed_channels = embed_channels
        self.embed_norm_cfg = embed_norm_cfg  # To test if using BN or not
        if unseen_classID is not None:
            assert exp_type > 1
        if exp_type > 1:
            assert unseen_classID is not None

        self.unseen_classID = unseen_classID
        self.exp_type = exp_type

        self.codet_init_layers()
        self.ignore_class0 = True
        if classwise_loss:
            classwise_loss.update(
                dict(embed_channels=embed_channels,
                     num_classes=self.num_classes))
            self.loss_classwise = build_loss(classwise_loss)
            self.ignore_class0 = self.ignore_class0 * \
                classwise_loss['ignore_class0']
        if pairwise_loss:
            self.loss_pairwise = build_loss(pairwise_loss)
            self.ignore_class0 = self.ignore_class0 * \
                pairwise_loss['ignore_class0']

        assert self.with_classwise_loss or self.with_pairwise_loss
        self.codet_init_weight()

    @property
    def with_obj_branch(self):
        # We create a separate branch to detect objectness if
        # number stack_obj_convs >0
        return self.stacked_obj_convs > 0

    @property
    def with_classwise_loss(self):
        return hasattr(self, 'loss_classwise') and \
            self.loss_classwise is not None

    @property
    def with_pairwise_loss(self):
        return hasattr(self, 'loss_pairwise') and \
            self.loss_pairwise is not None

    def codet_init_layers(self):
        # cls_head is modified to be class-agnostic, e.g,
        # classify if an anchor is foreground/background
        self.cls_out_channels = 1
        self.atss_cls = nn.Conv2d(
            self.feat_channels, self.num_anchors*self.cls_out_channels,
            3, padding=1)

        # We create separate objectness branch
        self.obj_convs = nn.ModuleList()
        for i in range(self.stacked_obj_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.obj_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        # Embed head
        # Layer to extract embed_feat don't have bias and activation.
        self.pre_embed_head = nn.Conv2d(
            self.feat_channels, self.embed_channels, 3, padding=1, bias=False)

    def codet_init_weight(self):
        if self.with_obj_branch:
            for m in self.obj_convs:
                normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.atss_cls, std=0.01, bias=bias_cls)
        normal_init(self.pre_embed_head, std=0.01)

    def forward_single(self, x, scale):
        # Similar to the regular Single Stage head, except we have an extra
        # branch to compute embedding feature.
        embed_feat = x
        reg_feat = x
        obj_feat = x

        # Classification (Objectness) branch
        for obj_conv in self.obj_convs:
            obj_feat = obj_conv(obj_feat)
        cls_score = self.atss_cls(obj_feat)

        # Bounding box and centerness
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)

        # Embedded branch
        for cls_conv in self.cls_convs:
            embed_feat = cls_conv(embed_feat)
        embed_feat = self.pre_embed_head(embed_feat)
        embed_feat = F.normalize(embed_feat)
        return cls_score, bbox_pred, centerness, embed_feat

    def loss_single(self, anchors, cls_score, bbox_pred, centerness,
                    embed_feat, labels, label_weights, bbox_targets,
                    num_total_samples, cfg):

        # This is identical to the regular loss of Single stage.
        # Specifically, we compute:
        # + Classification loss to objectness (we replace agnostic labels)
        # + Regression loss to bounding box and centerness
        # In addition, we return the reshape embeded features and their labels

        # Agnostic_Class loss
        agnostic_labels = torch.clamp(labels, min=0, max=1)
        if self.exp_type == 3:
            # set label_weights=0 for the unseen classes
            for unseen_ID in self.unseen_classID:
                ignore_samples = (labels != unseen_ID).float()
                label_weights *= ignore_samples

        loss_cls, loss_bbox, loss_centerness, centerness_targets = \
            super().loss_single(
                anchors, cls_score, bbox_pred, centerness, agnostic_labels,
                label_weights, bbox_targets, num_total_samples, cfg)

        # Return embed feature and its label
        embed_feat = embed_feat.permute(
            0, 2, 3, 1).reshape(-1, self.embed_channels)
        labels = labels.reshape(-1)

        # Use objectness as weight for embedded loss.
        with torch.no_grad():
            weight = cls_score.sigmoid()*centerness.sigmoid()
            weight = weight.permute(0, 2, 3, 1).reshape(-1) + 1

        if self.ignore_class0:
            # In Obj Detection, class0 is background.
            pos_inds = torch.nonzero(labels).squeeze(1)
            if len(pos_inds) > 0:
                embed_feat = embed_feat[pos_inds, :]
                labels = labels[pos_inds]
                weight = weight[pos_inds]
            else:
                embed_feat = None
                labels = None
                weight = None
        return (loss_cls, loss_bbox, loss_centerness, centerness_targets.sum(),
                embed_feat, labels, weight)

    @force_fp32(apply_to=(
        'cls_scores', 'bbox_preds', 'centernesses', 'embed_feat'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             embed_feat,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.atss_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)

        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos).cuda()).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, loss_centerness, bbox_avg_factor, \
            embed_feat, labels, weights = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                centernesses,
                embed_feat,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                num_total_samples=num_total_samples,
                cfg=cfg)

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))

        # Detection outputs
        loss_outputs = dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_centerness=loss_centerness)

        return loss_outputs, (embed_feat, labels, weights)

    @force_fp32(apply_to=('embed_feat'))
    def loss_embed(self, embed_feat, labels, weights=None):
        # Cat features from multiple levels:
        embed_feat = torch.cat(
            [item for item in embed_feat if item is not None], dim=0)
        labels = torch.cat([item for item in labels if item is not None])
        weights = torch.cat([item for item in weights if item is not None])
        # Remove unseen classes from loss
        if self.unseen_classID is not None:
            indicators = torch.ones_like(labels).bool()
            for unseen_ID in self.unseen_classID:
                indicators = indicators & (labels != unseen_ID)
            if torch.any(indicators):
                embed_feat = embed_feat[indicators]
                labels = labels[indicators]
                weights = weights[indicators]
            else:
                # return dict()
                outputs = dict()
                # Classwise loss
                if self.with_classwise_loss:
                    outputs['loss_angular'] = embed_feat.sum() * 0
                    outputs['cw_cos_p'] = embed_feat.sum() * 0
                    outputs['cw_cos_n'] = embed_feat.sum() * 0
                    outputs['cw_class_acc'] = embed_feat.sum() * 0
                    outputs['cw_dist_p'] = embed_feat.sum() * 0
                    outputs['cw_dist_n'] = embed_feat.sum() * 0
                    outputs['cw_acc'] = embed_feat.sum() * 0
                    outputs['cw_prec'] = embed_feat.sum() * 0
                # pairwise Loss
                if self.with_pairwise_loss:
                    outputs['loss_contrastive'] = embed_feat.sum() * 0
                    outputs['pw_dist_p'] = embed_feat.sum() * 0
                    outputs['pw_dist_n'] = embed_feat.sum() * 0
                    outputs['pw_acc'] = embed_feat.sum() * 0
                    outputs['pw_prec'] = embed_feat.sum() * 0
                return outputs
        outputs = dict()
        avg_factor = (labels > 0).sum()
        # Classwise loss
        if self.with_classwise_loss:
            loss_classwise = self.loss_classwise(
                embed_feat, labels, weights, avg_factor=avg_factor)
            outputs.update(loss_classwise)
        # pairwise Loss
        if self.with_pairwise_loss:
            loss_pairwise = self.loss_pairwise(
                embed_feat, labels, weights, avg_factor=avg_factor)
            outputs.update(loss_pairwise)
        return outputs

    @force_fp32(apply_to=(
        'cls_scores', 'bbox_preds', 'centernesses', 'embed_feat'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   embed_feat,
                   img_metas,
                   cfg,
                   rescale=False):

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device
        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(
                cls_scores[i].size()[-2:],
                self.anchor_strides[i],
                device=device) for i in range(num_levels)
        ]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            embed_feat_list = [
                embed_feat[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               centerness_pred_list,
                                               embed_feat_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          embed_feats,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(
            mlvl_anchors) == len(embed_feats)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_embed = []
        for cls_score, bbox_pred, centerness, anchors, embed_feat in zip(
                cls_scores, bbox_preds, centernesses,
                mlvl_anchors, embed_feats):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            embed_feat = embed_feat.permute(1, 2, 0).reshape(
                -1, self.embed_channels)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
                embed_feat = embed_feat[topk_inds, :]

            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_embed.append(embed_feat)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        mlvl_embed = torch.cat(mlvl_embed)

        det_bboxes, det_labels, det_embeds = multiclass_nms_with_logits(
            mlvl_bboxes,
            mlvl_scores,
            mlvl_embed,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)

        if cfg.get('infer_cls_label', True) and self.with_classwise_loss:
            if len(det_bboxes):
                det_labels = self.loss_classwise.infer_classes(det_embeds)

        return det_bboxes, det_labels, det_embeds

    def get_pair_score(self, embed_feat1, embed_feat2,
                       proposal_score1=None, proposal_score2=None):
        """
        embed_feat is a Tensor of shape [N,C], where:
            - N is the number of proposal
            - C is the number of feature to compute cosine distance.
        proposal_score is a tensor of shape [N,1], represent the conf of objness
        """
        scores = torch.mm(embed_feat1, embed_feat2.t())

        if proposal_score1 is not None and proposal_score2 is not None:
            scores = torch.mm(
                proposal_score1, proposal_score2.t())*torch.sqrt(scores)

        return scores

    def get_topk_pairs(self, scores, boxes1, boxes2,
                       labels1, labels2, codet_cfg):
        """
        boxes1/2 is the boxes from Img1/2, dim=(N1,5)/(N2/5), where the last
        element is objectness's score.
        scores is the matching score of these boxes, dim=(N1,N2)
        """
        # Select the top-k, and greater than a threshold
        thr_mat = scores > codet_cfg['matching_thr']
        scores_select = scores[torch.nonzero(thr_mat, as_tuple=True)]
        pair_idx = torch.nonzero(thr_mat)
        N_pairs = len(pair_idx)
        if N_pairs > 0:
            topk_val, topk_idx = torch.topk(
                scores_select, min(N_pairs, codet_cfg['max_pairs']))
            pair_idx = pair_idx[topk_idx]

            # Extract pair_scores
            pair_idx = pair_idx.cpu().tolist()
            topk_val = topk_val.cpu().tolist()
            matching_pair = [pair+[val]
                             for pair, val in zip(pair_idx, topk_val)]
        else:
            matching_pair = []
        # Extract box_idx
        boxes1 = boxes1.cpu().numpy()
        boxes2 = boxes2.cpu().numpy()
        labels1 = labels1.cpu().numpy()
        labels2 = labels2.cpu().numpy()

        results = dict(
            boxes=({id: boxes1[[id]] for id in range(len(boxes1))},
                   {id: boxes2[[id]] for id in range(len(boxes2))}),
            labels=({id: labels1[id] for id in range(len(labels1))},
                    {id: labels2[id] for id in range(len(labels2))}),
            pairs=matching_pair)

        return results
