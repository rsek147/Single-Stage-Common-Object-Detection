import torch
from torch.nn import functional as F

from mmdet.core import bbox2result
from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class SSCOD(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SSCOD, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):

        if isinstance(img, list):
            # Use the forward_train of SingleStageDetector class
            losses_0, embed_loss_input_0 = super().forward_train(
                img[0], img_metas[0],
                gt_bboxes[0], gt_labels[0],
                None if gt_bboxes_ignore is None else gt_bboxes_ignore[0])
            losses_1, embed_loss_input_1 = super().forward_train(
                img[1], img_metas[1],
                gt_bboxes[1], gt_labels[1],
                None if gt_bboxes_ignore is None else gt_bboxes_ignore[1])

            # Merge loss from two sub-minibatch
            losses = dict()
            for k, v in losses_0.items():
                if isinstance(v, list):
                    losses[k] = [0.5*(v[i]+losses_1[k][i])
                                 for i in range(len(v))]
                else:
                    losses[k] = 0.5*(v+losses_1[k])

            # Merge embed_loss_input=(embed_feat,labels,pos_ind)
            # from two sub-minibatches
            embed_loss_input = [
                v0+v1
                for v0, v1 in zip(embed_loss_input_0, embed_loss_input_1)]
            embed_loss_input = tuple(embed_loss_input)
        else:
            # Use the forward_train of SingleStageDetector class
            losses, embed_loss_input = super().forward_train(
                img, img_metas,
                gt_bboxes, gt_labels, gt_bboxes_ignore)

        loss_embed = self.bbox_head.loss_embed(*embed_loss_input)
        losses.update(loss_embed)
        return losses

    def simple_test(self, img, img_metas, rescale=False, return_dict=True):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_metas, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        det_result = bbox_list[0]

        if return_dict:
            return {'boxes': det_result[0].cpu().numpy(),
                    'classes': det_result[1].cpu().numpy(),
                    'embed': det_result[2].cpu().numpy()}
        else:
            boxes1, labels1, embed_feat1 =  det_result[0],det_result[1],det_result[2]
            return boxes1, labels1, embed_feat1
            # bbox_results = [
            #     bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            #     for det_bboxes, det_labels, det_embeds in bbox_list
            # ]
            # return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        # Pass a band of images (can be >2) to simple_test
        N_imgs = len(imgs)
        infer_single = []
        for i in range(N_imgs):
            _ = self.simple_test(imgs[i], img_metas[i][0], rescale=rescale, return_dict=False)
            infer_single.append(_)
        # Compute the matching score across image-pairs
        outputs = []
        for i in range(N_imgs-1):
            boxes1, labels1, embed_feat1 = infer_single[i]
            for j in range(i+1, N_imgs):
                boxes2, labels2, embed_feat2 = infer_single[j]
                if self.test_cfg.codet.get('multiply_obj_score', False):
                    pair_score = self.bbox_head.get_pair_score(
                        embed_feat1, embed_feat2,
                        boxes1[:, -1], boxes2[:, -1])
                else:
                    pair_score = self.bbox_head.get_pair_score(
                        embed_feat1, embed_feat2)

                matching_pairs = self.bbox_head.get_topk_pairs(
                    pair_score, boxes1, boxes2,
                    labels1, labels2, self.test_cfg.codet)

                result = {'img_metas': (
                    img_metas[i][0][0], img_metas[j][0][0])}

                result.update(matching_pairs)
                outputs.append(result)

        return outputs
