from ..registry import DETECTORS
from .atss import ATSS


@DETECTORS.register_module
class SSCOD_Baseline(ATSS):

    # def simple_test(self, img, img_meta, rescale=False):
    #     x = self.extract_feat(img)
    #     outs = self.bbox_head(x)
    #     bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
    #     bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

    #     det_bboxes, det_labels, det_logits = bbox_list[0]
    #     return dict(
    #         bboxes=det_bboxes.cpu().numpy(),
    #         classes=det_labels.cpu().numpy(),
    #         logits=det_logits.cpu().numpy(),
    #     )
	def simple_test(self, img, img_metas, rescale=False, return_dict=True):
		x = self.extract_feat(img)
		outs = self.bbox_head(x)
		bbox_inputs = outs + (img_metas, self.test_cfg, rescale)
		bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
		det_result = bbox_list[0]

		if return_dict:
			return {'boxes':det_result[0].cpu().numpy(),
					'classes':det_result[1].cpu().numpy(),
					'embed':det_result[2].cpu().numpy()}
		else:
			bbox_results = [
				bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
				for det_bboxes, det_labels, det_embeds in bbox_list
			]
			return bbox_results[0]

	def aug_test(self, imgs,img_metas,rescale=False):
		# Pass a band of images (can be >2) to simple_test
		N_imgs = len(imgs)
		infer_single = [self.simple_test(imgs[i][0],img_metas[i][0],rescale,return_dict=False) for i in range(N_imgs)]

		# Compute the matching score across image-pairs
		outputs=[]
		for i in range(N_imgs-1):
			boxes1, labels1, embed_feat1= infer_single[i]
			for j in range(i+1,N_imgs):
				boxes2, labels2, embed_feat2= infer_single[j]
				if self.test_cfg.codet.get('multiply_obj_score',False):
					pair_score = self.bbox_head.get_pair_score(embed_feat1,embed_feat2,
															boxes1[:,-1],boxes2[:,-1])
				else:
					pair_score = self.bbox_head.get_pair_score(embed_feat1,embed_feat2)

				matching_pairs = self.bbox_head.get_topk_pairs(pair_score,
															boxes1,boxes2,
															labels1,labels2,                                                            
															self.test_cfg.codet)
				result={'img_metas':(img_metas[i][0][0],img_metas[j][0][0])}
				result.update(matching_pairs)
				outputs.append(result)

		return outputs
