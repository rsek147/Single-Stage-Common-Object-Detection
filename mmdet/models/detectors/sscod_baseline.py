from ..registry import DETECTORS
from .atss import ATSS


@DETECTORS.register_module
class SSCOD_Baseline(ATSS):

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        det_bboxes, det_labels, det_logits = bbox_list[0]
        return dict(
            bboxes=det_bboxes.cpu().numpy(),
            classes=det_labels.cpu().numpy(),
            logits=det_logits.cpu().numpy(),
        )
