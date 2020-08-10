import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET

from .voc import VOCDataset
from .registry import DATASETS


@DATASETS.register_module
class VOC_COD_Dataset(VOCDataset):

    def __init__(self,
                 ignore_difficulty=False,
                 used_class_ids=None,
                 treat_unused_classes_as_bg=True,
                 **kwargs):

        super(VOCDataset, self).__init__(**kwargs)
        self.ignore_difficulty = ignore_difficulty
        self.treat_unused_classes_as_bg = treat_unused_classes_as_bg

        if used_class_ids is not None:
            self.CLASSES = [self.CLASSES[i-1] for i in used_class_ids]
        self.USED_CLASS_IDS = [self.cat2label[cat] for cat in self.CLASSES]

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations',
                            '{}.xml'.format(img_id))

        tree = ET.parse(xml_path)
        root = tree.getroot()

        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            label = self.cat2label[name]

            # Ignore sample with label not considered for training
            ignore = False
            if label not in self.USED_CLASS_IDS:
                if self.treat_unused_classes_as_bg:
                    continue
                    ignore = True

            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            # if difficult or ignore:
            if (difficult and self.ignore_difficulty) or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

        # Empty gt?
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)

        # bboxes_ignore
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)

        # Return
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
