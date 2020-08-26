import json
import random

from .registry import DATASETS

from .coco import CocoDataset
from .voc import VOCDataset
from .pipelines import Compose


class ContrastiveDataset:
    def __init__(self, pipeline_aug, pair_itself=True, class_imgID_file=None):
        self.pipeline_aug = Compose(pipeline_aug)
        self.class_imgID_file = class_imgID_file
        self.pair_itself = pair_itself
        if not pair_itself:
            self.class_imgID = None
            if class_imgID_file is not None:
                if isinstance(self.class_imgID_file, list):
                    for i, imgID_file in enumerate(self.class_imgID_file):
                        with open(imgID_file) as json_file:
                            if i == 0:
                                self.class_imgID = json.load(json_file, object_hook=lambda d: {
                                                             int(k): v for k, v in d.items()})
                            else:
                                class_imgID_dict = json.load(json_file, object_hook=lambda d: {
                                                             int(k): v for k, v in d.items()})
                                for k, v in class_imgID_dict.items():
                                    self.class_imgID[k].extend(v)
                else:
                    with open(self.class_imgID_file) as json_file:
                        self.class_imgID = json.load(json_file, object_hook=lambda d: {
                                                     int(k): v for k, v in d.items()})
            if self.class_imgID is not None:
                self.generate_img_info_byID()
                # missing=0
                # for cls_id,v in self.class_imgID.items():
                #     for id_ in v:
                #         if id_ in self.img_infos_byID.keys():
                #             print(id_)
                #             missing+=1
                # import pdb; pdb.set_trace()

    def generate_img_info_byID(self):
        pass

    def get_ann_info_by_imgID(self, img_id):
        pass

    def prepare_train_img(self, idx):
        # We get the first sample as usual
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)

        data0 = self.pipeline(results)

        # We try several ways to get the second sample
        if self.pair_itself:
            # We use the same image, but take different Augmentation
            data1 = self.pipeline_aug(results)
        else:
            # We sample another image random from the list
            if self.class_imgID is None:
                # This may not efficient for large dataset
                found_pair = False
                k = 0
                while not found_pair:
                    k += 1
                    idx_aug = random.randint(0, len(self.img_infos)-1)
                    ann_info_aug = self.get_ann_info(idx_aug)
                    if idx_aug != idx and len(set(ann_info).intersection(set(ann_info_aug))) > 0:
                        found_pair = True
                        img_info_aug = self.img_infos[idx_aug]
                        results_aug = dict(
                            img_info=img_info_aug, ann_info=ann_info_aug)
                        if self.proposals is not None:
                            results_aug['proposals'] = self.proposals[idx]
                        self.pre_pipeline(results_aug)
                        data1 = self.pipeline_aug(results_aug)
                    if k > 200:
                        found_pair = True
                        data1 = self.pipeline_aug(results)
            else:
                found_pair = False
                k = 0
                while not found_pair:
                    k += 1
                    # randomly pick an image from other class
                    other_class_id = random.choice(ann_info['labels'])
                    img_id_aug = random.choice(
                        self.class_imgID[other_class_id])
                    if img_id_aug != img_info['id']:
                        found_pair = True
                        img_info_aug, ann_info_aug = self.get_ann_info_by_imgID(
                            img_id_aug)
                        results_aug = dict(
                            img_info=img_info_aug, ann_info=ann_info_aug)
                        if self.proposals is not None:
                            results_aug['proposals'] = self.proposals[idx]
                        self.pre_pipeline(results_aug)
                        data1 = self.pipeline_aug(results_aug)
                    if k > 200:
                        found_pair = True
                        data1 = self.pipeline_aug(results)

        data = dict(img_meta=[data0['img_meta'], data1['img_meta']],
                    img=[data0['img'], data1['img']],
                    gt_bboxes=[data0['gt_bboxes'], data1['gt_bboxes']],
                    gt_labels=[data0['gt_labels'], data1['gt_labels']],
                    )
        if data0.get('gt_masks', None):
            data['gt_masks'] = [data0['gt_masks'], data1['gt_masks']]

        return data


@DATASETS.register_module
class CocoContrastiveDataset(CocoDataset, ContrastiveDataset):
    def __init__(self, pipeline_aug, pair_itself=False, class_imgID_file=None, **kwargs):
        super().__init__(**kwargs)
        ContrastiveDataset.__init__(
            self, pipeline_aug, pair_itself, class_imgID_file)

    def generate_img_info_byID(self):
        self.img_infos_byID = {
            img_info['id']: img_info for img_info in self.img_infos}
        # coco_img_ids = self.coco.getImgIds()
        # self.img_infos_byID ={}
        # min_size=32
        # for i in coco_img_ids:
        #     img_info = self.coco.loadImgs([i])[0]
        #     if min(img_info['width'], img_info['height']) >= min_size:
        #         img_info['filename'] = img_info['file_name']
        #         self.img_infos_byID[img_info['id']]=img_info

    def get_ann_info_by_imgID(self, img_id):
        # follow the code of function get_ann_info
        img_info = self.img_infos_byID.get(img_id)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return img_info, self._parse_ann_info(img_info, ann_info)

    def prepare_train_img(self, idx):
        return ContrastiveDataset.prepare_train_img(self, idx)


@DATASETS.register_module
class VOCContrastiveDataset(VOCDataset, ContrastiveDataset):
    def __init__(self, pipeline_aug, pair_itself=False, class_imgID_file=None, **kwargs):
        super().__init__(**kwargs)
        ContrastiveDataset.__init__(
            self, pipeline_aug, pair_itself, class_imgID_file)

    def generate_img_info_byID(self):
        self.img_infos_byID = {img_info['id']: (img_info, self.get_ann_info(
            idx)) for idx, img_info in enumerate(self.img_infos)}

    def get_ann_info_by_imgID(self, img_id):
        # follow the code of function get_ann_info
        img_info, ann_info = self.img_infos_byID.get(img_id)
        return img_info, ann_info

    def prepare_train_img(self, idx):
        return ContrastiveDataset.prepare_train_img(self, idx)
