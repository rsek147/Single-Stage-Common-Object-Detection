import pandas as pd
import numpy as np

data_root = '/data/coco/'

df = pd.read_csv('configs/sscod/coco/coco_classes_caseB.csv')
seen_classes = np.where(df['take'].values != 0)[0] + 1
unseen_classes = np.where(df['take'].values == 0)[0] + 1
# used_classes_for_eval = None
# used_classes_for_eval = seen_classes
used_classes_for_eval = unseen_classes

# model settings
model = dict(
    type='SSCOD',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_eval=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        conv_cfg=dict(type='ConvWS'),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        num_outs=5),
    bbox_head=dict(
        type='ATSS_COD_Head',
        stacked_obj_convs=4,
        embed_channels=256,
        exp_type=2,
        unseen_classID=unseen_classes,
        classwise_loss=None,
        pairwise_loss=dict(
            type='CurContrastLoss', in_channels=256,
            embed_channels=None, scale=1.0, margin=0.5,
            easy_margin=False, ignore_class0=True, loss_weight=1.0),
        embed_norm_cfg=None,
        num_classes=81,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=8,
        scales_per_octave=1,
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        conv_cfg=dict(type='ConvWS'),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True,
            gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100,
    codet=dict(
        multiply_obj_score=False,
        max_pairs=100,
        matching_thr=0.54))
# dataset settings
dataset_type = 'VOCDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ObjDetAugmentation', policy='v0'),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CocoContrastiveDataset',
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/train2017/',
        pipeline=train_pipeline,
        pipeline_aug=train_pipeline,
        pair_itself=False),
    val=dict(
        type='CocoCODDataset',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type='CocoCODDataset',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline,
        used_class_ids=used_classes_for_eval,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=4e-2, momentum=0.9, weight_decay=4e-5)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnealing', min_lr=4e-4, by_epoch=False,
    warmup='linear', warmup_iters=500, warmup_ratio=1.0 / 3)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
