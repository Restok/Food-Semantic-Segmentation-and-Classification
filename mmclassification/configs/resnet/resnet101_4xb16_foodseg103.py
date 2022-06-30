_base_ = ['../_base_/datasets/foodseg103_bs16.py', '../_base_/default_runtime.py']

# use different head for multilabel task
model = dict(
    type='ImageClassifier',
    
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        frozen_stages =1,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=12,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=True)))


# optimizer
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0,
    paramwise_cfg=dict(custom_keys={'.backbone.classifier': dict(lr_mult=10)}))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=20, gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=40)