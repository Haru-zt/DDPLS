_base_ = [
    'rotated-fcos-le90_r50_fpn_dotav15.py', '../_base_/default_runtime.py',
    '../_base_/datasets/semi_dotav15_detection.py'
]
# todo: fix this import issue
custom_imports = dict(
    imports=['mmrotate.engine.hooks.mean_teacher_hook'],
    allow_failed_imports=False)

detector = _base_.model
model = dict(
    _delete_=True,
    type='DDPLS',
    detector=detector,
    data_preprocessor=dict(
        type='mmdet.MultiBranchDataPreprocessor',
        data_preprocessor=detector.data_preprocessor),
    semi_train_cfg=dict(
        freeze_teacher=True,
        iter_count=0,
        burn_in_steps=10000,
        sup_weight=1.0,
        unsup_weight=1.0,
        k_ratio=0.01,
        cls_weight=1.0,
        bbox_loss_type='l1',    # option: 'l1' 'RotatedIoULoss', 'DenseTeacherIoULoss'
        bbox_weight=1.0,
        centerness_weight=1.0,
        visual=True,
        visual_interval=800,
    ),
    semi_test_cfg=dict(predict_on='teacher'))

# 30% coco train2017 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
labeled_dataset.ann_file = 'train_30_labeled/annfiles'
labeled_dataset.data_prefix = dict(img_path='train_30_labeled/images/')

unlabeled_dataset = _base_.unlabeled_dataset
unlabeled_dataset.ann_file = 'train_30_unlabeled/empty_annfiles/'
unlabeled_dataset.data_prefix = dict(img_path='train_30_unlabeled/images/')

batch_size = 3
num_workers = 6
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(
        #todo fix the height and width problem in GroupMultiSourceSampler
        # new： the same in DOTA
        # type='mmdet.GroupMultiSourceSampler',
        type='mmdet.MultiSourceSampler',
        batch_size=batch_size,
        source_ratio=[2, 1]),
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))

# training schedule for 180k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=180000, val_interval=3200)
val_cfg = dict(type='mmdet.TeacherStudentValLoop')
test_cfg = dict(type='TestLoop')

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=1000,
        end=180000,
        by_epoch=False,
        milestones=[120000, 160000],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2)
)


default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=3200, max_keep_ckpts=1000, save_best='auto'))

log_processor = dict(by_epoch=False)

custom_hooks = [
    dict(type='MeanTeacherHook', start_iter=3200, momentum=0.0004),
]

randomness = dict(
    seed=42,
    diff_rank_seed=False,
    deterministic=True
)

vis_backends = [dict(type='TensorboardVisBackend')]

visualizer = dict(
    type='RotLocalVisualizer', vis_backends=vis_backends, name='visualizer', save_dir='work_dirs/ddpls_2xb3-180000k_semi-0.3-dotav1.5')