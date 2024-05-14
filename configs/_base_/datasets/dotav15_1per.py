_base_ = ['dotav15.py']

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        ann_file='train_1_labeled/annfiles/',
        data_prefix=dict(img_path='train_1_labeled/images/')
        ))



