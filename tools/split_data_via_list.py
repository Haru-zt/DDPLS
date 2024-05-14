#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import json
import os
import shutil

def split_img_with_list(list_dir, src_dir):

    list_file = [None, None, None, None, None]
    list_file[0] = os.path.join(list_dir, '10p_list.json')
    list_file[1] = os.path.join(list_dir, '20p_list.json')
    list_file[2] = os.path.join(list_dir, '30p_list.json')
    list_file[3] = os.path.join(list_dir, '5p_list.json')
    list_file[4] = os.path.join(list_dir, '1p_list.json')
    assert all([os.path.exists(list_file_) for list_file_ in list_file])

    file_list = [list(), list(), list(), list(), list()]
    for i in range(0, len(list_file)):
        with open(list_file[i], 'r', encoding='utf-8') as f:
            file_list[i] = json.load(f)

    all_files = dict()

    train_dir = os.path.join(src_dir, 'train')
    labeled10_out_dir = os.path.join(src_dir, 'train_10_labeled')
    unlabeled10_out_dir = os.path.join(src_dir, 'train_10_unlabeled')
    labeled20_out_dir = os.path.join(src_dir, 'train_20_labeled')
    unlabeled20_out_dir = os.path.join(src_dir, 'train_20_unlabeled')
    labeled30_out_dir = os.path.join(src_dir, 'train_30_labeled')
    unlabeled30_out_dir = os.path.join(src_dir, 'train_30_unlabeled')
    labeled5_out_dir = os.path.join(src_dir, 'train_5_labeled')
    unlabeled5_out_dir = os.path.join(src_dir, 'train_5_unlabeled')
    labeled1_out_dir = os.path.join(src_dir, 'train_1_labeled')
    unlabeled1_out_dir = os.path.join(src_dir, 'train_1_unlabeled')

    train_img_dir = os.path.join(train_dir, 'images')
    train_ann_dir = os.path.join(train_dir, 'annfiles')

    for file_ in glob.glob(os.path.join(train_img_dir, '*.png')):
        all_files[file_.split('/')[-1]] = file_
    print(f"Total images: {len(all_files)}")

    if os.path.exists(labeled10_out_dir):
        shutil.rmtree(labeled10_out_dir)
    if os.path.exists(unlabeled10_out_dir):
        shutil.rmtree(unlabeled10_out_dir)
    if os.path.exists(labeled20_out_dir):
        shutil.rmtree(labeled20_out_dir)
    if os.path.exists(unlabeled20_out_dir):
        shutil.rmtree(unlabeled20_out_dir)
    if os.path.exists(labeled30_out_dir):
        shutil.rmtree(labeled30_out_dir)
    if os.path.exists(unlabeled30_out_dir):
        shutil.rmtree(unlabeled30_out_dir)
    if os.path.exists(labeled5_out_dir):
        shutil.rmtree(labeled5_out_dir)
    if os.path.exists(unlabeled5_out_dir):
        shutil.rmtree(unlabeled5_out_dir)
    if os.path.exists(labeled1_out_dir):
        shutil.rmtree(labeled1_out_dir)
    if os.path.exists(unlabeled1_out_dir):
        shutil.rmtree(unlabeled1_out_dir)
    os.makedirs(labeled10_out_dir+'/images')
    os.makedirs(labeled10_out_dir+'/annfiles')
    os.makedirs(unlabeled10_out_dir+'/images')
    os.makedirs(unlabeled10_out_dir+'/annfiles')
    os.makedirs(unlabeled10_out_dir+'/empty_annfiles')
    os.makedirs(labeled20_out_dir+'/images')
    os.makedirs(labeled20_out_dir+'/annfiles')
    os.makedirs(unlabeled20_out_dir+'/images')
    os.makedirs(unlabeled20_out_dir+'/annfiles')
    os.makedirs(unlabeled20_out_dir+'/empty_annfiles')
    os.makedirs(labeled30_out_dir+'/images')
    os.makedirs(labeled30_out_dir+'/annfiles')
    os.makedirs(unlabeled30_out_dir+'/images')
    os.makedirs(unlabeled30_out_dir+'/annfiles')
    os.makedirs(unlabeled30_out_dir+'/empty_annfiles')
    os.makedirs(labeled5_out_dir+'/images')
    os.makedirs(labeled5_out_dir+'/annfiles')
    os.makedirs(unlabeled5_out_dir+'/images')
    os.makedirs(unlabeled5_out_dir+'/annfiles')
    os.makedirs(unlabeled5_out_dir+'/empty_annfiles')
    os.makedirs(labeled1_out_dir+'/images')
    os.makedirs(labeled1_out_dir+'/annfiles')
    os.makedirs(unlabeled1_out_dir+'/images')
    os.makedirs(unlabeled1_out_dir+'/annfiles')
    os.makedirs(unlabeled1_out_dir+'/empty_annfiles')


    for file_name, file_path in all_files.items():
        if (file_name.split('__')[0]+'.png') in file_list[0]:
            os.symlink(file_path, os.path.join(labeled10_out_dir, 'images', file_name))
            os.symlink(os.path.join('/'.join(file_path.split('/')[0:-2]), 'annfiles', file_path.split('/')[-1]).split('.')[0]+'.txt',
                       os.path.join(labeled10_out_dir, 'annfiles', file_name.split('.')[0]+'.txt'))
        if (file_name.split('__')[0]+'.png') in file_list[1]:
            os.symlink(file_path, os.path.join(labeled20_out_dir, 'images', file_name))
            os.symlink(os.path.join('/'.join(file_path.split('/')[0:-2]), 'annfiles', file_path.split('/')[-1]).split('.')[0]+'.txt',
                       os.path.join(labeled20_out_dir, 'annfiles', file_name.split('.')[0]+'.txt'))
        if (file_name.split('__')[0]+'.png') in file_list[2]:
            os.symlink(file_path, os.path.join(labeled30_out_dir, 'images', file_name))
            os.symlink(os.path.join('/'.join(file_path.split('/')[0:-2]), 'annfiles', file_path.split('/')[-1]).split('.')[0]+'.txt',
                       os.path.join(labeled30_out_dir, 'annfiles', file_name.split('.')[0]+'.txt'))
        if (file_name.split('__')[0]+'.png') in file_list[3]:
            os.symlink(file_path, os.path.join(labeled5_out_dir, 'images', file_name))
            os.symlink(os.path.join('/'.join(file_path.split('/')[0:-2]), 'annfiles', file_path.split('/')[-1]).split('.')[0]+'.txt',
                       os.path.join(labeled5_out_dir, 'annfiles', file_name.split('.')[0]+'.txt'))
        if (file_name.split('__')[0]+'.png') not in file_list[0]:
            os.symlink(file_path, os.path.join(unlabeled10_out_dir, 'images', file_name))
            os.symlink(os.path.join('/'.join(file_path.split('/')[0:-2]), 'annfiles', file_path.split('/')[-1]).split('.')[0]+'.txt',
                       os.path.join(unlabeled10_out_dir, 'annfiles', file_name.split('.')[0]+'.txt'))
            open(os.path.join(unlabeled10_out_dir, 'empty_annfiles', file_name.split('.')[0]+'.txt'), 'w').close()
        if (file_name.split('__')[0]+'.png') not in file_list[1]:
            os.symlink(file_path, os.path.join(unlabeled20_out_dir, 'images', file_name))
            os.symlink(os.path.join('/'.join(file_path.split('/')[0:-2]), 'annfiles', file_path.split('/')[-1]).split('.')[0] + '.txt',
                       os.path.join(unlabeled20_out_dir, 'annfiles', file_name.split('.')[0] + '.txt'))
            open(os.path.join(unlabeled20_out_dir, 'empty_annfiles', file_name.split('.')[0]+'.txt'), 'w').close()
        if (file_name.split('__')[0]+'.png') not in file_list[2]:
            os.symlink(file_path, os.path.join(unlabeled30_out_dir, 'images', file_name))
            os.symlink(os.path.join('/'.join(file_path.split('/')[0:-2]), 'annfiles', file_path.split('/')[-1]).split('.')[0] + '.txt',
                       os.path.join(unlabeled30_out_dir, 'annfiles', file_name.split('.')[0] + '.txt'))
            open(os.path.join(unlabeled30_out_dir, 'empty_annfiles', file_name.split('.')[0]+'.txt'), 'w').close()
        if (file_name.split('__')[0]+'.png') not in file_list[3]:
            os.symlink(file_path, os.path.join(unlabeled5_out_dir, 'images', file_name))
            os.symlink(os.path.join('/'.join(file_path.split('/')[0:-2]), 'annfiles', file_path.split('/')[-1]).split('.')[0] + '.txt',
                       os.path.join(unlabeled5_out_dir, 'annfiles', file_name.split('.')[0] + '.txt'))
            open(os.path.join(unlabeled5_out_dir, 'empty_annfiles', file_name.split('.')[0]+'.txt'), 'w').close()
        if (file_name.split('__')[0]+'.png') in file_list[4]:
            os.symlink(file_path, os.path.join(labeled1_out_dir, 'images', file_name))
            os.symlink(os.path.join('/'.join(file_path.split('/')[0:-2]), 'annfiles', file_path.split('/')[-1]).split('.')[0]+'.txt',
                       os.path.join(labeled1_out_dir, 'annfiles', file_name.split('.')[0]+'.txt'))
        if (file_name.split('__')[0]+'.png') not in file_list[4]:
            os.symlink(file_path, os.path.join(unlabeled1_out_dir, 'images', file_name))
            os.symlink(os.path.join('/'.join(file_path.split('/')[0:-2]), 'annfiles', file_path.split('/')[-1]).split('.')[0]+'.txt',
                       os.path.join(unlabeled1_out_dir, 'annfiles', file_name.split('.')[0]+'.txt'))
            open(os.path.join(unlabeled1_out_dir, 'empty_annfiles', file_name.split('.')[0]+'.txt'), 'w').close()
    print(f"Finish symlink labeled image.")


if __name__ == '__main__':

    list_dir = '/home/xxx/DDPLS/tools/ss_data_lists'
    src_dir = '/home/xxx/DDPLS/data/split_ss_dota1_5'

    split_img_with_list(list_dir, src_dir)

