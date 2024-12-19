## DDPLS: DENSITY-GUIDED DENSE PSEUDO LABEL SELECTION FOR SEMI-SUPERVISED ORIENTED OBJECT DETECTION
This repository provides the complete code for reproducing the results in the paper.
## Introduction
DDPLS is built on MMRotate, a rotated object detection toolbox and benchmark. It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

## File Orgnizations

```
├── configs              
    ├── _base_
    │   |-- datasets/
    |       | -- dota15.py 
    |       |   # dota15 dataset config
    |       | -- dota15_1/5/10/20/30per.py
    |       |   # dota15 1%/5%/10%/20%/30% dataset config
    |       | -- semi_dota15_detection.py
    |       |   # dota15 semi dataset config   
    |   |-- default_runtime.py     
    |       # default runtime config
    ├── rotated_fcos
    |   |-- rotated-fcos-le90_r50_fpn_3x_dotav1.5_1/5/10/20/30per.py           
    |       # rotated fcos 1%/5%/10%/20%/30% config
    |   |-- rotated-fcos-le90_r50_fpn_3x_dotav1.5.py
    |       # rotated fcos 100% config
    ├── ddpls
    |   |-- ddpls_2xb3-180000k_semi-0.01/0.05/0.1/0.2/0.3-dotav1.5.py
    |       # DDPLS 1%/5%/10%/20%/30% config
    |   |-- ddpls_2xb3-180000k_semi-full-dotav1.5.py
    |       # DDPLS 100% config
├── mmrotate
    |-- models/detectors/DDPLS.py
    |   # DDPLS class file
    |-- models/detectors/semi_base.py
    |   # Semi base class file
├── tools
    |-- ss_data_lists/
    |    |  -- 1/5/10/20/30p_list.json
    |    |    # dota15 dataset 1/5/10/20/30% split lists
    |-- split_data_via_list.py
    |   # Split dota15 dataset via list
    |-- data/dota/
    |   # dota data preprocessing
    |-- train.py/test.py
    |   # Main file for train and evaluate the models

```

## Usage
### Requirements
- `Pytorch=1.13.x`
- `mmdetection=3.0.0`
- `mmpretrain=1.1.0`

## Installation
For mmdetection and mmpretrain, please refer to [mmdetection](https://mmdetection.readthedocs.io/en/v3.0.0/get_started.html) and [mmpretrain](https://mmpretrain.readthedocs.io/en/latest/get_started.html) for installation.
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet==3.0.0

mim install mmpretrain==1.1.0
```
After that
```angular2html
pip install future tensorboard
cd DDPLS
pip install -v -e .
```

## Data Preparation

Please refer to [data_preparation.md](tools/data/dota/README.md) to prepare the original data.

After that, the data folder should be organized as follows,
```
├── data
│   ├── split_ss_dota1_5
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── val
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── test
│   │   │   ├── images
│   │   │   ├── annfiles
```
change the `list_dir` and `src_dir` in `tools/split_data_via_list.py` and run it.

After that, the data folder should be organized as follows,
```
├── data
│   ├── split_ss_dota1_5
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_1_labeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_1_unlabeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_5_labeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_5_unlabeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_10_labeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_10_unlabeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_20_labeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_20_unlabeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_30_labeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_30_unlabeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── val
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── test
│   │   │   ├── images
│   │   │   ├── annfiles
```

## Training
For rotated-fcos baseline

- To train with 10% labeled data, run:
```
python tools/train.py configs/rotated_fcos/rotated-fcos-le90_r50_fpn_3x_dotav1.5_10per.py 
```
For DDPLS
- To train DDPLS with 10% labeled data, run:
```
 CUDA_VISIBLE_DEVICES=0,1 PORT=29501 bash ./tools/dist_train.sh configs/ddpls/ddpls_2xb3-180000k_semi-0.1-dotav1.5.py 2
```

## Acknowledgement
- This code was inspired from mmrotate, mmdet and SOOD. Thanks for their great works!





