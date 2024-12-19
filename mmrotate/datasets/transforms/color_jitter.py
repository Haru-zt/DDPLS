#abandon

# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform
from mmcv.transforms import Pad as MMCV_Pad
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine.dataset import BaseDataset
from mmengine.utils import is_str
from numpy import random

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.utils import log_img_scale

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

Number = Union[int, float]

@TRANSFORMS.register_module()
class Colorjitter(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 1)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 0)

    Required Keys:

    - img (np.uint8)

    Modified Keys:

    - img (np.float32)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (sequence): range of contrast.
        saturation_range (sequence): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_range: Sequence[Number] = (0.5, 1.5),
                 contrast_range: Sequence[Number] = (0.5, 1.5),
                 saturation_range: Sequence[Number] = (0.5, 1.5),
                 hue_range: Sequence[Number] = (-0.5, 0.5),
                 prob: float = 0.5)-> None:
        self.brightness_lower, self.brightness_upper = brightness_range
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_lower, self.hue_upper = hue_range
        self.prob = prob

    @cache_randomness
    def _random_flags(self) -> Sequence[Number]:
        # mode = random.randint(2)
        # brightness_flag = random.randint(2)
        # contrast_flag = random.randint(2)
        # saturation_flag = random.randint(2)
        # hue_flag = random.randint(2)
        mode = 1
        if random.random() < self.prob:
            brightness_flag = contrast_flag = saturation_flag = hue_flag = 1
        else:
            brightness_flag = contrast_flag = saturation_flag = hue_flag = 0
        brightness_value = random.uniform(self.brightness_lower, self.brightness_upper)
        contrast_value = random.uniform(self.contrast_lower, self.contrast_upper)
        saturation_value = random.uniform(self.saturation_lower,
                                          self.saturation_upper)
        hue_value = random.uniform(self.hue_lower, self.hue_upper)

        return (mode, brightness_flag, contrast_flag, saturation_flag,
                hue_flag, brightness_value, contrast_value, saturation_value, hue_value)

    def transform(self, results: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        assert 'img' in results, '`img` is not found in results'
        img = results['img']
        img = img.astype(np.float32)

        (mode, brightness_flag, contrast_flag, saturation_flag,
         hue_flag, brightness_value, contrast_value, saturation_value,
         hue_value) = self._random_flags()

        # random brightness
        if brightness_flag:
            img *= brightness_value

        # mode == 1 --> do random contrast first
        # mode == 0 --> do random contrast last
        if mode == 1:
            if contrast_flag:
                img *= contrast_value

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if saturation_flag:
            img[..., 1] *= saturation_value
            # For image(type=float32), after convert bgr to hsv by opencv,
            # valid saturation value range is [0, 1]
            if saturation_value > 1:
                img[..., 1] = img[..., 1].clip(0, 1)

        # random hue
        if hue_flag:
            img[..., 0] *= hue_value
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if contrast_flag:
                img *= contrast_value

        results['img'] = img
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(brightness_delta={self.brightness_delta}, '
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)}, '
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)}, '
        repr_str += f'hue_delta={self.hue_delta})'
        repr_str += f'prob={self.prob})'
        return repr_str