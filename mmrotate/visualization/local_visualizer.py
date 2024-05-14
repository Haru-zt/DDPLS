# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import numpy as np
import torch
from mmdet.structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon
from mmdet.visualization import DetLocalVisualizer, jitter_color
from mmdet.visualization.palette import _get_adaptive_scales
from mmengine.structures import InstanceData
from torch import Tensor

from mmrotate.registry import VISUALIZERS
from mmrotate.structures.bbox import QuadriBoxes, RotatedBoxes
from .palette import get_palette

from mmengine.dist import master_only
import mmcv


@VISUALIZERS.register_module()
class RotLocalVisualizer(DetLocalVisualizer):
    """MMRotate Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str, tuple(int), optional): Color of texts.
            The tuple of color should be in BGR order.
            Defaults to (200, 200, 200).
        mask_color (str, tuple(int), optional): Color of masks.
            The tuple of color should be in BGR order.
            Defaults to None.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        alpha (int, float): The transparency of bboxes or mask.
            Defaults to 0.8.
    """

    def _draw_instances(self, image: np.ndarray, instances: ['InstanceData'],
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.
        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)

        if 'bboxes' in instances:
            bboxes = instances.bboxes
            labels = instances.labels

            max_label = int(max(labels) if len(labels) > 0 else 0)
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            bbox_color = palette if self.bbox_color is None \
                else self.bbox_color
            bbox_palette = get_palette(bbox_color, max_label + 1)
            colors = [bbox_palette[label] for label in labels]

            if isinstance(bboxes, Tensor):
                if bboxes.size(-1) == 5:
                    bboxes = RotatedBoxes(bboxes)
                elif bboxes.size(-1) == 8:
                    bboxes = QuadriBoxes(bboxes)
                else:
                    raise TypeError(
                        'Require the shape of `bboxes` to be (n, 5) '
                        'or (n, 8), but get `bboxes` with shape being '
                        f'{bboxes.shape}.')

            bboxes = bboxes.cpu()
            polygons = bboxes.convert_to('qbox').tensor
            polygons = polygons.reshape(-1, 4, 2)
            polygons = [p for p in polygons]
            self.draw_polygons(
                polygons,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=self.line_width)

            positions = bboxes.centers + self.line_width
            scales = _get_adaptive_scales(bboxes.areas)

            for i, (pos, label) in enumerate(zip(positions, labels)):
                label_text = classes[
                    label] if classes is not None else f'class {label}'
                if 'scores' in instances:
                    score = round(float(instances.scores[i]) * 100, 1)
                    label_text += f': {score}'

                self.draw_texts(
                    label_text,
                    pos,
                    colors=text_colors[i],
                    font_sizes=int(13 * scales[i]),
                    bboxes=[{
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    }])

        if 'masks' in instances:
            labels = instances.labels
            masks = instances.masks
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()

            masks = masks.astype(bool)

            max_label = int(max(labels) if len(labels) > 0 else 0)
            mask_color = palette if self.mask_color is None \
                else self.mask_color
            mask_palette = get_palette(mask_color, max_label + 1)
            colors = [jitter_color(mask_palette[label]) for label in labels]
            text_palette = get_palette(self.text_color, max_label + 1)
            text_colors = [text_palette[label] for label in labels]

            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)
            self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)
        return self.get_image()

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            draw_pseudo: bool = False,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        image = image.clip(0, 255).astype(np.uint8)
        original_image = image.copy()
        # input is original img,
        # so we need to transform it to the actual image in training
        if data_sample.scale_factor is not None:
            scale_factor = data_sample.scale_factor[0]
            image = mmcv.imrescale(image, scale_factor)
        if hasattr(data_sample, 'flip'):
            if data_sample.flip == True:
                image = mmcv.imflip(image, direction=data_sample.flip_direction)
        image = mmcv.impad(image, shape=data_sample.pad_shape, pad_val=(114, 114, 114))

        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None
        pseudo_img_data = []

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            gt_img_data = image
            if 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances(image,
                                                   data_sample.gt_instances,
                                                   classes, palette)

            if 'gt_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                gt_img_data = self._draw_panoptic_seg(
                    gt_img_data, data_sample.gt_panoptic_seg, classes)

        if draw_pred and data_sample is not None:
            pred_img_data = image
            if 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr]
                pred_img_data = self._draw_instances(image, pred_instances,
                                                     classes, palette)
            if 'pred_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                pred_img_data = self._draw_panoptic_seg(
                    pred_img_data, data_sample.pred_panoptic_seg.numpy(),
                    classes)

        if draw_pseudo and data_sample is not None:

            instances = InstanceData(bboxes=[], labels=[])
            cls_map = {'plane': 0, 'baseball-diamond': 1, 'bridge': 2, 'ground-track-field': 3, 'small-vehicle': 4,
                       'large-vehicle': 5, 'ship': 6, 'tennis-court': 7, 'basketball-court': 8, 'storage-tank': 9,
                       'soccer-ball-field': 10, 'roundabout': 11, 'harbor': 12, 'swimming-pool': 13,
                       'helicopter': 14,
                       'container-crane': 15}
            annfile_path = '/'.join(data_sample.img_path.split('/')[:-2]) + '/annfiles/' + \
                           data_sample.img_path.split('/')[-1][:-3] + 'txt'
            with open(annfile_path) as f:
                s = f.readlines()
                for si in s:
                    bbox_info = si.split()
                    instances.bboxes.append([float(i) for i in bbox_info[:8]])
                    cls_name = bbox_info[8]
                    instances.labels.append(cls_map[cls_name])
                instances.bboxes = torch.tensor(instances.bboxes)
                instances.labels = torch.tensor(instances.labels)

            if len(instances.labels) > 0:
                original_image = self._draw_instances(original_image,
                                                   instances,
                                                   classes, palette)
                if data_sample.scale_factor is not None:
                    scale_factor = data_sample.scale_factor[0]
                    original_image = mmcv.imrescale(original_image, scale_factor)
                if data_sample.flip == True:
                    original_image = mmcv.imflip(original_image, direction=data_sample.flip_direction)
                original_image = mmcv.impad(original_image, shape=data_sample.pad_shape, pad_val=(114, 114, 114))
                pseudo_img_data.append(original_image)
            else:
                if data_sample.scale_factor is not None:
                    scale_factor = data_sample.scale_factor[0]
                    original_image = mmcv.imrescale(original_image, scale_factor)
                if data_sample.flip == True:
                    original_image = mmcv.imflip(original_image, direction=data_sample.flip_direction)
                original_image = mmcv.impad(original_image, shape=data_sample.pad_shape, pad_val=(114, 114, 114))
                pseudo_img_data.append(original_image)

            if 'pseudo_points' in data_sample and 'predict_probs' in data_sample and 'predict_labels' in data_sample:
                for i in range(len(data_sample.pseudo_points)):
                    self.set_image(image)
                    point_color = [palette[label] for label in data_sample.predict_labels[i]]
                    self.draw_points(data_sample.pseudo_points[i], colors=point_color, sizes=data_sample.predict_probs[i].cpu() * 100)
                    pseudo_img_data.append(self.get_image())

        if gt_img_data is not None and pred_img_data is not None and len(pseudo_img_data) > 0:
            drawn_img = np.concatenate((*pseudo_img_data, gt_img_data, pred_img_data), axis=1)
        elif pred_img_data is not None and len(pseudo_img_data) > 0:
            drawn_img = np.concatenate((*pseudo_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        elif len(pseudo_img_data) > 0:
            drawn_img = np.concatenate(pseudo_img_data, axis=1)
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image


        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)