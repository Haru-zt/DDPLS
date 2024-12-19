# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Optional, Tuple

import torch
import torch.futures
from einops import rearrange
from torch import Tensor

from mmdet.models.utils import rename_loss_dict, reweight_loss_dict
from mmrotate.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.losses import CrossEntropyLoss, GIoULoss, QualityFocalLoss, SmoothL1Loss
from mmrotate.models.losses import RotatedIoULoss
from mmrotate.models.detectors.semi_base import RotatedSemiBaseDetector

import torch.nn.functional as F
import torch.nn as nn
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmrotate.registry import TASK_UTILS

import mmcv
from mmrotate.visualization import RotLocalVisualizer
from mmengine.logging import MessageHub

@MODELS.register_module()
class DDPLS(RotatedSemiBaseDetector):
    """
    Args:
        detector (:obj:`ConfigDict` or dict): The detector config.
        semi_train_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised training config.
        semi_test_cfg (:obj:`ConfigDict` or dict, optional):
            The semi-supervised testing config.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.num_classes = self.student.bbox_head.cls_out_channels

        self.loss_cls_weight = self.semi_train_cfg.get('cls_weight', 1.)

        self.bbox_loss_type = self.semi_train_cfg.get('bbox_loss_type', 'l1')
        assert self.bbox_loss_type in ['RotatedIoULoss', 'l1', 'DenseTeacherIoULoss']
        if self.bbox_loss_type == 'RotatedIoULoss':  # abandoned
            self.bbox_coder = TASK_UTILS.build(dict(type='DistanceAnglePointCoder', angle_version='le90'))
            self.angle_coder = TASK_UTILS.build(dict(type='PseudoAngleCoder'))
            self.loss_bbox = RotatedIoULoss(
                reduction='none',
                loss_weight=self.semi_train_cfg.get('bbox_weight', 1.))
        elif self.bbox_loss_type == 'l1':
            self.loss_bbox = SmoothL1Loss(
                reduction='none',
                loss_weight=self.semi_train_cfg.get('bbox_weight', 1.))

        # use_sigmoid=True for BCEWithLogits
        self.loss_centerness = CrossEntropyLoss(
            use_sigmoid=True,
            reduction='none',
            loss_weight=self.semi_train_cfg.get('centerness_weight', 1.))

        self.prior_generator = MlvlPointGenerator(strides=[8, 16, 32, 64, 128], offset=0.5)


    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[SampleList, Optional[dict]]:
        """Get pseudo instances from teacher model."""
        # todo: find out the error reason
        #  self.teacher.eval()
        dense_predicts = self.teacher(batch_inputs)
        batch_info = {}
        batch_info['dense_predicts'] = dense_predicts
        return batch_data_samples, batch_info

    @staticmethod
    def permute_to_N_HWA_K(tensor: Tensor, K: int) -> Tensor:
        """Transpose/reshape a tensor from (N, (A x K), H, W) to (N, (HxWxA),
               K)"""

        assert tensor.dim() == 4, tensor.shape
        tensor = rearrange(tensor, 'N (A K) H W -> N (H W A) K', K=K)
        return tensor

    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_info: Optional[dict] = None) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
            batch_info (dict): Batch information of teacher model
                forward propagation process. Defaults to None.
        Returns:
            dict: A dictionary of loss components
        """

        # note that such way only works for one unlabeled data sample in a batch for a gpu device, if you want to use
        # multiple unlabeled data samples in a batch, you should modify the code.
        teacher_cls_scores_logits, teacher_bbox_preds, teacher_angle_pred, teacher_centernesses = batch_info[
            'dense_predicts']
        student_cls_scores_logits, student_bbox_preds, student_angle_pred, student_centernesses = self.student(
            batch_inputs)

        # featmap = teacher_cls_scores_logits
        # featmap1 = student_cls_scores_logits

        featmap_sizes = [featmap.size()[-2:] for featmap in teacher_cls_scores_logits]
        batch_size = teacher_cls_scores_logits[0].size(0)

        student_cls_scores_logits = torch.cat([
            self.permute_to_N_HWA_K(x, self.num_classes)
            for x in student_cls_scores_logits
        ],
            dim=1).view(-1, self.num_classes)
        teacher_cls_scores_logits = torch.cat([
            self.permute_to_N_HWA_K(x, self.num_classes)
            for x in teacher_cls_scores_logits
        ],
            dim=1).view(-1, self.num_classes)

        student_bbox_preds = torch.cat(
            [self.permute_to_N_HWA_K(torch.cat([x, y], dim=1), 5) for x, y in zip(student_bbox_preds, student_angle_pred)],
            dim=1).view(-1, 5)
        teacher_bbox_preds = torch.cat(
            [self.permute_to_N_HWA_K(torch.cat([x, y], dim=1), 5) for x, y in zip(teacher_bbox_preds, teacher_angle_pred)],
            dim=1).view(-1, 5)

        # teacher_angle_pred = torch.cat(
        #     [self.permute_to_N_HWA_K(x, 1) for x in teacher_angle_pred],
        #     dim=1).view(-1, 1)
        # student_angle_pred = torch.cat(
        #     [self.permute_to_N_HWA_K(x, 1) for x in student_angle_pred],
        #     dim=1).view(-1, 1)

        student_centernesses = torch.cat(
            [self.permute_to_N_HWA_K(x, 1) for x in student_centernesses],
            dim=1).view(-1, 1)
        teacher_centernesses = torch.cat(
            [self.permute_to_N_HWA_K(x, 1) for x in teacher_centernesses],
            dim=1).view(-1, 1)

        with torch.no_grad():
            # Region Selection according to the FSR
            # ratio = self.semi_train_cfg.get('k_ratio', 0.01)
            # count_num = int(teacher_cls_scores_logits.size(0) * ratio)

            teacher_probs = teacher_cls_scores_logits.sigmoid()
            max_vals = torch.max(teacher_probs, 1)[0]
            avg_fsr = max_vals.mean()
            avg_fsr = self.semi_train_cfg.get('fsr_weight', 1.0) * avg_fsr
            if torch.isnan(avg_fsr):
                avg_fsr = 0.0

            indictor = self.semi_train_cfg.get('indictor', 'top')
            assert indictor in ['top', 'threshold']
            if indictor == 'top':
                count_num = int(teacher_cls_scores_logits.size(0) * avg_fsr)

                # monte carlo sampling
                # index = torch.LongTensor(random.sample(range(len(max_vals)), 100))
                # mt_max_vals = max_vals[index]
                # avg_mt_fsr = mt_max_vals.mean()

                sorted_vals, sorted_inds = torch.topk(max_vals,
                                                      teacher_cls_scores_logits.size(0))
                mask = torch.zeros_like(max_vals)
                mask[sorted_inds[:count_num]] = 1.
                # mask[sorted_inds[:count_num]] = sorted_vals[:count_num]
                fg_num = sorted_vals[:count_num].sum()

                message_hub = MessageHub.get_current_instance()
                message_hub.update_scalar('train/avg_fsr', avg_fsr)
                message_hub.update_scalar('train/fg_num', fg_num)

                b_mask = mask > 0.
            elif indictor == 'threshold':
                mask = torch.zeros_like(max_vals)
                mask[max_vals > avg_fsr] = 1.
                fg_num = max_vals[max_vals > avg_fsr].sum()
                message_hub = MessageHub.get_current_instance()
                message_hub.update_scalar('train/avg_fsr', avg_fsr)
                message_hub.update_scalar('train/fg_num', fg_num)

                b_mask = mask > 0.

        if avg_fsr == 0.0:
            loss_cls = torch.tensor(0., device=student_cls_scores_logits.device)
            loss_bbox = torch.tensor(0., device=student_cls_scores_logits.device)
            loss_centerness = torch.tensor(0., device=student_cls_scores_logits.device)
            losses = {
                'loss_cls': loss_cls,
                'loss_bbox': loss_bbox,
                'loss_centerness': loss_centerness,
            }
            unsup_weight = self.semi_train_cfg.get('unsup_weight', 1.)
            burn_in_steps = self.semi_train_cfg.get('burn_in_steps', 6400)

            # apply burnin strategy to reweight the unsupervised weights
            target = burn_in_steps * 2
            if self.iter_count <= target:
                unsup_weight *= (self.iter_count -
                                 burn_in_steps) / burn_in_steps

            return rename_loss_dict('unsup_',
                                    reweight_loss_dict(losses, unsup_weight))


        loss_cls = self.loss_cls_weight * QFLv2(
            student_cls_scores_logits.sigmoid(),
            teacher_cls_scores_logits.sigmoid(),
            weight=mask,
            reduction="sum",
        ) / fg_num

        if self.bbox_loss_type == 'RotatedIoULoss':
            all_level_points = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=student_bbox_preds[0].dtype,
                device=student_bbox_preds[0].device)
            flatten_points = torch.cat(
                [points.repeat(batch_size, 1) for points in all_level_points])

            teacher_angle_pred = self.angle_coder.decode(
                teacher_angle_pred, keepdim=True)
            student_angle_pred = self.angle_coder.decode(
                student_angle_pred, keepdim=True)

            teacher_rbbox_preds = torch.cat(
                [teacher_bbox_preds, teacher_angle_pred], dim=-1)
            student_rbbox_preds = torch.cat(
                [student_bbox_preds, student_angle_pred], dim=-1)
            student_bbox_preds = self.bbox_coder.decode(flatten_points, student_rbbox_preds)[b_mask]
            teacher_bbox_preds = self.bbox_coder.decode(flatten_points, teacher_rbbox_preds)[b_mask]
            loss_bbox = self.loss_bbox(
                student_bbox_preds,
                teacher_bbox_preds,
            ) * teacher_centernesses.sigmoid()[b_mask]
            #todo fix bug
            nan_indexes = ~torch.isnan(loss_bbox)
            if nan_indexes.sum() == 0:
                loss_bbox = torch.zeros(1, device=student_cls_scores_logits.device).sum()
            else:
                loss_bbox = loss_bbox[nan_indexes].mean()

        elif self.bbox_loss_type == 'l1':
            loss_bbox = (self.loss_bbox(student_bbox_preds[b_mask],
                                    teacher_bbox_preds[b_mask])* teacher_centernesses.sigmoid()[b_mask]).mean()

        elif self.bbox_loss_type == 'DenseTeacherIoULoss':
            # loss_bbox = iou_loss(student_bbox_preds[b_mask],
            #                         teacher_bbox_preds[b_mask],
            #                         box_mode="ltrb",
            #                         loss_type="giou",
            #                         reduction="mean")
            loss_bbox = iou_loss(student_bbox_preds,
                                    teacher_bbox_preds,
                                    weight=mask[:, None],
                                    box_mode="ltrb",
                                    loss_type="giou",
                                    reduction="sum")/fg_num

        loss_centerness = F.binary_cross_entropy(
            student_centernesses[b_mask].sigmoid(),
            teacher_centernesses[b_mask].sigmoid(),
            reduction='mean'
        )

        losses = {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_centerness': loss_centerness,
        }

        unsup_weight = self.semi_train_cfg.get('unsup_weight', 1.)
        burn_in_steps = self.semi_train_cfg.get('burn_in_steps', 6400)

        # apply burnin strategy to reweight the unsupervised weights
        target = burn_in_steps * 2
        if self.iter_count <= target:
            unsup_weight *= (self.iter_count -
                             burn_in_steps) / burn_in_steps

        return rename_loss_dict('unsup_',
                                    reweight_loss_dict(losses, unsup_weight))

def QFLv2(pred_sigmoid,
          teacher_sigmoid,
          weight=None,
          beta=2.0,
          reduction='mean'):
    # all goes to 0
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = weight > 0

    # positive goes to bbox quality
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss

def iou_loss(
    inputs,
    targets,
    weight=None,
    box_mode="xyxy",
    loss_type="iou",
    smooth=False,
    reduction="none"
):
    """
    Compute iou loss of type ['iou', 'giou', 'linear_iou']

    Args:
        inputs (tensor): pred values
        targets (tensor): target values
        weight (tensor): loss weight
        box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
        loss_type (str): 'giou' or 'iou' or 'linear_iou'
        reduction (str): reduction manner

    Returns:
        loss (tensor): computed iou loss.
    """
    if box_mode == "ltrb":
        inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
        targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = torch.finfo(torch.float32).eps

    inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) \
        * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) \
        * (targets[..., 3] - targets[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(inputs[..., 2], targets[..., 2])
                   - torch.max(inputs[..., 0], targets[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(inputs[..., 3], targets[..., 3])
                   - torch.max(inputs[..., 1], targets[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect
    if smooth:
        ious = (area_intersect + 1) / (area_union + 1)
    else:
        ious = area_intersect / area_union.clamp(min=eps)

    if loss_type == "iou":
        loss = -ious.clamp(min=eps).log()
    elif loss_type == "linear_iou":
        loss = 1 - ious
    elif loss_type == "giou":
        g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) \
            - torch.min(inputs[..., 0], targets[..., 0])
        g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) \
            - torch.min(inputs[..., 1], targets[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1 - gious
    else:
        raise NotImplementedError
    if weight is not None:
        loss = loss * weight.view(loss.size())
        if reduction == "mean":
            loss = loss.sum() / max(weight.sum().item(), eps)
    else:
        if reduction == "mean":
            loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()

    return loss