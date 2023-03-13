from typing import Union, Sequence, Tuple, List, Optional
from mmdet.utils import (OptMultiConfig, ConfigType,
                         OptConfigType, OptInstanceList)
import torch
from torch import Tensor
import torch.nn as nn

from mmyolo.models.utils import make_divisible
from mmyolo.registry import MODELS, TASK_UTILS

from mmdet.models.utils import multi_apply
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

from mmengine.structures import InstanceData
from mmengine.model import BaseModule
from mmengine.logging import print_log


def corner_confidences(gt_corners, pr_corners, th=80, sharpness=2, im_width=640, im_height=480):
    """calculate confidences"""
    eps = 1e-5
    distthresh = torch.tensor([th])
    dist = gt_corners - pr_corners
    dist = torch.sqrt(torch.sum((dist)**2, dim=1)).squeeze()
    conf = torch.exp(sharpness*(1 - dist/distthresh))-1
    conf0 = torch.exp(sharpness*(1 - torch.zeros(conf.size(0),1))) - 1 + eps
    conf = conf / conf0.squeeze()
    mean_conf = torch.mean(conf, dim=0)
    return mean_conf


@MODELS.register_module()
class YOLO6DHeadModule(BaseModule):
    """YOLO6DHead head module used in 'YOLO6D'.
    
    """
    def __init__(self,
                 num_classes: int = 1,
                 in_channels: Union[int, Sequence] = 1024,
                 widen_factor: float = 1.0,
                 num_base_priors: int = 1,
                 num_cpt: int = 9,
                 featmap_strides: Sequence[int] = [32],
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.widen_factor = widen_factor
        self.num_cpt = num_cpt
        
        self.featmap_strides = featmap_strides
        self.num_out_attrib = 19 + self.num_classes

        self.num_levels = len(self.featmap_strides)
        
        self.num_base_priors = num_base_priors
        
        if isinstance(in_channels, int):
            self.in_channels = [make_divisible(in_channels, widen_factor)
                                * self.num_levels]
        else:
            self.in_channels = [
                make_divisible(i, widen_factor) for i in in_channels]
        
        # 通过卷积将feature map通道数转为: num_keypoints(9)*2 + conf(1) + cls(c)
        self._init_layer()
        
    def _init_layer(self):
        """initialize conv layers in YOLO6D head"""
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Conv2d(self.in_channels[i],
                                  self.num_base_priors * self.num_out_attrib,
                                  1)
            self.convs_pred.append(conv_pred)
        
    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network"""
        
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.convs_pred)
        
    def forward_single(self, x: Tensor,
                       convs: nn.Module) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level"""
        
        pred_map = convs(x)
        bs, _, ny, nx = pred_map.shape
        # 将通道分为若干待回归的数据
        pred_map = pred_map.view(bs, self.num_base_priors,
                                 self.num_out_attrib, ny, nx)
        # 0-17 是控制点的回归值 18是置信度 19之后是类别
        cpt_pred = pred_map[:, :, :self.num_cpt*2, ...].reshape(bs, -1, ny, nx)
        conf_pred = pred_map[:, :, self.num_cpt*2, ...].reshape(bs, -1, ny, nx)
        cls_score = pred_map[
            :, :, (self.num_cpt*2+1):, ...].reshape(bs, -1, ny, nx)
        
        return cpt_pred, conf_pred, cls_score


@MODELS.register_module()
class YOLO6DHead(BaseDenseHead):
    def __init__(self,
                 head_module: ConfigType = dict(
                     type = 'YOLO6DHeadModule',
                    #  num_classes = 1,
                    #  in_channels = 1024,
                     ),
                 # TODO:先验框的大小？？
                #  prior_generator: ConfigType = dict(
                #      type='mmdet.YOLOAnchorGenerator',
                #      base_sizes=[[(1,1)]],
                #      strides=[32]),
                 bbox_coder: ConfigType = dict(type='YOLOv5BBoxCoder'),
                # TODO:创建9个控制点的类？？
                #  cpt_coder: 
                 loss: ConfigType = dict(
                     type='RegionLoss'
                 ),
                #  loss_conf,
                 num_cpts: int = 9,
                 prior_match_thr: float = 4.0,
                 near_neighbor_thr: float = 0.5,
                 obj_level_weights: List[float] = [1.0],
                 ignore_iof_thr: float = -1.0,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 ):
        super().__init__(init_cfg=init_cfg)
        
        self.head_module = MODELS.build(head_module)
        self.num_classes = self.head_module.num_classes
        self.num_cpts = num_cpts
        self.featmap_strides = self.head_module.featmap_strides
        self.num_levels = len(self.featmap_strides)

        self.loss: nn.Module = MODELS.build(loss)
        # self.loss_conf: nn.Module = MODELS.build(loss_conf)
        
        # self.prior_generator = TASK_UTILS.build(prior_generator)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        # self.num_base_priors = self.prior_generator.num_base_priors[0]
        
        self.prior_match_thr = prior_match_thr
        self.near_neighbor_thr = near_neighbor_thr
        self.obj_level_weights = obj_level_weights
        self.ignore_iof_thr = ignore_iof_thr
        
        self.special_init()
    
    def special_init(self):
        """initialization process"""
        assert len(self.obj_level_weights) == len(
            self.featmap_strides) == self.num_levels
        if self.prior_match_thr != 4.0:
            print_log(
                "!!!Now, you've changed the prior_match_thr "
                'parameter to something other than 4.0. Please make sure '
                'that you have modified both the regression formula in '
                'bbox_coder and before loss_box computation, '
                'otherwise the accuracy may be degraded!!!')
        
        if self.num_classes == 1:
            print_log('!!!You are using `YOLOv5Head` with num_classes == 1.'
                      ' The loss_cls will be 0. This is a normal phenomenon.')
        
        # priors_base_sizes = torch.tensor(
        #     self.prior_generator.base_sizes, dtype=torch.float)
        featmap_strides = torch.tensor(
            self.featmap_strides, dtype=torch.float)[:, None, None] # n层 每层n个框 一个框2个值
        # 将先验框归一化？
        # self.register_buffer(
        #     'priors_base_sizes',
        #     priors_base_sizes / featmap_strides,
        #     persistent=False)
        
        grid_offset = torch.tensor([
            [0, 0], # center
            [1, 0], # left
            [0, 1], # up
            [-1, 0], # right
            [0, -1], # bottom
        ]).float()
        self.register_buffer(
            'grid_offset', grid_offset[:, None], persistent=False)
        
        # prior_inds = torch.arange(self.num_base_priors).float().view(
        #     self.num_base_priors, 1)
        # self.register_buffer('prior_inds', prior_inds, persistent=False)

    def forward(self, x: Tuple[Tensor]):
        """Forward features from the upstream network
        
        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
            a 4D-tensor
        
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        return self.head_module(x)

    def predict_by_feat(
                        )-> List[InstanceData]:
        pass

    def loss(self, x, batch_data_samples):
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.
        """        
        if isinstance(batch_data_samples, list):
            outs = super().loss(x, batch_data_samples)
        else:
            outs = self(x)
            # Fast version
            loss_inputs = outs + (batch_data_samples['cpts_confs_labels'],
                                  batch_data_samples['img_metas'])
            losses = self.loss_by_feat(*loss_inputs)

        return losses
        
    def loss_by_feat(self,
                     cpt_preds: Sequence[Tensor],
                     conf_preds: Sequence[Tensor],
                     cls_scores: Sequence[Tensor],
                    #  bbox_preds: Sequence[Tensor],
                     batch_gt_instances: Sequence[InstanceData],
                     batch_img_metas: Sequence[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features
        extracted by the detection head
        """
        if self.ignore_iof_thr != -1:
            pass
        
        # 1. Convert gt to norm format
        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas)
        
        device = cls_scores[0].device
        loss_cls = torch.zeros(1, device=device)
        # loss_box = torch.zeros(1, device=device)
        loss_conf = torch.zeros(1, device=device)
        loss_cpt = torch.zeros(1, device=device)

        scaled_factor = torch.ones(21, device=device)
        
        for i in range(self.num_levels):
            # i表示输出特征层的第i层, cpt_pred (2,18,13,13)
            # conf_pred(1,1,13,13) (batch_size, channels, h, w)
            batch_size, _, h, w = cpt_preds[i].shape
            # batch_size, _, h, w = bbox_preds[i].shape
            target_obj = torch.zeros_like(conf_preds[i])
            
            # empty gt bboxes
            if batch_targets_normed.shape[1] == 0:
                loss_cls += cls_scores[i].sum() * 0
                loss_cpt += cpt_preds[i].sum()*0
                loss_conf += self.loss_conf(
                    conf_preds[i], target_obj) * self.obj_level_weights[i]
                continue
            # priors_base_size_i = self.priors_base_sizes[i]
            
            scaled_factor[2:20] = torch.tensor(
                cpt_preds[i].shape)[[3, 2]*9]
            
            # Scale batch_targets from range 0-1 to range 0-features_maps size
            batch_targets_scaled = batch_targets_normed * scaled_factor
            
            # 2. Shape match
            
            # 3. Prepare pred results and positive sample indexes to
            # calculate class loss and bbox loss
            # TODO: 扩展样本数量 Positive samples with additional neighbors
            batch_targets_cxcy = batch_targets_scaled[:, 2:4]
            grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
            left, up = ((batch_targets_cxcy%1 < self.near_neighbor_thr) &
                        (batch_targets_cxcy>1)).T
            right, bottom = ((grid_xy%1 < self.near_neighbor_thr) & 
                             (grid_xy>1)).T
            offset_inds = torch.stack(
                (torch.ones_like(left), left, up, right, bottom))
            
            batch_targets_scaled = batch_targets_scaled.repeat(
                (5, 1, 1))[offset_inds]
            
            retained_offsets = self.grid_offset.repeat(1, offset_inds.shape[1],
                                                       1)[offset_inds]
            
            # prepare pred results and positive sample indexes to 
            # calculate class loss and bbox loss
            _chunk_targets = batch_targets_scaled.chunk(11, 1)
            img_class_inds = _chunk_targets[0]
            grid_center = _chunk_targets[1]
            grid_corners = [i for i in _chunk_targets[2:-1]]
            conf = _chunk_targets[-1]
            img_inds, class_inds = img_class_inds.long().T

            grid_center_long = (grid_center - 
                                retained_offsets*self.near_neighbor_thr).long()
            grid_corners_long = [(grid_corner -
                                 retained_offsets*self.near_neighbor_thr).long()
                                 for grid_corner in grid_corners]
            cpts_target = torch.cat((grid_center_long, torch.cat(grid_corners_long)), dim=0)
            cpts_target_x = cpts_target[:, 0]
            cpts_target_y = cpts_target[:, 1]
            # TODO: 将cpts_targets分成X,Y

            grid_center_x_inds, grid_center_y_inds = grid_center_long.T
            grid_corners_x_inds = []
            grid_corners_y_inds = []
            for grid_corner_long in grid_corners_long:
                grid_corners_x_inds.append(grid_corner_long.T[0])
                grid_corners_y_inds.append(grid_corner_long.T[1])
            grid_corners_x_inds = torch.stack(grid_corners_x_inds)
            grid_corners_y_inds = torch.stack(grid_corners_y_inds)

            grid_cpts_x_inds = torch.cat((grid_center_x_inds[None,:],
                                          grid_corners_x_inds), dim=0)
            grid_cpts_y_inds = torch.cat((grid_center_y_inds[None, :],
                                          grid_corners_y_inds), dim=0)
            
            # prepare pred results cpt_pred (3,18,13,13)
            retained_center_x_pred = cpt_preds[
                i].reshape(batch_size, -1, h, w)[
                    img_inds, class_inds, grid_cpts_x_inds, -1]
            retained_center_y_pred = cpt_preds[
                i].reshape(batch_size, -1, h, w)[
                    img_inds, class_inds, -1, grid_cpts_y_inds]
                
            retained_center_x_pred[0, :] = torch.sigmoid(
                retained_center_x_pred[0, :])
            retained_center_x_pred = retained_center_x_pred.view(-1)
            retained_center_y_pred[0, :] = torch.sigmoid(
                retained_center_y_pred[0, :])
            retained_center_y_pred = retained_center_y_pred.view(-1)

            # 4. Calculate loss
            # cpt loss
            loss_cpt_x = nn.MSELoss(size_average=False)(
                retained_center_x_pred, cpts_target_x)/2.0
            loss_cpt_y = nn.MSELoss(size_average=False)(
                retained_center_y_pred, cpts_target_y)/2.0
            loss_cpt += (loss_cpt_x+loss_cpt_y)

            # conf loss
            retained_center_pred = torch.stack((retained_center_x_pred,
                                             retained_center_y_pred), dim=1)
            loss_cls += corner_confidences(cpts_target, retained_center_pred)

            # cls loss
            if self.num_classes > 1:
                pass
            else:
                loss_cls += cls_scores[i].sum() * 0

        return dict(
            loss_cls = loss_cls * batch_size,
            loss_conf = loss_conf * batch_size,
            loss_cpt = loss_cpt * batch_size)

    def _convert_gt_to_norm_format(self,
                                   batch_gt_instances: Sequence[InstanceData],
                                   batch_img_metas: Sequence[dict]) -> Tensor:
        if isinstance(batch_gt_instances, torch.Tensor):
            # fast version
            # 对控制点坐标进行归一化处理
            img_shape = batch_img_metas[0]['batch_input_shape']
            gt_cpts_xy = batch_gt_instances[:, 1:self.num_cpts]
            gt_cpts_xy[:, 0::2] /= img_shape[1]
            gt_cpts_xy[:, 1::2] /= img_shape[0]
            batch_gt_instances[:, 1:self.num_cpts] = gt_cpts_xy

            # batch_targets_normed = batch_gt_instances.repeat(
            #     1, 1, 1)
            batch_targets_normed = batch_gt_instances
        else:
            batch_target_list = []
            # convert xyxy bbox to yolo format
            for i, gt_instances in enumerate(batch_gt_instances):
                img_shape = batch_img_metas[i]['batch_input_shape']
                # todo: 修改json文件的bbox和label名称
                bboxes = gt_instances.bboxes
                labels = gt_instances.labels
                control_points = gt_instances.control_points

                xy1, xy2 = bboxes.split((2,2), dim=-1)
                bboxes = torch.cat([(xy1+xy2)/2, (xy2-xy1)], dim=-1)
                bboxes[:, 1::2] /= img_shape[0]
                bboxes[:, 0::2] /= img_shape[1]

                control_points[:, 0] /= img_shape[0]
                control_points[:, 1] /= img_shape[1]

                index = bboxes.new_full((len(bboxes), 1), i)

                # (batch_idx, label, normed_bbox, normed_control_points)
                target = torch.cat((index, labels[:, None].float(), bboxes,
                                    control_points), dim=1)
                batch_target_list.append(target)

            # (num_base_priors, num_bboxes, 6+18)
            batch_targets_normed = torch.cat(
                batch_target_list, dim=0).repeat(self.num_base_priors, 1, 1)

        return batch_targets_normed
