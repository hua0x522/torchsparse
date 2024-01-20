import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ... import criterions
from ...utils.iou3d import iou3d_utils
from ..backbone_3d import SparseResNet
from ..backbone_2d import DenseRPNHead, ToBEVConvolutionBlock
from ..heads import AnchorHead, AnchorHeadMulti
from .single_stage_detector import AnchorBasedSingleStageDetector

from typing import Optional, Union, List, Tuple


__all__ = ['SECOND']


class SECOND(AnchorBasedSingleStageDetector):
    def __init__(self,
                 classes: List[str],
                 voxel_size: Union[np.ndarray, List[float]],
                 pc_area_scope: Union[np.ndarray, List[List[float]]],
                 mean_center_z: dict,
                 mean_size: dict,
                 last_kernel_size: int,
                 num_dir_bins: int,
                 cls_weight: float,
                 reg_weight: float,
                 dir_weight: float,
                 pos_cls_weight: float,
                 neg_cls_weight: float,
                 proposal_stage: int,
                 negative_anchor_threshold: List[float],
                 positive_anchor_threshold: List[float], 
                 max_number_of_voxels: Optional[Union[List[int], int]] = None,
                 max_points_per_voxel: Optional[Union[List[int], int]] = None,
                 sep_heads: Optional[bool] = False,
                 use_multihead: Optional[bool] = False,
                 reg_config: Optional[dict] = None,
                 rpn_head_configs: Optional[dict] = None,
                 backbone_2d_configs: Optional[dict] = None,
                 nms_configs: Optional[dict] = None,
                 **kwargs
                 ) -> None:
        super().__init__(
            classes=classes, voxel_size=voxel_size, pc_area_scope=pc_area_scope,
            mean_center_z=mean_center_z, mean_size=mean_size, last_kernel_size=last_kernel_size,
            num_dir_bins=num_dir_bins, cls_weight=cls_weight, reg_weight=reg_weight,
            dir_weight=dir_weight, pos_cls_weight=pos_cls_weight, neg_cls_weight=neg_cls_weight,
            proposal_stage=proposal_stage, negative_anchor_threshold=negative_anchor_threshold,
            positive_anchor_threshold=positive_anchor_threshold, max_number_of_voxels=max_number_of_voxels,
            sep_heads=sep_heads, use_multihead=use_multihead, reg_config=reg_config, 
            rpn_head_configs=rpn_head_configs, backbone_2d_configs=backbone_2d_configs, 
            nms_configs=nms_configs, **kwargs
        )
        
        self.encoder = SparseResNet(
            in_channels=kwargs.get('in_channels', 4)
        )
        
        self.to_bev = ToBEVConvolutionBlock(
            self.encoder.out_channels,
            128, 
            self.loc_min,
            self.loc_max,
            proposal_stride=[8, 16, 8]
        )

        self.rpn = DenseRPNHead(
            classes=self.classes,
            in_channels=256,
            loc_min=None,
            loc_max=None,
            proposal_stride=None,
            need_tobev=False,
            sep_heads=sep_heads,
            **backbone_2d_configs
        )
