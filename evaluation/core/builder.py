from typing import Callable

import numpy as np
import torch
import torch.optim
from torch import nn
from utils.config import configs

__all__ = [
    'make_dataset', 'make_model'
]

def make_dataset(n_sample=-1):
    if configs.dataset.name == 'kitti':
        from .datasets import KITTI
        dataset = KITTI(root=configs.dataset.root,
                        max_points=configs.dataset.max_points)
    elif configs.dataset.name == 'nuscenes':
        from core.datasets import NuScenes
        voxel_size = configs.dataset.get(
            'voxel_size', [0.1, 0.2, 0.1]
        )
        pc_area_scope = configs.dataset.get(
            'pc_area_scope', [[-51.2, 51.2], [-3, 5], [-51.2, 51.2]]
        )
        rotation_bound = configs.dataset.get(
            'rotation_bound', [-0.3925, 0.3925]
        )
        scale_bound = configs.dataset.get(
            'scale_bound', [0.95, 1.05]
        )
        translation_std = configs.dataset.get(
            'translation_std', 0.5
        )
        cbgs = configs.dataset.get('cbgs', False)
        dataset = NuScenes(
            configs.dataset.root,
            max_sweeps=configs.dataset.max_sweeps,
            train_max_points=configs.dataset.train_max_points,
            max_points=configs.dataset.val_max_points,
            val_max_points=configs.dataset.val_max_points,
            voxel_size=voxel_size,
            pc_area_scope=pc_area_scope,
            rotation_bound=rotation_bound,
            scale_bound=scale_bound,
            translation_std=translation_std,
            cbgs=cbgs
        )
        if n_sample > 0 and n_sample < len(dataset):
            dataset.infos = dataset.infos[:n_sample]
    elif configs.dataset.name == 'waymo':
        from core.datasets import Waymo
        voxel_size = configs.dataset.get(
            'voxel_size', [0.1, 0.15, 0.1]
        )
        pc_area_scope = configs.dataset.get(
            'pc_area_scope', [[-75.2, 75.2], [-4, 2], [-75.2, 75.2]]
        )
        rotation_bound = configs.dataset.get(
            'rotation_bound', [-0.78539816, 0.78539816]
        )
        scale_bound = configs.dataset.get(
            'scale_bound', [0.95, 1.05]
        )
        translation_std = configs.dataset.get(
            'translation_std', 0.0
        )
        sample_stride = configs.dataset.get(
            'sample_stride', 1
        )
        dataset = Waymo(
            configs.dataset.root,
            max_sweeps=configs.dataset.max_sweeps,
            train_max_points=configs.dataset.train_max_points,
            val_max_points=configs.dataset.val_max_points,
            max_points=configs.dataset.val_max_points,
            voxel_size=voxel_size,
            pc_area_scope=pc_area_scope,
            rotation_bound=rotation_bound,
            scale_bound=scale_bound,
            translation_std=translation_std,
            sample_stride=sample_stride
        )
        if n_sample > 0 and n_sample < len(dataset):
            dataset.infos = dataset.infos[:n_sample]
    elif configs.dataset.name == 'nuscenes-lidarseg':
        from core.datasets import NuScenesLiDARSeg
        dataset = NuScenesLiDARSeg(
            root_dir=configs.dataset.root,
            max_points=configs.dataset.max_points,
            max_sweeps=configs.dataset.get('max_sweeps', 1)
        )
        if n_sample > 0 and n_sample < len(dataset):
            dataset.infos = dataset.infos[:n_sample]
    elif configs.dataset.name == 'semantic-kitti':
        from core.datasets import SemanticKITTI
        dataset = SemanticKITTI(
            root_dir=configs.dataset.root,
            max_points=configs.dataset.max_points
        )
        if n_sample > 0 and n_sample < len(dataset):
            dataset.fns = dataset.fns[:n_sample]
    else:
        raise NotImplementedError(configs.dataset.name)
    return dataset


# here dataset is a class!
def make_model(dataset) -> nn.Module:
    backend = configs.model.get('backend', 'torchsparse')
    kmap_mode = configs.model.get('kmap_mode', 'hashmap')
    if configs.model.name == 'minkunet':
        from .models.segmentation_models import MinkUNet
        param_dict = dict(backend=backend)
        param_dict.update(in_channels=configs.model.get('in_channels', 4))
        param_dict.update(cr=configs.model.get('cr', 1.0))
        param_dict.update(num_classes=configs.dataset.num_classes)
        model = MinkUNet(**param_dict)
        return model
        
    elif configs.model.name == 'second' or configs.model.name == 'pointpillars' or configs.model.name == 'centerpoint':
        from .models.detectors import SECOND, CenterPoint
        param_dict = dict(backend=backend, kmap_mode=kmap_mode)
        param_dict.update(in_channels = configs.model.get('in_channels', 4))
        param_dict.update(classes = dataset.classes)
        param_dict.update(pc_area_scope = configs.model.get('pc_area_scope', dataset.pc_area_scope))
        param_dict.update(last_kernel_size = configs.model.get('last_kernel_size', 1))
        param_dict.update(num_dir_bins = configs.model.get('num_dir_bins', None))
        param_dict.update(cls_weight = configs.model.cls_weight)
        param_dict.update(reg_weight = configs.model.reg_weight)
        param_dict.update(dir_weight = configs.model.dir_weight)
        param_dict.update(use_multihead = configs.model.get('use_multihead', False))
        param_dict.update(use_direction_classifier = configs.model.get('use_direction_classifier', False))
        param_dict.update(use_iou_head = configs.model.get('use_iou_head', False))
        param_dict.update(code_cfg = configs.model.get('box_code_cfg', 
            {'code_size': 7, 'encode_angle_by_sincos': False}
        ))
        param_dict.update(loss_cfg = configs.model.get('loss_cfg', None))
        param_dict.update(nms_configs = configs.model.get('nms_configs', None))
        param_dict.update(reg_config = configs.model.get('reg_config', None))
        param_dict.update(rpn_head_configs = configs.model.get('rpn_head_configs', None))
        param_dict.update(backbone_2d_configs = configs.model.get('backbone_2d_configs', None))

        if 'voxelization_cfg' not in configs.model:
            param_dict.update(voxel_size = dataset.voxel_size)
            param_dict.update(max_number_of_voxels = None)
            param_dict.update(max_points_per_voxel = None)
        else:
            param_dict.update(voxel_size = np.array(configs.model.voxelization_cfg.voxel_size, 'float'))
            # pointpillars
            param_dict.update(input_voxel_size = dataset.voxel_size)
            param_dict.update(max_number_of_voxels = configs.model.voxelization_cfg.max_number_of_voxels)
            param_dict.update(max_points_per_voxel = configs.model.voxelization_cfg.max_points_per_voxel)

        if configs.model.name in ['second', 'pointpillars']:    
            param_dict.update(proposal_stage = configs.model.proposal_stage)        
            param_dict.update(mean_center_z = dataset.mean_center_z)
            param_dict.update(mean_size = dataset.mean_size)
            
            
            param_dict.update(pos_cls_weight = configs.model.get('pos_cls_weight', 1.0))
            param_dict.update(neg_cls_weight = configs.model.get('neg_cls_weight', 1.0))
            
            param_dict.update(negative_anchor_threshold = configs.model.negative_anchor_threshold)
            param_dict.update(positive_anchor_threshold = configs.model.positive_anchor_threshold)
                
        if configs.model.name == 'second':
            detector = SECOND
        elif configs.model.name == 'centerpoint':
            detector = CenterPoint
        else:
            detector = PointPillars
        
        model = detector(**param_dict)

        if 'bn' in configs.model:
            for name, module in model.named_modules():
                if hasattr(module, 'reset_bn_params'):
                    module.reset_bn_params(
                        momentum=configs.model.bn.momentum,
                        eps=configs.model.bn.eps
                    )
        
    else:
        raise NotImplementedError(configs.model.name)
    return model
