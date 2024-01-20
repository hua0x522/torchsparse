import numpy as np
import torch
from torch import nn
#import torch_scatter
#from torchsparse import nn as spnn
from core.models.modules.voxelization.voxelize import Voxelization
from .backbone3d_template import Backbone3DTemplate
from typing import List, Optional, Tuple, Union



__all__ = ['PillarsEncoderHard']


class PillarsEncoderHard(Backbone3DTemplate):
    def __init__(self, 
                 in_channels: Optional[int] = 4,
                 out_channels: Optional[int] = 64,
                 pc_area_scope: Optional[Union[List[List[float]], np.ndarray]] = None,
                 voxel_size: Optional[Union[List[float], Tuple[float, ...], np.ndarray]] = [0.16, 4, 0.16],
                 bev_dim: Optional[Union[List[int], Tuple[int, ...]]] = [0, 2],
                 **kwargs) -> None:
        super().__init__()
        
        # dataset parameters
        assert pc_area_scope is not None
        input_voxel_size = kwargs.get('input_voxel_size', None)
        assert input_voxel_size is not None
        
        #voxel_size = np.array(voxel_size)
        # [Important] we need to make sure that BEV size is divisible by 8
        bev_size = (pc_area_scope[:, 1] - pc_area_scope[:, 0]) / voxel_size
        bev_size = np.floor(bev_size).astype(np.int32)
        self.bev_size = bev_size[bev_dim].astype(np.int64)
        self.bev_dim = bev_dim

        self.in_channels = in_channels + 6
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv1d(self.in_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True)
        )
        
        self.voxelization = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=pc_area_scope.transpose().reshape(-1),
            max_num_points=kwargs['max_points_per_voxel'],
            max_voxels=kwargs['max_number_of_voxels']
        )
        
        self.voxel_size = nn.Parameter(
            torch.from_numpy(voxel_size.reshape(1, -1)).float(),
            requires_grad=False
        )
        
        self.input_voxel_size = nn.Parameter(
            torch.from_numpy(input_voxel_size.reshape(1, -1)).float(),
            requires_grad=False
        )
        
        self.pc_area_min = nn.Parameter(
            torch.from_numpy(pc_area_scope[:, 0].reshape(1, -1)).float(),
            requires_grad=False
        )

    
    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator
    
    def forward(self, x):
        _coords, features = x.C, x.F
        coords = _coords[:, :3].float() * self.input_voxel_size
        features_ls = [features]
        # N x 2
        coords_f = coords[:, :3].float()
        coords_f /= self.voxel_size
        # N x 2
        coords_i = torch.round(coords_f)
        coords_i = torch.cat([_coords[:, -1].unsqueeze(-1), coords_i], -1).long()
        coords_i[:, 1] = torch.clamp(coords_i[:, 1], 0, self.bev_size[0]-1)
        coords_i[:, 2] = torch.clamp(coords_i[:, 2], 0, self.bev_size[1]-1)
        unq, unq_inv = torch.unique(
            coords_i, return_inverse=True, return_counts=False, dim=0
        )
                
        # voxels: M x 32 x 4
        voxel_features = []
        voxel_num_points = []
        coords = []
        
        with torch.no_grad():
            for idx in range(coords_i[:, 0].max().int().item() + 1):
                _voxel_features, _coords, _voxel_num_points = self.voxelization(features[coords_i[:, 0] == idx])
                voxel_features.append(_voxel_features)
                voxel_num_points.append(_voxel_num_points)
                coords.append(
                    torch.cat([_coords, torch.zeros_like(_coords[:, 0:1]) + idx], 1)
                )
        
            voxel_features = torch.cat(voxel_features, 0)
            voxel_num_points = torch.cat(voxel_num_points, 0)
            coords = torch.cat(coords, 0)
        
            points_mean = voxel_features[:, :, :3].sum(
                dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            f_cluster = voxel_features[:, :, :3] - points_mean
        
            f_center = torch.zeros_like(voxel_features)
            f_center = voxel_features[..., :3] - (coords[..., [2,1,0]] * self.voxel_size).unsqueeze(1)
            f_center -= self.voxel_size / 2. + self.pc_area_min
        
            features = [voxel_features, f_cluster, f_center]
            features = torch.cat(features, dim=-1)

            voxel_count = features.shape[1]
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
            features *= mask
            #print(features[..., -3:].min(0).values.min(0).values, features[..., -3:].max(0).values.max(0).values)
        
            # M x 10 x 32
            features = features.permute(0, 2, 1).contiguous()
        # M x OC
        features = self.net(features).max(-1).values

        # map to bev
        batch_size = int(coords_i[:, 0].max().item()) + 1
        out_data_shape = [batch_size] + self.bev_size.tolist() + [features.size(-1)]
        bev_features = torch.zeros(*out_data_shape, device=features.device).float()        
        coords = coords.long()
        bev_features[coords[:, -1], coords[:, 2], coords[:, 0], :] = features   
        
        # B x C x H x W
        bev_features = bev_features.permute(0,3,1,2).contiguous()
        return [bev_features]

