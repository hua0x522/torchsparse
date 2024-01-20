import torch
from torch import nn
import torchsparse
from torchsparse import nn as spnn
from .backbone2d_template import Backbone2DTemplate

__all__ = ['ToBEVConvolutionBlock']


class ToBEVConvolutionBlock(Backbone2DTemplate):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 loc_min: torch.Tensor, 
                 loc_max: torch.Tensor,
                 proposal_stride,
                 backend='torchsparse') -> None:
        super().__init__()

        self.backend = backend
        self.to_bev = nn.Sequential(
            spnn.SparseCrop(coords_min=loc_min, coords_max=loc_max),
            spnn.ToBEVHeightCompression(
                in_channels,
                shape=(loc_max - loc_min) // proposal_stride,
                offset=loc_min
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.backend == 'torchsparse' or self.backend == 'torchsparse-1.4.0':
            return self.to_bev(x)
        elif self.backend == 'ME':
            x = torchsparse.SparseTensor(x.F, x.C[:, [1, 2, 3, 0]], x.tensor_stride)
            return self.to_bev(x)

        else:
            spatial_features = x.dense()
            N, C, H, D, W = spatial_features.shape
            spatial_features = spatial_features.permute(0, 1, 3, 2, 4).contiguous()
            return spatial_features.view(N, C * D, H, W)
