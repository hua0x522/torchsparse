from torch import nn
from torchsparse import nn as spnn
from core.models.modules.layers_3d import SparseConvBlock, SparseResBlock
import core.models.modules.wrapper as wrapper
from .backbone3d_template import Backbone3DTemplate
import time
import torch

__all__ = ['SparseResNet']


class SparseResNet(Backbone3DTemplate):
    def __init__(self, in_channels: int = 4, **kwargs) -> None:
        super().__init__()

        self.num_channels = num_channels = [16, 32, 64, 128]
        self.in_channels = in_channels
        self.out_channels = num_channels[-1]
        self.backend = backend = kwargs.get('backend', 'torchsparse')
        self.kmap_mode = kwargs.get('kmap_mode', 'hashmap')
        self.wrapper = wrapper.Wrapper(backend=self.backend)

        self.stem = nn.Sequential(
            SparseConvBlock(in_channels, num_channels[0], 3, stride=1, padding=1, indice_key='subm1', backend=backend, kmap_mode=self.kmap_mode),
            SparseResBlock(num_channels[0], num_channels[0], 3, indice_key='res1', backend=backend, kmap_mode=self.kmap_mode),
            SparseResBlock(num_channels[0], num_channels[0], 3, indice_key='res1', backend=backend, kmap_mode=self.kmap_mode)
        )
        self.stage1 = nn.Sequential(
            SparseConvBlock(num_channels[0], num_channels[1], 3, stride=2, padding=1, indice_key='spconv2', backend=backend, kmap_mode=self.kmap_mode),
            SparseResBlock(num_channels[1], num_channels[1], 3, indice_key='res2', backend=backend, kmap_mode=self.kmap_mode),
            SparseResBlock(num_channels[1], num_channels[1], 3, indice_key='res2', backend=backend, kmap_mode=self.kmap_mode),
        )
        self.stage2 = nn.Sequential(
            SparseConvBlock(num_channels[1], num_channels[2], 3, stride=2, padding=1, indice_key='spconv3', backend=backend, kmap_mode=self.kmap_mode),
            SparseResBlock(num_channels[2], num_channels[2], 3, indice_key='res3', backend=backend, kmap_mode=self.kmap_mode),
            SparseResBlock(num_channels[2], num_channels[2], 3, indice_key='res3', backend=backend, kmap_mode=self.kmap_mode),
        )
        self.stage3 = nn.Sequential(
            SparseConvBlock(num_channels[2], num_channels[3], 3, stride=2, padding=(1, 0, 1), indice_key='spconv4', backend=backend, kmap_mode=self.kmap_mode),
            SparseResBlock(num_channels[3], num_channels[3], 3, indice_key='res4', backend=backend, kmap_mode=self.kmap_mode),
            SparseResBlock(num_channels[3], num_channels[3], 3, indice_key='res4', backend=backend, kmap_mode=self.kmap_mode),
        )

        self.stage4 = SparseConvBlock(
            num_channels[3], num_channels[3], [1, 3, 1], stride=[1, 2, 1], indice_key='spconv_down2', backend=backend, kmap_mode=self.kmap_mode
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        #print(x0.indices.shape, x1.indices.shape, x2.indices.shape, x3.indices.shape, x4.indices.shape)
        return [x0, x1, x2, x3, x4]
