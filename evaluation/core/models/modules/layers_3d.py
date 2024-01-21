from torch import nn
from torchsparse import nn as spnn
from typing import Union
import core.models.modules.wrapper as wrapper
import time
import torch


__all__ = ['SparseConvBlock', 'SparseDeConvBlock', 'SparseResBlock']


class SparseConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, list, tuple],
                 stride: Union[int, list, tuple] = 1,
                 dilation: int = 1,
                 padding: Union[int, list, tuple] = 0,
                 indice_key: str = None,
                 backend: str = 'torchsparse',
                 kmap_mode: str = 'hashmap') -> None:
        super().__init__()
        self.backend = backend
        self.wrapper = wrapper.Wrapper(backend=self.backend)
        self.net = self.wrapper.sequential(
            self.wrapper.conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        dilation=dilation,
                        padding=padding,
                        indice_key=indice_key,
                        kmap_mode=kmap_mode),
            self.wrapper.bn(out_channels),
            self.wrapper.relu(True)
        )

    def forward(self, x):
        # torch.cuda.synchronize()
        # st = time.time()
        out = self.net(x)
        # torch.cuda.synchronize()
        # ed = time.time()
        # print(self.backend, (ed-st) * 1000)
        return out



class SparseDeConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, list, tuple],
                 stride: Union[int, list, tuple] = 1,
                 dilation: int = 1,
                 indice_key: str = None,
                 backend: str = 'torchsparse',
                 kmap_mode: str = 'hashmap') -> None:
        super().__init__()
        self.backend = backend
        self.wrapper = wrapper.Wrapper(backend=self.backend)
        self.net = self.wrapper.sequential(
            self.wrapper.conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        dilation=dilation,
                        transpose=True,
                        indice_key=indice_key,
                        kmap_mode=kmap_mode),
            self.wrapper.bn(out_channels),
            self.wrapper.relu(True)
        )

    def forward(self, x):
        return self.net(x)


class SparseResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1,
                 indice_key: str = None,
                 backend: str = 'torchsparse',
                 kmap_mode: str = 'hashmap') -> None:
        super().__init__()
        self.backend = backend
        self.wrapper = wrapper.Wrapper(backend=self.backend)
        if stride != 1 and self.backend == 'spconv':
            raise NotImplementedError

        self.net = self.wrapper.sequential(
            self.wrapper.conv3d(in_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation,
                        stride=stride,
                        indice_key=indice_key,
                        kmap_mode=kmap_mode),
            self.wrapper.bn(out_channels),
            self.wrapper.relu(True),
            self.wrapper.conv3d(out_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation,
                        indice_key=indice_key,
                        kmap_mode=kmap_mode),
            self.wrapper.bn(out_channels)
        )

        if in_channels != out_channels or stride > 1:
            self.downsample = self.wrapper.sequential(
                self.wrapper.conv3d(
                    in_channels, 
                    out_channels, 
                    1, 
                    stride=stride,
                    indice_key=indice_key,
                    kmap_mode=kmap_mode),
                self.wrapper.bn(out_channels),
            )
        else:
            self.downsample = self.wrapper.sequential()

        self.relu = self.wrapper.relu(True)

    def forward(self, x):
        if self.backend != 'spconv':
            # torch.cuda.synchronize()
            # st = time.time()

            x = self.relu(self.net(x) + self.downsample(x))

            # torch.cuda.synchronize()
            # ed = time.time()
            # print(self.backend, (ed-st) * 1000)
            return x
        else:
            # torch.cuda.synchronize()
            # st = time.time()

            x_temp = self.net(x)
            x_temp = x_temp.replace_feature(x_temp.features + self.downsample(x).features)
            out = self.wrapper.sequential(self.relu)(x_temp)

            # torch.cuda.synchronize()
            # ed = time.time()
            # print(self.backend, (ed-st) * 1000)
            return out
