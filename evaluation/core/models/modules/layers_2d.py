import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ConvBlock', 'ConvTransposeBlock', 'SCConvBlock', 'ResBlock']

class SCConv2d(nn.Module):
    """
    Jiang-Jiang Liu, Qibin Hou, Ming-Ming Cheng, Changhu Wang and Jiashi Feng,
    Improving Convolutional Networks with Self-Calibrated Convolutions.
    In CVPR 2020.

    The same implementation as paper but slightly different from official code.
    
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False) -> None:
        super().__init__()
        paths = []
        for i in range(4):
            paths.append(
                nn.Conv2d(
                    in_channels // 2,
                    out_channels // 2, 
                    kernel_size=kernel_size,
                    stride=1 if i not in [0, 3] else stride,
                    padding=(kernel_size-1)//2,
                    bias=False
                )
            )
        self.paths = nn.ModuleList(paths)
        self.pool = nn.AvgPool2d(4, 4)

    def forward(self, inputs):
        B, C, H, W = inputs.size()
        x1, x2 = inputs[:, :C//2, :, :], inputs[:, C//2:, :, :]
        # B, C//2, H, W
        y2 = self.paths[0](x2)
        # B, C//2, H, W
        y1_d = F.interpolate(
            self.paths[1](self.pool(x1)),
            (H, W)
        )
        y1_d = torch.sigmoid(x1 + y1_d)
        # B, C//2, H, W
        y1 = self.paths[2](x1)
        # B, C//2, H, W
        y1 = y1 * y1_d
        # B, C//2, H, W
        y1 = self.paths[3](y1)
        # B, C, H, W
        y = torch.cat([y1, y2], dim=1)
        return y


class ConvBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False,
                 momentum: float = 0.1,
                 eps: float = 1e-05) -> None:
        super().__init__(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(out_channels,
                      eps=eps,
                      momentum=momentum),
            nn.ReLU(True),
        )


class SCConvBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False,
                 momentum: float = 0.1,
                 eps: float = 1e-05) -> None:
        super().__init__(
            SCConv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(out_channels,
                      eps=eps,
                      momentum=momentum),
            nn.ReLU(True),
        )


class ConvTransposeBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1) -> None:
        super().__init__(
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels,
                      in_channels,
                      kernel_size,
                      stride=stride,
                      padding=kernel_size // 2,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      padding=kernel_size // 2,
                      dilation=dilation,
                      bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels or stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

