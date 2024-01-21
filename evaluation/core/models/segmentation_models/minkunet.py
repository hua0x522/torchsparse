import torchsparse
import torch.nn as nn
import torchsparse.nn as spnn
import core.models.modules.wrapper as wrapper
from core.models.modules.layers_3d import SparseConvBlock, SparseDeConvBlock, SparseResBlock
from torchsparse.utils import make_ntuple


class MinkUNet(nn.Module):
    def __init__(self, backend='torchsparse', **kwargs):
        super().__init__()
        ic = kwargs.get('in_channels', 4)
        cr = kwargs.get('cr', 1.0)
        cs = [64, 64, 64, 128, 256, 256, 128, 64, 64]
        cs = [int(cr * x) for x in cs]
        # make sure #channels is even
        for i, x in enumerate(cs):
            if x % 2 != 0:
                cs[i] = x + 1

        self.backend = backend
        self.wrapper = wrapper.Wrapper(backend=self.backend)
        
        self.stem = self.wrapper.sequential(
            self.wrapper.conv3d(ic, cs[0], kernel_size=3, stride=1, indice_key="pre"),
            self.wrapper.bn(cs[0]), self.wrapper.relu(True),
            self.wrapper.conv3d(cs[0], cs[0], kernel_size=3, stride=1, indice_key="pre"),
            self.wrapper.bn(cs[0]), self.wrapper.relu(True))

        self.stage1 = nn.Sequential(
            SparseConvBlock(cs[0], cs[0], kernel_size=2, stride=2, dilation=1, indice_key="down1", backend=self.backend),
            SparseResBlock(cs[0], cs[1], kernel_size=3, stride=1, dilation=1, indice_key="res1", backend=self.backend),
            SparseResBlock(cs[1], cs[1], kernel_size=3, stride=1, dilation=1, indice_key="res1", backend=self.backend),
        )

        self.stage2 = nn.Sequential(
            SparseConvBlock(cs[1], cs[1], kernel_size=2, stride=2, dilation=1, indice_key="down2", backend=self.backend),
            SparseResBlock(cs[1], cs[2], kernel_size=3, stride=1, dilation=1, indice_key="res2", backend=self.backend),
            SparseResBlock(cs[2], cs[2], kernel_size=3, stride=1, dilation=1, indice_key="res2", backend=self.backend))

        self.stage3 = nn.Sequential(
            SparseConvBlock(cs[2], cs[2], kernel_size=2, stride=2, dilation=1, indice_key="down3", backend=self.backend),
            SparseResBlock(cs[2], cs[3], kernel_size=3, stride=1, dilation=1, indice_key="res3", backend=self.backend),
            SparseResBlock(cs[3], cs[3], kernel_size=3, stride=1, dilation=1, indice_key="res3", backend=self.backend),
        )

        self.stage4 = nn.Sequential(
            SparseConvBlock(cs[3], cs[3], kernel_size=2, stride=2, dilation=1, indice_key="down4", backend=self.backend),
            SparseResBlock(cs[3], cs[4], kernel_size=3, stride=1, dilation=1, indice_key="res4", backend=self.backend),
            SparseResBlock(cs[4], cs[4], kernel_size=3, stride=1, dilation=1, indice_key="res4", backend=self.backend),
        )

        self.up1 = nn.ModuleList([
            SparseDeConvBlock(cs[4], cs[5], kernel_size=2, stride=2, indice_key="down4", backend=self.backend),
            nn.Sequential(
                SparseResBlock(cs[5] + cs[3], cs[5], kernel_size=3, stride=1,
                              dilation=1, indice_key="res3", backend=self.backend),
                SparseResBlock(cs[5], cs[5], kernel_size=3, stride=1, dilation=1, indice_key="res3", backend=self.backend),
            )
        ])

        self.up2 = nn.ModuleList([
            SparseDeConvBlock(cs[5], cs[6], kernel_size=2, stride=2, indice_key="down3", backend=self.backend),
            nn.Sequential(
                SparseResBlock(cs[6] + cs[2], cs[6], kernel_size=3, stride=1,
                              dilation=1, indice_key="res2", backend=self.backend),
                SparseResBlock(cs[6], cs[6], kernel_size=3, stride=1, dilation=1, indice_key="res2", backend=self.backend),
            )
        ])

        self.up3 = nn.ModuleList([
            SparseDeConvBlock(cs[6], cs[7], kernel_size=2, stride=2, indice_key="down2", backend=self.backend),
            nn.Sequential(
                SparseResBlock(cs[7] + cs[1], cs[7], kernel_size=3, stride=1,
                              dilation=1, indice_key="res1", backend=self.backend),
                SparseResBlock(cs[7], cs[7], kernel_size=3, stride=1, dilation=1, indice_key="res1", backend=self.backend),
            )
        ])

        self.up4 = nn.ModuleList([
            SparseDeConvBlock(cs[7], cs[8], kernel_size=2, stride=2, indice_key="down1", backend=self.backend),
            nn.Sequential(
                SparseResBlock(cs[8] + cs[0], cs[8], kernel_size=3, stride=1,
                              dilation=1, indice_key="pre", backend=self.backend),
                SparseResBlock(cs[8], cs[8], kernel_size=3, stride=1, dilation=1, indice_key="pre", backend=self.backend),
            )
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[8],
                                                  kwargs['num_classes']))

        self.weight_initialization()
        #self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = x['pts_input']
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        
        y1 = self.up1[0](x4)
        y1 = self.wrapper.cat([y1, x3])
        y1 = self.up1[1](y1)

        # print_shape("x0", x0, self.backend)
        # print_shape("x1", x1, self.backend)
        # print_shape("x2", x2, self.backend)
        # print_shape("x3", x3, self.backend)
        # print_shape("x4", x4, self.backend)
        # print_shape("y1", y1, self.backend)
        
        y2 = self.up2[0](y1)
        y2 = self.wrapper.cat([y2, x2])
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = self.wrapper.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = self.wrapper.cat([y4, x0])
        y4 = self.up4[1](y4)

        # if self.backend == 'ME':
        #     print('y4')
        #     print(y4.coordinate_manager)

        # if self.backend == 'torchsparse':
        #     print('y4')
        #     kmap = y4.kmaps.get((y4.stride, make_ntuple(3, ndim=3), make_ntuple(1, ndim=3), \
        #         make_ntuple(1, ndim=3)))
        #     print(sum(kmap[1].tolist()))


        if self.backend != 'spconv':
            out = self.classifier(y4.F)
        else:
            out = self.classifier(y4.features)

        return out


def print_shape(msg, x, backend):
    if backend == 'torchsparse':
        print(msg, x.F.shape)
    else:
        print(msg, x.features.shape)
