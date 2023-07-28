import math

import torch
from torch import nn
from torch.autograd import Variable

from .blocks import Downsample

# from init import *


class MResNetC(nn.Module):
    def __init__(self, block, layers, pretrain=False, num_classes=100):
        super().__init__()
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.strides = [1, 2, 2]
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain
        ks = []
        for i in range(3):
            ks += [
                nn.Parameter(
                    torch.Tensor(self.planes[i] * block.expansion, 1, 1).uniform_(
                        -0.1, -0.0
                    )
                )
                for j in range(layers[i])
            ]
        self.ks = nn.ParameterList(ks)

        blocks = []
        for i in range(3):
            blocks.append(block(self.in_planes, self.planes[i], self.strides[i]))
            self.in_planes = self.planes[i] * block.expansion
            for j in range(1, layers[i]):
                blocks.append(block(self.in_planes, self.planes[i]))
        self.blocks = nn.ModuleList(blocks)
        self.downsample1 = Downsample(16, 64, stride=1)
        # self.downsample1=nn.Conv2d(16, 64,
        #                    kernel_size=1, stride=1, bias=False)
        self.downsample21 = Downsample(16 * block.expansion, 32 * block.expansion)
        # self.downsample22=Downsample(16*block.expansion,32*block.expansion)
        self.downsample31 = Downsample(32 * block.expansion, 64 * block.expansion)
        # self.downsample32=Downsample(32*block.expansion,64*block.expansion)

        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def change_state(self):
        self.pretrain = not self.pretrain

    def forward(self, x):
        x = self.conv1(x)
        # x=self.bn1(x)
        # x=self.relu(x)

        if self.block.expansion == 4:
            residual = self.downsample1(x)
        else:
            residual = x
        x = self.blocks[0](x) + residual
        last_residual = residual
        if self.training and self.noise:
            x += Variable(
                torch.FloatTensor(x.size()).cuda().uniform_(0, self.noise_level),
                requires_grad=False,
            )
        for i, b in enumerate(self.blocks):
            if i == 0:
                continue
            residual = x

            if b.in_planes != b.planes * b.expansion:
                if b.planes == 32:
                    residual = self.downsample21(x)
                    # if not self.pretrain:
                    # last_residual=self.downsample22(last_residual)
                elif b.planes == 64:
                    residual = self.downsample31(x)
                    # if not self.pretrain:
                    # last_residual=self.downsample32(last_residual)
                x = b(x) + residual
            elif self.pretrain:
                x = b(x) + residual
            else:
                x = (
                    b(x)
                    + self.ks[i].expand_as(residual) * residual
                    + (1 - self.ks[i]).expand_as(last_residual) * last_residual
                )
            last_residual = residual
            if self.training and self.noise:
                x += Variable(
                    torch.FloatTensor(x.size()).cuda().uniform_(0, self.noise_level),
                    requires_grad=False,
                )
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
