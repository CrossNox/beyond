import math

from torch import nn

from .blocks import DenseBlock

# from init import *


class DenseResNet(nn.Module):
    def __init__(self, block, layers, pretrain=False, num_classes=100):
        self.in_planes = 16
        self.planes = [16, 32, 64]
        self.strides = [1, 2, 2]
        super().__init__()
        self.block = block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pretrain = pretrain

        self.denseblock1 = DenseBlock(self.block, layers[0], 16, self.planes[0], 1)

        self.denseblock2 = DenseBlock(
            self.block, layers[1], self.planes[0], self.planes[1], 2
        )

        self.denseblock3 = DenseBlock(
            self.block, layers[2], self.planes[1], self.planes[2], 2
        )

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

        x = self.denseblock1(x)
        x = self.denseblock2(x)
        x = self.denseblock3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
