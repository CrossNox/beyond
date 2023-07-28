# from init import *

import torch
from torch import nn
from torch.autograd import Variable


class BasicBlockWithDeathRate(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, death_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes
        self.death_rate = death_rate

    def forward(self, x):
        if not self.training or torch.rand(1)[0] >= self.death_rate:
            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)
            if self.training:
                out /= 1.0 - self.death_rate
        else:
            if self.stride == 1:
                out = Variable(
                    torch.FloatTensor(x.size()).cuda().zero_(), requires_grad=False
                )
            else:
                size = list(x.size())
                size[-1] //= 2
                size[-2] //= 2
                size[-3] *= 2
                size = torch.Size(size)
                out = Variable(
                    torch.FloatTensor(size).cuda().zero_(), requires_grad=False
                )
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            return x + torch.autograd.Variable(
                torch.randn(x.size()).cuda() * self.stddev, requires_grad=False
            )
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.in_planes = in_planes
        self.planes = planes

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        return out


class Downsample(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
        )

    def forward(self, x):
        x = self.downsample(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, block, layers, in_planes, planes, stride=2):
        super().__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.layers = layers
        blocks = []
        blocks.append(block(self.in_planes, self.planes, self.stride))
        for _ in range(1, layers):
            blocks.append(block(self.planes, self.planes))

        self.downsample = None
        if in_planes != planes * block.expansion or stride != 1:
            self.downsample = Downsample(in_planes, planes, stride=stride)

        self.ks = nn.ParameterList(
            [
                nn.Parameter(torch.Tensor(1).uniform_(-0.1, -0.0))
                for i in range(layers * layers)
            ]
        )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        residuals = []
        for i, b in enumerate(self.blocks):
            if i == 0 and self.downsample is not None:
                residuals.append(self.downsample(x))
            else:
                residuals.append(x)

            residual = (self.ks[i * self.layers + i]).expand_as(
                residuals[i]
            ) * residuals[i]
            sumk = self.ks[i * self.layers + i].clone()
            for j in range(i):
                residual += (self.ks[i * self.layers + j]).expand_as(
                    residuals[j]
                ) * residuals[j]
                sumk += self.ks[i * self.layers + j]
            x = residual / sumk.expand_as(residual) + b(x)
        return x
