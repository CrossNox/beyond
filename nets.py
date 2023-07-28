from .blocks import BasicBlock, BasicBlockWithDeathRate, Bottleneck
from .DenseResNet import DenseResNet
from .MResNet import MResNet
from .MResNetC import MResNetC
from .ResNet import ResNet
from .ResNet_N import ResNet_N


# MResNet
def MResNet20(**kwargs):
    return MResNet(BasicBlock, [3, 3, 3], **kwargs)


def MResNet44(**kwargs):
    return MResNet(BasicBlock, [7, 7, 7], **kwargs)


def MResNet56(**kwargs):
    return MResNet(BasicBlock, [9, 9, 9], **kwargs)


def MResNet110(**kwargs):
    return MResNet(BasicBlock, [18, 18, 18], **kwargs)


def MResNet164(**kwargs):
    return MResNet(Bottleneck, [18, 18, 18], **kwargs)


# ResNet
def ResNet_20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)


# ResNet N
def ResNet_N20(**kwargs):
    return ResNet_N(BasicBlock, [3, 3, 3], **kwargs)


def ResNet_N110(**kwargs):
    return ResNet_N(BasicBlock, [18, 18, 18], **kwargs)


# MResNet SD
def MResNetSD20(**kwargs):
    return MResNet(BasicBlockWithDeathRate, [3, 3, 3], stochastic_depth=True, **kwargs)


def MResNetSD110(**kwargs):
    return MResNet(
        BasicBlockWithDeathRate, [18, 18, 18], stochastic_depth=True, **kwargs
    )


# MResNet C
def MResNetC20(**kwargs):
    return MResNetC(BasicBlock, [3, 3, 3], **kwargs)


def MResNetC32(**kwargs):
    return MResNetC(BasicBlock, [5, 5, 5], **kwargs)


def MResNetC44(**kwargs):
    return MResNetC(BasicBlock, [7, 7, 7], **kwargs)


def MResNetC56(**kwargs):
    return MResNetC(BasicBlock, [9, 9, 9], **kwargs)


# DenseNet
def DenseResNet20(**kwargs):
    return DenseResNet(BasicBlock, [3, 3, 3], **kwargs)


def DenseResNet110(**kwargs):
    return DenseResNet(BasicBlock, [18, 18, 18], **kwargs)
