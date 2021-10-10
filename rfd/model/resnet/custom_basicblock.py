# -*- coding: utf-8 -*-

"""
@date: 2021/8/29 下午2:36
@file: custom_basicblock.py
@author: zj
@description: 
"""

from torch import Tensor
import torch.nn as nn
from torchvision.models.resnet import BasicBlock


class CustomBasicBlock(nn.Module):

    def __init__(self, basicblock):
        super().__init__()
        assert isinstance(basicblock, BasicBlock)

        self.model = basicblock
        self.bn3 = basicblock.bn2

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.model.conv1(x)
        out = self.model.bn1(out)
        out = self.model.relu(out)

        out = self.model.conv2(out)
        out = self.model.bn2(out)

        if self.model.downsample is not None:
            identity = self.model.downsample(x)

        out += identity
        # remove last relu operation
        # out = self.relu(out)

        return out
