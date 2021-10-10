# -*- coding: utf-8 -*-

"""
@date: 2021/7/20 下午10:20
@file: resnet.py
@author: zj
@description: 
"""

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models

from zcls.config.key_word import KEY_OUTPUT

from .custom_basicblock import CustomBasicBlock
from .custom_bottlneck import CustomBottleneck
from rfd.config.key_word import KEY_FEAT


class ResNet(nn.Module):

    def __init__(self, num_classes=1000, arch='resnet50'):
        super().__init__()

        self.arch = arch
        assert arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101',
                        'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']
        self.model = eval(f'models.{arch}')(pretrained=True)

        block = CustomBasicBlock if arch in ['resnet18', 'resnet34'] else CustomBottleneck
        self.model.layer1[-1] = block(self.model.layer1[-1])
        self.model.layer2[-1] = block(self.model.layer2[-1])
        self.model.layer3[-1] = block(self.model.layer3[-1])
        self.model.layer4[-2] = block(self.model.layer4[-2])
        self.model.layer4[-1] = block(self.model.layer4[-1])

        self.init_weight(num_classes)

    def init_weight(self, num_classes):
        if num_classes != 1000:
            old_fc = self.model.fc
            assert isinstance(old_fc, nn.Linear)

            in_features = old_fc.in_features
            new_fc = nn.Linear(in_features, num_classes, bias=True)
            nn.init.normal_(new_fc.weight, 0, 0.01)
            nn.init.zeros_(new_fc.bias)

            self.model.fc = new_fc

    def _forward_impl(self, x: Tensor):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        feats_list = list()
        feat1 = self.model.layer1(x)
        x = F.relu(feat1)
        feat2 = self.model.layer2(x)
        x = F.relu(feat2)
        feat3 = self.model.layer3(x)
        x = F.relu(feat3)

        feats_list.append(feat1)
        feats_list.append(feat2)
        feats_list.append(feat3)
        length_layer4 = len(self.model.layer4)
        for i in range(len(self.model.layer4)):
            x = self.model.layer4[i](x)
            if i in [(length_layer4 - 2), (length_layer4 - 1)]:
                feats_list.append(x)

        x = F.relu(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x, feats_list

    def forward(self, x):
        res, feat_list = self._forward_impl(x)

        return {
            KEY_OUTPUT: res,
            KEY_FEAT: feat_list
        }

    def get_distill_channels(self):
        # if self.arch in ['resnet18', 'resnet34']:
        #     channel1 = self.model.layer1[-1].bn2.num_features
        #     channel2 = self.model.layer2[-1].bn2.num_features
        #     channel3 = self.model.layer3[-1].bn2.num_features
        #     channel4 = self.model.layer4[-2].bn2.num_features
        #     channel5 = self.model.layer4[-1].bn2.num_features
        # else:
        channel1 = self.model.layer1[-1].bn3.num_features
        channel2 = self.model.layer2[-1].bn3.num_features
        channel3 = self.model.layer3[-1].bn3.num_features
        channel4 = self.model.layer4[-2].bn3.num_features
        channel5 = self.model.layer4[-1].bn3.num_features

        return [channel1, channel2, channel3, channel4, channel5]


def get_resnet(num_classes=1000, arch='resnet50'):
    return ResNet(num_classes=num_classes, arch=arch)
