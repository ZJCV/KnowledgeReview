# -*- coding: utf-8 -*-

"""
@date: 2021/7/20 下午10:20
@file: mobilenet_v2.py
@author: zj
@description: 
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torchvision.models.mobilenetv2 import InvertedResidual, ConvBNActivation, ConvNormActivation

from zcls.config.key_word import KEY_OUTPUT
from rfd.config.key_word import KEY_FEAT


def get_feats_before_relu(module, x):
    if isinstance(module, (ConvBNActivation, ConvNormActivation)):
        # Conv2d
        x = module[0](x)
        # BatchNorm2d
        feats = module[1](x)
        # ReLU6
        x = module[2](feats)
    elif isinstance(module, InvertedResidual):
        inputs = x
        # ConvBNActivation
        x = module.conv[0](x)

        # ConvBNActivation
        module_list = list(module.conv[1])
        x = module_list[0](x)
        feats = module_list[1](x)
        x = module_list[2](feats)

        # Conv2d
        x = module.conv[2](x)
        # BatchNorm2d
        x = module.conv[3](x)

        if module.use_res_connect:
            x = inputs + x
    else:
        raise ValueError('module does not match')

    return feats, x


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, arch='mobilenet_v2'):
        super().__init__()

        assert arch in ['mobilenet_v2']
        self.model = eval(f'models.{arch}')(pretrained=True)

        self.init_weight(num_classes)

    def init_weight(self, num_classes):
        if num_classes != 1000:
            old_fc = self.model.classifier[1]
            assert isinstance(old_fc, nn.Linear)

            in_features = old_fc.in_features
            new_fc = nn.Linear(in_features, num_classes, bias=True)
            nn.init.normal_(new_fc.weight, 0, 0.01)
            nn.init.zeros_(new_fc.bias)

            self.model.classifier[1] = new_fc

    def _forward_impl(self, x: Tensor):
        feature_list = list(self.model.features)

        feats_list = list()
        for i in range(len(feature_list)):
            if i in [3, 6, 13, 17, 18]:
                feats, x = get_feats_before_relu(feature_list[i], x)
                feats_list.append(feats)
            else:
                x = feature_list[i](x)

        # Cannot use "squeeze" as batch-size can be 1
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)

        return x, feats_list

    def forward(self, x):
        res, feat_list = self._forward_impl(x)

        return {
            KEY_OUTPUT: res,
            KEY_FEAT: feat_list
        }

    def get_distill_channels(self):
        feature_list = list(self.model.features)
        channel1 = feature_list[3].conv[1][1].num_features
        channel2 = feature_list[6].conv[1][1].num_features
        channel3 = feature_list[13].conv[1][1].num_features
        channel4 = feature_list[17].conv[1][1].num_features
        channel5 = feature_list[18][1].num_features

        return [channel1, channel2, channel3, channel4, channel5]


def get_mobilenet_v2(num_classes=1000, arch='mobilenet_v2'):
    return MobileNetV2(num_classes=num_classes, arch=arch)
