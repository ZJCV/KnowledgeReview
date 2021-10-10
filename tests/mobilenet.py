# -*- coding: utf-8 -*-

"""
@date: 2021/9/10 下午5:24
@file: mobilenet.py
@author: zj
@description: 
"""

import torch
import torchvision.models
from torchvision.models.mobilenetv2 import InvertedResidual, ConvBNActivation

from rfd.model.mobilenet.mobilenet_v2 import get_mobilenet_v2
from zcls.config.key_word import KEY_OUTPUT
from rfd.config.key_word import KEY_FEAT

if __name__ == '__main__':
    model = get_mobilenet_v2()
    # print(model)

    data = torch.randn(1, 3, 224, 224)
    res = model(data)

    feats_list = res[KEY_FEAT]
    outputs = res[KEY_OUTPUT]
    for feats in feats_list:
        print(feats.shape)

    # feature_list = list(model.model.features)
    # x = data
    # for features in feature_list:
    #     # x = features(x)
    #     # print(x.shape)
    #     print(features)
    #
    # print(feature_list[-1][1].num_features)

    print(model.get_distill_channels())
