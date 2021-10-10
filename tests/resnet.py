# -*- coding: utf-8 -*-

"""
@date: 2021/9/25 下午12:16
@file: resnet.py
@author: zj
@description: 
"""

import torch

from rfd.model.resnet.resnet import get_resnet
from rfd.config.key_word import KEY_FEAT

if __name__ == '__main__':
    # model = get_resnet(arch='resnet18')
    model = get_resnet(arch='resnet50')
    print(model.get_distill_channels())

    data = torch.randn(1, 3, 224, 224)
    res_dict = model(data)
    for feats in res_dict[KEY_FEAT]:
        print(feats.shape)

    print(model.get_distill_channels())
