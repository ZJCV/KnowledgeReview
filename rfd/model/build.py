# -*- coding: utf-8 -*-

"""
@date: 2021/7/20 下午10:20
@file: build.py
@author: zj
@description: 
"""

import torch
from zcls.util.checkpoint import CheckPointer
from zcls.util import logging

logger = logging.get_logger(__name__)

from .resnet.resnet import get_resnet
from .mobilenet.mobilenet_v2 import get_mobilenet_v2


def build_model(arch_name, num_classes, preloaded):
    if 'resnet' in arch_name or 'resnext' in arch_name:
        model = get_resnet(num_classes=num_classes, arch=arch_name)
    elif 'mobilenet' in arch_name:
        model = get_mobilenet_v2(num_classes=num_classes, arch=arch_name)
    else:
        raise ValueError(f"{arch_name} doesn't exists")

    if preloaded != "":
        logger.info(f'load preloaded: {preloaded}')
        cpu_device = torch.device('cpu')
        check_pointer = CheckPointer(model)
        check_pointer.load(preloaded, map_location=cpu_device)
        logger.info("finish loading model weights")

    return model
