# -*- coding: utf-8 -*-

"""
@date: 2021/8/29 下午3:16
@file: build.py
@author: zj
@description: 
"""

from zcls.model import registry
from zcls.model.criterions.crossentropy_loss import CrossEntropyLoss
from zcls.model.criterions.label_smoothing_loss import LabelSmoothingLoss

from .rfd_loss import RFDLoss


def build_criterion(cfg, device):
    return registry.CRITERION[cfg.MODEL.CRITERION.NAME](cfg).to(device=device)
