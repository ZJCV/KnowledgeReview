# -*- coding: utf-8 -*-

"""
@date: 2021/8/29 下午2:46
@file: __init__.py.py
@author: zj
@description: 
"""

from yacs.config import CfgNode as CN
from zcls.config import get_cfg_defaults


def add_custom_config(_C):
    # Add your own customized config.
    _C.DISTILL = CN()
    _C.DISTILL.T_ARCH = 'resnet50'
    _C.DISTILL.T_NUM_CLASSES = 1000
    _C.DISTILL.T_PRELOADED = ''

    _C.DISTILL.S_ARCH = 'resnet18'
    _C.DISTILL.S_NUM_CLASSES = 1000
    _C.DISTILL.S_PRELOADED = ''

    _C.DISTILL.DISTILLER = ''
    _C.DISTILL.LAMBDA = 1e-4

    return _C


cfg = add_custom_config(get_cfg_defaults())
