# -*- coding: utf-8 -*-

"""
@date: 2021/8/29 下午3:02
@file: build.py
@author: zj
@description: 
"""

from torch.nn.parallel import DistributedDataParallel as DDP

import zcls.util.distributed as du

from ..model.build import build_model
from .rfd_distiller import RFDDistiller


def build_distiller(cfg, device):
    s_arch_name = cfg.DISTILL.S_ARCH
    s_num_classes = cfg.DISTILL.S_NUM_CLASSES
    s_preloaded = cfg.DISTILL.S_PRELOADED
    s_net = build_model(s_arch_name, s_num_classes, s_preloaded)

    distiller_name = cfg.DISTILL.DISTILLER
    if distiller_name == 'RFD':
        t_arch_name = cfg.DISTILL.T_ARCH
        t_num_classes = cfg.DISTILL.T_NUM_CLASSES
        t_preloaded = cfg.DISTILL.T_PRELOADED
        t_net = build_model(t_arch_name, t_num_classes, t_preloaded)

        model = RFDDistiller(t_net, s_net)
    else:
        # normal training
        model = s_net

    model = model.to(device=device)
    if du.get_world_size() > 1:
        model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    return model
