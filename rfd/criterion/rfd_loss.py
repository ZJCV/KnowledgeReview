# -*- coding: utf-8 -*-

"""
@date: 2021/8/29 上午11:56
@file: ofd_loss.py
@author: zj
@description: 
"""

from abc import ABC
import torch.nn as nn
import torch.nn.functional as F
from zcls.config.key_word import KEY_OUTPUT, KEY_LOSS
from zcls.model import registry

from rfd.config.key_word import KEY_T_FEAT, KEY_S_FEAT, KEY_DISTILL_LOSS, KEY_TASK_LOSS


def hcl(t_feat, s_feat):
    assert t_feat.shape == s_feat.shape

    N, C, H, W = t_feat.shape[:4]
    loss = F.mse_loss(t_feat, s_feat, reduction='mean')
    cnt = 1.0
    tot = 1.0
    for level in [4, 2, 1]:
        if level >= H:
            continue
        tmp_t_feat = F.adaptive_avg_pool2d(t_feat, (level, level))
        tmp_s_feat = F.adaptive_avg_pool2d(s_feat, (level, level))
        cnt /= 2.0
        loss += F.mse_loss(tmp_t_feat, tmp_s_feat, reduction='mean') * cnt
        tot += cnt
    loss = loss / tot

    return loss


@registry.CRITERION.register('RFDLoss')
class RFDLoss(nn.Module, ABC):

    def __init__(self, cfg):
        super(RFDLoss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss(reduction='mean')
        self.distill_loss = nn.MSELoss(reduction='none')
        self.lam = cfg.DISTILL.LAMBDA

    def __call__(self, output_dict, targets):
        assert isinstance(output_dict, dict) and KEY_OUTPUT in output_dict.keys()
        inputs = output_dict[KEY_OUTPUT]
        task_loss = self.task_loss(inputs, targets)

        t_feat_list = output_dict[KEY_T_FEAT]
        s_feat_list = output_dict[KEY_S_FEAT]
        assert len(t_feat_list) == len(s_feat_list)

        distill_loss = 0
        for i, (t_feat, s_feat) in enumerate(zip(t_feat_list, s_feat_list)):
            distill_loss += hcl(t_feat, s_feat)
        distill_loss = distill_loss * self.lam

        return {
            KEY_LOSS: task_loss + distill_loss,
            KEY_TASK_LOSS: task_loss,
            KEY_DISTILL_LOSS: distill_loss,
        }
