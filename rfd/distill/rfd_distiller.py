# -*- coding: utf-8 -*-

"""
@date: 2021/8/29 下午1:11
@file: ofd_distiller.py
@author: zj
@description: 
"""

import torch

import torch.nn as nn
import torch.nn.functional as F
from zcls.config.key_word import KEY_OUTPUT
from zcls.model.init_helper import init_weights

from rfd.config.key_word import KEY_FEAT, KEY_T_FEAT, KEY_S_FEAT
from rfd.model.resnet.resnet import ResNet
from rfd.model.mobilenet.mobilenet_v2 import MobileNetV2


class ABF(nn.Module):

    def __init__(self, in_channel, out_channel, mid_channel, is_fuse=True):
        super(ABF, self).__init__()

        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(mid_channel)
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.att_conv = None if not is_fuse else nn.Sequential(
            nn.Conv2d(mid_channel * 2, 2, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

        self.__init_weights()

    def __init_weights(self):
        nn.init.kaiming_uniform_(self.conv_first[0].weight, a=1)
        nn.init.kaiming_uniform_(self.conv_last[0].weight, a=1)

    def forward(self, x, y=None, shape=None):
        assert len(x.shape) == 4
        N, _, H, W = x.shape[:4]

        x = self.conv_first(x)
        if self.att_conv is not None:
            # up sample residual features
            y = F.interpolate(y, shape, mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)

            x = (x * z[:, 0].view(N, 1, H, W) + y * z[:, 1].view(N, 1, H, W))
        y = self.conv_last(x)

        return y, x


class RFDDistiller(nn.Module):

    def __init__(self, t_net, s_net):
        super().__init__()
        assert isinstance(t_net, torch.nn.Module)
        assert isinstance(s_net, torch.nn.Module)

        reversed_t_net_channels = list(reversed(t_net.get_distill_channels()))
        reversed_s_net_channels = list(reversed(s_net.get_distill_channels()))
        assert len(reversed_t_net_channels) == len(reversed_s_net_channels)
        self.channels_len = len(reversed_t_net_channels)

        mid_channel = 512 if isinstance(s_net, ResNet) else 256
        s_transform_list = list()
        for idx, (t_channel, s_channel) in enumerate(zip(reversed_t_net_channels, reversed_s_net_channels)):
            s_transform_list.append(ABF(s_channel, t_channel, mid_channel, idx != 0))

        self.t_net = t_net
        # freeze grad update abd use eval state
        self.t_net.requires_grad_(False)
        self.t_net.eval()

        self.s_net = s_net
        self.s_net.train()

        self.s_transform_list = nn.ModuleList(s_transform_list)
        # self.__init_weights__()

    def __init_weights__(self):
        for student_transform in self.s_transform_list:
            init_weights(student_transform)

    def forward(self, x):
        t_outputs_dict = self.t_net(x)
        s_outputs_dict = self.s_net(x)

        t_transform_feat_list = t_outputs_dict[KEY_FEAT]
        s_transform_feat_list = s_outputs_dict[KEY_FEAT]
        assert len(t_transform_feat_list) == len(s_transform_feat_list) == self.channels_len

        res_features = None
        res_s_transform_feat_list = list()
        for idx in range(self.channels_len):
            s_feat = s_transform_feat_list[self.channels_len - idx - 1]
            s_transform = self.s_transform_list[idx]

            _, _, h, w = s_feat.shape[:4]
            shape = (h, w)
            if idx == 0:
                out_features, res_features = s_transform(s_feat, shape=shape)
            else:
                out_features, res_features = s_transform(s_feat, y=res_features, shape=shape)
            res_s_transform_feat_list.append(out_features)
        res_s_transform_feat_list.reverse()

        return {
            KEY_OUTPUT: s_outputs_dict[KEY_OUTPUT],
            KEY_T_FEAT: t_transform_feat_list,
            KEY_S_FEAT: res_s_transform_feat_list
        }
