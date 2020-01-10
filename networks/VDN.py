#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-01 19:35:06

import torch.nn as nn
from .DnCNN import DnCNN
from .UNet import UNet

def weight_init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return net

class VDN(nn.Module):
    def __init__(self, in_channels, wf=64, dep_S=5, dep_U=4, slope=0.2):
        super(VDN, self).__init__()
        self.DNet = UNet(in_channels, in_channels*2, wf=wf, depth=dep_U, slope=slope)
        self.SNet = DnCNN(in_channels, in_channels*2, dep=dep_S, num_filters=64, slope=slope)

    def forward(self, x, mode='train'):
        if mode.lower() == 'train':
            phi_Z = self.DNet(x)
            phi_sigma = self.SNet(x)
            return phi_Z, phi_sigma
        elif mode.lower() == 'test':
            phi_Z = self.DNet(x)
            return phi_Z
        elif mode.lower() == 'sigma':
            phi_sigma = self.SNet(x)
            return phi_sigma
