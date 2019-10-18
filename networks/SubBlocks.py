#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-01 19:19:32

import torch
import torch.nn as nn
import sys
import torch.nn.functional as F

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def pixel_unshuffle(input, upscale_factor):
    '''
    Input:
        input: (N, C, rH, rW) tensor
    output:
        (N, r^2C, H, W)
    Written by Kai Zhang: https://github.com/cszn/FFDNet
    '''
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view( batch_size, channels, out_height, upscale_factor,
                                                                         out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

class PixelUnShuffle(nn.Module):
    '''
    Input:
        input: (N, C, rH, rW) tensor
    output:
        (N, r^2C, H, W)
    Written by Kai Zhang: https://github.com/cszn/FFDNet
    '''
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

