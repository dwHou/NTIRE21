#!/usr/bin/env python

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_module import *

class UNetHDC(nn.Module):
    __constants__ = ['branch1', 'branch2', 'branch3', 'branch5']

    def __init__(self, in_channels, out_channels, stride_=1):
        super(UNetHDC, self).__init__()

        act = nn.ReLU()

        self.branch1_ = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(in_channels // 4, in_channels - 3 * (in_channels // 4), kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        )

        self.branch2_ = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            act,
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)
        )

        self.branch3_ = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            act,
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)
        )

        self.branch4_ = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1),
            act,
            nn.Conv2d(in_channels // 8, in_channels * 3 // 16, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            act,
            nn.Conv2d(in_channels * 3 // 16, in_channels // 4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            act,
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=5, bias=True, dilation=5)
        )
# ---------------------------------------------------------------------------------------------------------
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=stride_),
            act,
            nn.Conv2d(out_channels // 4, out_channels - 3 * (out_channels // 4), kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=stride_),
            act,
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            act,
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=stride_),
            act,
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            act,
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 8, kernel_size=1, stride=stride_),
            act,
            nn.Conv2d(out_channels // 8, out_channels * 3 // 16, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            act,
            nn.Conv2d(out_channels * 3 // 16, out_channels // 4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            act,
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=5, bias=True, dilation=5)
        )


        self.ConvLinear1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.ConvLinear2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        # because input, output not align
        # self.ShortPath = nn.Conv2d(nf, nf, kernel_size=1, stride=1)
        self.lrelu = nn.LeakyReLU(0.1, False)

        # self.sa = ESA(out_channels)

    def _forward1(self, x):
        branch1 = self.branch1_(x)
        branch2 = self.branch2_(x)
        branch3 = self.branch3_(x)
        branch4 = self.branch4_(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def _forward2(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        # the inside of RB_last do not use residual learning
        # res = x
        # res = self.lrelu(self.ShortPath(res))

        outputs = self._forward1(x)
        outputs = torch.cat(outputs, 1)
        outputs = self.lrelu(self.ConvLinear1(outputs))

        outputs = self._forward2(outputs)
        outputs = torch.cat(outputs, 1)
        outputs = self.lrelu(self.ConvLinear2(outputs))
        # outputs, mask = self.sa(outputs)

        # outputs += res
        return outputs


class DownHDC(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            UNetHDC(in_channels, out_channels, stride_=2),
            EHDCwoSA(out_channels),
            EHDCwoSA(out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)



class UpHDC_Ori(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = UNetHDC(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = UNetHDC(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class UpHDC(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = UNetHDC(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = UNetHDC(in_channels, out_channels)

    # x1 => sum , x2 => current , x3 => image
    def forward(self, x1, x2, image):

        diffY = x2.size()[2] - image.size()[2]
        diffX = x2.size()[3] - image.size()[3]

        image = F.pad(image, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                 diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2, image], dim=1)
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, act=True):
        super(OutConv, self).__init__()
        self.act = act
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        if self.act:
            return self.relu(self.conv(x))
        else:
            return self.conv(x)
