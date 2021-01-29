#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dcn.modules.deform_conv import *
import torch.nn.functional as F
# from .ops.dcn.deform_conv import ModulatedDeformConv
from .base_module import *
from .guide_unet_parts import *
import functools

# ==========
# Spatio-temporal deformable fusion module
# ==========

class CAR(nn.Module):
    def __init__(self, nf):
        super(CAR, self).__init__()

        self.lrelu = nn.LeakyReLU(0.01)

        self.head = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
            )

        self.fusion1 = ERIR(3, nf)
        self.fusion2 = ERIR(6, nf)
        self.fusion3 = ERIR(6, nf)

        rfa = []
        rfa.append(ConvNorm(nf * 4, nf * 2))
        rfa.append(self.lrelu)
        rfa.append(ConvNorm(nf * 2, nf))
        self.rfa = nn.Sequential(*rfa)

        self.ca = CALayer(nf)
        self.tail = nn.Conv2d(nf, nf, kernel_size=1, stride=1)

    def forward(self, x, info):
        """
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats = self.head(x)
        feats_ori = feats
        feats1 = self.fusion1(feats, info)
        feats2 = self.fusion2(feats1, info)
        feats3 = self.fusion3(feats2, info)
        # B, C+C+C+C, H, W
        feats = torch.cat([feats_ori, feats1, feats2, feats3], 1)
        feats = self.ca(self.rfa(feats)) # add rfa

        out = self.tail(feats)
        return out



# ==========
# Quality enhancement module
# ==========


def predict_image(in_planes):
    return nn.Conv2d(in_planes,3,kernel_size=3,stride=1,padding=1,bias=False)

class PreImage(nn.Module):

    def __init__(self, in_planes):
        super(PreImage, self).__init__()
        self.in_planes = in_planes
        self.conv = nn.Conv2d(self.in_planes,3,kernel_size=3,stride=1,padding=1,bias=False)

    def forward(self, x1, x2, image):
        diffY = x2.size()[2] - image.size()[2]
        diffX = x2.size()[3] - image.size()[3]
        image = F.pad(image, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x1, x2, image], dim=1)
        return self.conv(x)


class Guided_UNet(nn.Module):
    def __init__(self, bilinear=True):
        super(Guided_UNet, self).__init__()
        nf = 64
        self.head = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

        self.down1 = DownHDC(nf, 128)
        self.down2 = DownHDC(128, 256)
        self.down3 = DownHDC(256, 512)
        self.down4 = DownHDC(512, 1024)

        self.predict_image4 = predict_image(1024)
        self.predict_image3 = PreImage(1024 + 3)
        self.predict_image2 = PreImage(512 + 3)
        self.predict_image1 = PreImage(256 + 3)

        self.up4 = UpHDC_Ori(1024, 512, bilinear)
        self.up3 = UpHDC(1024 + 3, 256, bilinear)
        self.up2 = UpHDC(512 + 3, 128, bilinear)
        self.up1 = UpHDC(256 + 3, nf, bilinear)

        self.out = nn.Sequential(
            nn.Conv2d(131, nf, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, 3, kernel_size=3, stride=1, padding=1)
        )


        self.tail = ConvNorm(nf, 3)

    def forward(self, inputs):
        b, _, h, w = inputs.size()
        x0 = self.head(inputs)
        x1 = self.down1(x0) # 128
        x2 = self.down2(x1) # 256
        x3 = self.down3(x2) # 512
        x4 = self.down4(x3) # 1024

        x = self.up4(x4)
        image4 = self.predict_image4(x4)
        image4_up = F.interpolate(image4, scale_factor=2, mode='bilinear', align_corners=False)

        image3 = self.predict_image3(x, x3, image4_up)
        x = self.up3(x, x3, image4_up)
        image3_up = F.interpolate(image3, scale_factor=2, mode='bilinear', align_corners=False)

        image2 = self.predict_image2(x, x2, image3_up)
        x = self.up2(x, x2, image3_up)
        image2_up = F.interpolate(image2, scale_factor=2, mode='bilinear', align_corners=False)

        image1 = self.predict_image1(x, x1, image2_up)
        x = self.up1(x, x1, image2_up)
        image1_up = F.interpolate(image1, size=(h, w), mode='bilinear', align_corners=False)

        # 64 + 64 + 3 = 131
        concat0 = torch.cat([x, x0, image1_up], dim=1)
        image = self.out(concat0)

        return image4, image3, image2, image1, image

class D3DRIR(nn.Module):
    def __init__(self, n_basemodules, n_feats):
        super(D3DRIR, self).__init__()

        self.headConv = nn.Sequential(
            nn.Conv3d(in_channels=3 * 2 * 2, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        # define modules: head, body, tail
        modules_body = [
            ResBlock_3d(nf=n_feats)
            for _ in range(n_basemodules)]

        self.body = nn.Sequential(*modules_body)

        self.tailConv = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # Build input tensor B, C, H, W
        x = self.headConv(x)
        x = self.body(x)
        out = self.tailConv(x)
        return out


class D3DGA(nn.Module):

    def __init__(self):
        super(D3DGA, self).__init__()

        self.radius = 4
        self.input_len = 2 * self.radius + 1

        self.enc = D3DRIR(6, 16)

        self.tuEnc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, False),
            EHDCwSA(16),
            EHDCwSA(16),
            nn.Conv2d(16, 16, kernel_size=1, stride=1)
        )

        self.tem_att = TALayer(self.input_len)

        self.TA = nn.Sequential(
            nn.Conv2d(self.input_len * 4, 64, 1, 1, bias=True),
            nn.LeakyReLU(0.1, False),
            nn.Conv2d(64, 64, 1, 1, bias=True)
        )

        self.car = CAR(nf=64)

        self.rec = Guided_UNet()

        self.downshuffler = PixelShuffle3D(1/2)
        self.shuffler = PixelShuffle3D(2)

    def forward(self, x, info):
        B, T, C, H, W = x.size()
        x_center = x[:, T // 2, :, :, :]

        # side information
        info = self.tuEnc(info)

        # feature extraction and alignment
        # B, C, T, H, W
        x = self.downshuffler(x)
        x_in = x.permute(0,2,1,3,4).contiguous()
        x_enc = self.enc(x_in)
        x_enc = x_enc.permute(0,2,1,3,4).contiguous()
        x = self.shuffler(x_enc)

        # pqf = self.linear2(self.act(self.linear1(pqf)))
        # x_enc = torch.sigmoid(pqf.view(B, 9, 1, 1, 1)) * x_enc
        x_enc = self.tem_att(x_enc)
        x_enc = x_enc.reshape(B, -1, H, W)

        # bottleneck fusion
        x_fus = self.TA(x_enc)

        # reconstruction
        x_car = self.car(x_fus, info)

        # output multiscale image
        x4, x3, x2, x1, x = self.rec(x_car)

        # res: add middle frame

        b, _, h4, w4 = x4.size()
        b, _, h3, w3 = x3.size()
        b, _, h2, w2 = x2.size()
        b, _, h1, w1 = x1.size()

        x_center_4 = F.interpolate(x_center, size=(h4, w4), mode='bilinear', align_corners=False)
        x_center_3 = F.interpolate(x_center, size=(h3, w3), mode='bilinear', align_corners=False)
        x_center_2 = F.interpolate(x_center, size=(h2, w2), mode='bilinear', align_corners=False)
        x_center_1 = F.interpolate(x_center, size=(h1, w1), mode='bilinear', align_corners=False)

        image_4 = x4 + x_center_4
        image_3 = x3 + x_center_3
        image_2 = x2 + x_center_2
        image_1 = x1 + x_center_1

        image_finally = x + x_center

        return image_4,image_3,image_2,image_1,image_finally
