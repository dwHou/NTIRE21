#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from .BRCLSTM import *
from .ops.dcn.deform_conv import ModulatedDeformConv
from .base_module import *
from .unet_parts import *
from .guide_unet_parts import *

# ==========
# Spatio-temporal deformable fusion module
# ==========

class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        self.head = ConvNorm(in_channels, 16, kernel_size=1)
        self.extractfeats= RIR(2, 16)

    def forward(self, x):
        """
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats = self.head(x)
        feats = self.extractfeats(feats)

        return feats

class STDF(nn.Module):
    def __init__(self, input_len, in_nc, out_nc, nf, deform_ks=3, bilinear=True):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nl: num of conv layers in UNet encoder.
            deform_ks: size of the deformable kernel.
        """
        super(STDF, self).__init__()
        self.input_len = input_len
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        self.hardtanh = nn.Hardtanh(min_val=-64, max_val=64, inplace=False)

        self.inc = DoubleConv(in_nc, nf)
        self.down1 = Down(nf, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.dilated_conv_0 = EHDCwoSA(512 // factor)
        self.dilated_conv_1 = EHDCwoSA(512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(128, nf, bilinear)
        self.outc = OutConv(nf, nf, act=True)

        # regression head
        # why in_nc*3*size_dk?
        #   in_nc: each map use individual offset and mask
        #   2*size_dk: 2 coordinates for each point
        #   1*size_dk: 1 confidence (attention) score for each point
        #   这一层不能接ReLU
        self.offset_mask = nn.Conv2d(
            nf, self.input_len * 3 * self.size_dk, 3, padding=1
        )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        # notice group= 7, i.e., every 3 channels use individual offset and mask
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks // 2, deformable_groups=self.input_len
        )

    def forward(self, inputs):
        x1 = self.inc(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dilated_conv_0(x5)
        x5 = self.dilated_conv_1(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        off_msk = self.offset_mask(self.outc(x))
        off = self.hardtanh(
            off_msk[:, :self.input_len * 2 * self.size_dk, ...]
        )
        msk = torch.sigmoid(
            off_msk[:, self.input_len * 2 * self.size_dk:, ...]
        )

        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk),
            inplace=True
        )

        return fused_feat

class CAR(nn.Module):
    def __init__(self, nf):
        super(CAR, self).__init__()

        self.lrelu = nn.LeakyReLU(0.01)

        self.head = nn.Sequential(
            nn.Conv2d(288, nf, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
            )

        self.fusion1 = ERIR(2, nf)
        self.fusion2 = ERIR(4, nf)
        self.fusion3 = ERIR(4, nf)

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




class EMGA(nn.Module):

    def __init__(self):
        super(EMGA, self).__init__()

        self.BRCLSTM = BRCLSTM(input_dim=32,
                                 hidden_dim=[32],
                                 kernel_size=(3, 3),
                                 num_layers=1,
                                 bias=True,
                                 return_all_layers=False)

        self.radius = 4
        self.input_len = 2 * self.radius + 1

        self.enc = Encoder(in_channels=3)

        self.ffnet = STDF(
            input_len=self.input_len,
            in_nc=16 * self.input_len,
            out_nc=32 * self.input_len,
            nf=64,
            deform_ks=3,
            bilinear=True
        )

        self.tuEnc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, False),
            EHDCwSA(16),
            EHDCwSA(16),
            nn.Conv2d(16, 16, kernel_size=1, stride=1)

        )


        self.car = CAR(nf=64)

        self.rec = Guided_UNet()

        self.tem_att1 = TALayer(self.input_len)

        self.tem_att2 = TALayer(self.input_len)

    def forward(self, x, info):
        B, T, C, H, W = x.size()
        x_center = x[:, T // 2, :, :, :]

        # side information
        info = self.tuEnc(info)

        # feature extraction
        x_in = x.reshape(-1, C, H, W)
        x_enc = self.enc(x_in)
        x_enc = x_enc.reshape(B, -1, H, W)

        # deformable alignment warp
        x_align = self.ffnet(x_enc)
        x_align = x_align.view(B,  self.input_len, 32, H, W)
        # pqf = self.linear1(pqf)
        # x_align = torch.sigmoid(pqf.view(B, 9, 1, 1, 1)) * x_align

        x_align = self.tem_att1(x_align)

        # fusion
        x_fus = self.BRCLSTM(x_align)
        # pqf = self.linear2(pqf)
        # x_fus = torch.sigmoid(pqf.view(B, 9, 1, 1, 1)) * x_fus

        x_fus = self.tem_att2(x_fus)

        # reconstruction
        x_fus = x_fus.reshape(B, -1, H, W)
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
