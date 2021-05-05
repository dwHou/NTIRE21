#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import kornia

class disTan(nn.Module):
    def __init__(self):
        super(disTan, self).__init__()

        def untangled_res(self, x, input):
            # B, 1(Green), H, W
            x_G = x[:, 1:2, :, :]
            input_G = input[:, 1:2, :, :]

            input_G = F.interpolate(input_G, scale_factor=2, mode='bicubic')
            lpass_G = kornia.gaussian_blur2d(input_G, (5, 5), (1.5, 1.5))
            residual_lp = lpass_G - input_G
            residual_out = x_G - input_G

            res_product = residual_lp * residual_out
            res_product = res_product + 0.5

            # texture - 1 , artifact - 0, flat - 0.5
            texture = torch.ones_like(res_product) # G channel
            artifact = torch.zeros_like(res_product) # G channel
            res_product = torch.where(res_product < 0.5, texture, res_product) # 纹理异号
            texture_mask = torch.where(res_product == 1, texture, artifact)  # 纹理mask
            res_product = torch.where(res_product > 0.5, artifact, res_product) # 噪声同号

            res_product = res_product + texture_mask

            return res_product

        # WCE loss(weighted cross-entropy)
        def cntElement(self, input):
            # 通过零范数间接来求  L0范数是指向量中非0的元素的个数。
            # torch.norm(input, p=0)

            sum = input.shape[-1] * input.shape[-2]

            input_text = input - 1
            cnt_nottext = input_text.norm(0)

            input_arf = input - 0
            cnt_notarf = input_arf.norm(0)

            input_flat = input - 0.5
            cnt_notflat = input_flat.norm(0)

            return cnt_nottext / sum, cnt_notarf / sum, cnt_notflat / sum



        def forward(self, sr, hr, input):
            hr_untan = self.untangled_res(hr, input)
            sr_untan = self.untangled_res(sr, input)
            # w_text, w_arf, w_flat = self.cntElement(sr_untan) 如果类别不平衡严重，会考虑到使用这个。
            loss = F.mse_loss(sr_untan, hr_untan)
            return loss
