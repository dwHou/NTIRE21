#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

_reduction_modes = ['none', 'mean', 'sum']

class LoGLoss(nn.Module):
    """LoG loss.
    LoG即高斯-拉普拉斯（Laplacian of Gaussian）的缩写，使用高斯滤波器使图像平滑化之后再使用拉普拉斯滤波器使图像的轮廓更加清晰。
    """

    def __init__(self):
        super(LoGLoss, self).__init__()

    def forward(self, sr, hr):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        sr = sr[:, 1:2, :, :]
        hr = hr[:, 1:2, :, :]
        
        k = torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3).to(hr)

        # print(sr.shape)
        pred_grad = F.conv2d(sr, k, padding=1)
        target_grad = F.conv2d(hr, k, padding=1)
        

        loss = F.l1_loss(pred_grad, target_grad)

        return loss
