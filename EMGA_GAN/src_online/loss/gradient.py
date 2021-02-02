#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

_reduction_modes = ['none', 'mean', 'sum']

class GradientLoss(nn.Module):
    """Gradient loss.
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self):
        super(GradientLoss, self).__init__()

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

        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(hr)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(hr)

        # print(sr.shape)
        pred_grad_x = F.conv2d(sr, kx, padding=1)
        pred_grad_y = F.conv2d(sr, ky, padding=1)
        target_grad_x = F.conv2d(hr, kx, padding=1)
        target_grad_y = F.conv2d(hr, ky, padding=1)

        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        loss = loss_x + loss_y
        return loss
