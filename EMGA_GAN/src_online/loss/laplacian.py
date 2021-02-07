#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

_reduction_modes = ['none', 'mean', 'sum']

class LaplacianLoss(nn.Module):
    """Gradient loss.
    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self):
        super(LaplacianLoss, self).__init__()

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
