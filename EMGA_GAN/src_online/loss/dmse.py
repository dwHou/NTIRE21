# from model import common
https://github.com/open-mmlab/mmediting/tree/master/mmedit/models/losses
可以关注一下MMLab的实现版本

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class dMSE(nn.Module):
    def __init__(self):
        super(dMSE, self).__init__()
        
    def gradient_1order(self,x,h_x=None,w_x=None):
        if h_x is None and w_x is None:
            h_x = x.size()[-2]
            w_x = x.size()[-1]
            
#         r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
#         l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
#         t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
#         b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        # xgrad = 0.5 * torch.sqrt(torch.pow(r - l, 2) + torch.pow(t - b, 2))
        xgrad = torch.abs(r - l) + torch.abs(t - b)
        return xgrad    


    def forward(self, sr, hr):
        srgrad = self.gradient_1order(sr)
        hrgrad = self.gradient_1order(hr)
        
        loss = F.mse_loss(srgrad, hrgrad)

        return loss     


'''
以上代码如果用于计算图像梯度，获取图像的轮廓，完全是没问题的，但是当将计算结果作为梯度损失训练网络时，由于在训练网络时会进行梯度传播，
则会对代码代块中的参数进行优化，从而使网络出现梯度爆炸，从而无法很好的训练网络。
为了能够使用torch计算图像的梯度并作为梯度损失优化网络，本人希望设计卷积核并在参数列表中不需要梯度下降，从而不会造成梯度在该卷积操作中传播。
计算梯度的网络设计代码如下
'''


'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Gradient_Net(nn.Module):
    def __init__(self):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

        kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x)
        grad_y = F.conv2d(x, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient

gradient_model = Gradient_Net().to(device)

class dMSE(nn.Module):
    def __init__(self):
        super(dMSE, self).__init__()
        
    def forward(self, sr, hr):
        srgrad = gradient_model(sr)
        hrgrad = gradient_model(hr)
        
        loss = F.mse_loss(srgrad, hrgrad)

        return loss         
'''
