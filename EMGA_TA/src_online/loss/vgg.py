# from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()
        vgg = models.vgg19()
        #http://download.pytorch.org/models/vgg19-dcbb9e9d.pth
        pre=torch.load('./experiment/vgg19.pth')
        vgg.load_state_dict(pre)
        # vgg_features = models.vgg19(pretrained=True).features
        vgg_features = vgg.features
        
        modules = [m for m in vgg_features]
        if conv_index.find('22') >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.Sequential(*modules[:35])
        elif conv_index == '33':
            self.vgg = nn.Sequential(*modules[:16])
        elif conv_index == '44':
            self.vgg = nn.Sequential(*modules[:26])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])
        elif conv_index == 'P':
            self.vgg = nn.ModuleList([
                nn.Sequential(*modules[:8]),
                nn.Sequential(*modules[8:16]),
                nn.Sequential(*modules[16:26]),
                nn.Sequential(*modules[26:35])
            ])    

        # vgg_mean = (0.485, 0.456, 0.406)
        # vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        # self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            # x = self.sub_mean(x)
            x = self.vgg(x)
            return x
            
        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss
