import os
from importlib import import_module

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-9

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class MultiSupervision(torch.nn.Module):
    """L1 Charbonnierloss."""""
    def __init__(self, weights=None):
        super(MultiSupervision, self).__init__()
        self.eps = 1e-9
        self.weights = weights

    def one_scale(self, output, target):

        diff = torch.add(output, -target)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

    def forward(self, network_output, target_image):

        if type(network_output) not in [tuple, list]:
            network_output = [network_output]
        if self.weights is None:
            weights = [1.6, 0.5, 0.4]
        else:
            weights = self.weights
        assert (len(weights) == len(network_output))

        loss = 0
        for output, weight in zip(network_output, weights):
            loss += weight * self.one_scale(output, target_image)

        return loss



class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()

            elif loss_type.find('GDL') >= 0:
                module = import_module('loss.gradient')
                loss_function = getattr(module, 'GradientLoss')()

            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])


        device = torch.device('cuda' if args.cuda else 'cpu')
        self.loss_module.to(device)

        self.loss_module = nn.DataParallel(self.loss_module)


    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)

        loss_sum = sum(losses)

        return loss_sum
