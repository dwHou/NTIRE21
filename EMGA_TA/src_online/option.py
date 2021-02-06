# -*- coding:utf8 -*-
from __future__ import print_function
import argparse

# Settings

parser = argparse.ArgumentParser(description='Codec RefSR baseline')

# Hardware specifications

parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for dataloader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use.')

# Data specifications

parser.add_argument('--chop', action='store_true', help='enable memory-efficient forward')
parser.add_argument('--patchSize', type=int, default=64, help='RandomCrop size')  # maxium 240
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
# parser.add_argument('--data_range', type=str, default='0-38142/38142-38999', help='train/test data range')


# Model specifications

# parser.add_argument('--n_resblocks', type=int, default=16,
#                     help='number of residual blocks')
# parser.add_argument('--n_feats', type=int, default=64,
#                     help='number of feature maps')
# parser.add_argument('--n_resgroups', type=int, default=10,
#                     help='number of residual groups')
# parser.add_argument('--res_scale', type=float, default=1,
#                     help='residual scaling')

# Training specifications
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
# learning rate is supposed to be small for residual in residual model.
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=0.01')
parser.add_argument('--pre_train', type=str, default=None, help='pre-trained model directory')

parser.add_argument('--batchSize', type=int, default=64, help='training batch size') # 37716 // batchsize
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--loss', type=str, default='1*L1', help='loss functions')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')


opt = parser.parse_args()

