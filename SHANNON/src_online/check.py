# -*- coding:utf8 -*-
from __future__ import print_function
import datetime
import argparse
from math import log10
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.LQRefiner import LQRefiner
from model.vLQRefiner import vLQRefiner
from model.PTLSTM import PTLSTM
from option import opt
from tqdm import tqdm
import logging
from data.dataset import get_training_lqset, get_training_vlqset, get_test_lqset, get_test_vlqset
import time
from tensorboardX import SummaryWriter
from loss import L1_Charbonnier_loss


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'

    model_1stg = PTLSTM().cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_1stg = nn.DataParallel(model_1stg)



    state_dict = torch.load(opt.pre_train)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove 'module'
        name = k
        new_state_dict[name] = v
    print(new_state_dict.keys())



