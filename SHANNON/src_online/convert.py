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
from model.MPRNet import MPRNet
from option import opt
from tqdm import tqdm
import logging
import time


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5, 6'

    logging.basicConfig(filename='./LOG/' + 'Convert' + '.log', level=logging.INFO)

    opt = opt
    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)

    device = torch.device("cuda" if opt.cuda else "cpu")

    print('===> Loading datasets')



    print('===> Building model')
    model = MPRNet().to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if not (opt.pre_train is None):
        print('load model from %s ...' % opt.pre_train)
        model.load_state_dict(torch.load(opt.pre_train))
        print('success!')


    torch.save(model.state_dict(), './experiment/e16_29.71dB_version1.1.0.pth', _use_new_zipfile_serialization=False)
