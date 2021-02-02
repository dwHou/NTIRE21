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
from model.EMGA import EMGA
from option import opt
from tqdm import tqdm
import logging
from data.dataset import get_training_set, get_test_set
import time
from tensorboardX import SummaryWriter
from loss import L1_Charbonnier_loss, MultiScaleLoss


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'

    logging.basicConfig(filename='./LOG/' + 'LatestVersion' + '.log', level=logging.INFO)
    tb_logger = SummaryWriter('./LOG/')

    opt = opt
    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)

    device = torch.device("cuda" if opt.cuda else "cpu")

    print('===> Loading datasets')

    train_set = get_training_set()
    test_set = get_test_set()

    training_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)

    testing_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                     shuffle=False)


    print('===> Building model')
    model = EMGA().to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if not (opt.pre_train is None):
        print('load model from %s ...' % opt.pre_train)
        model.load_state_dict(torch.load(opt.pre_train))
        print('success!')


    torch.save(model.state_dict(), './experiment/M_28.72_version1.4.0.pth', _use_new_zipfile_serialization=False)
