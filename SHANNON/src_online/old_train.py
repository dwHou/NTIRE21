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
from model.MPRNet_new import MPRNet
from option import opt
from tqdm import tqdm
import logging
from data.dataset_part import get_training_010, get_test_010
import time
import loss
from loss import L1_Charbonnier_loss, MultiSupervision
from torch_ema import ExponentialMovingAverage
from collections.abc import Iterable


def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    # for name, child in model.named_children():
    for name, child in model.module.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

    for name, child in model.module.stage3_orsnet.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
    torch.backends.cudnn.benchmark = True
    logging.basicConfig(filename='./LOG/' + 'MPRNet' + '.log', level=logging.INFO)

    opt = opt
    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)

    device = torch.device("cuda" if opt.cuda else "cpu")

    print('===> Loading datasets')

    train_set = get_training_010()
    test_set = get_test_010()

    training_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, pin_memory=True,
                                     shuffle=True)

    testing_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                     shuffle=False)


    print('===> Building model')
    model = MPRNet().to(device)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if not (opt.pre_train is None):
        print('load model from %s ...' % opt.pre_train)
        model.load_state_dict(torch.load(opt.pre_train))
        print('success!')




    criterion_gdl = loss.Loss(opt)
    # criterion = nn.L1Loss()
    criterion_l1 = MultiSupervision()
    criterion_l2 = nn.MSELoss()
    # criterion = L1_Charbonnier_loss()
    # criterion_l2 = nn.MSELoss()
    MSE = nn.MSELoss()

    for param in model.parameters():
        param.requires_grad = False

    unfreeze_by_names(model, ('tail'))

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    optimizer.load_state_dict(torch.load(opt.pre_optim))
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999, use_num_updates=False)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.7, verbose=True, patience=5)

    def train(epoch):
        epoch_loss = 0
        model.eval()
        accumulation_steps = 3
        for iteration, batch in enumerate(training_loader, 1):
            inputs, target, info = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            # detect inplace-error
            # torch.autograd.set_detect_anomaly(True)

            prediction = model(inputs, info)
            # loss_car = criterion_l1(prediction, target) + criterion_gdl(prediction[1], target) + criterion_gdl(prediction[2], target)
            loss_car = criterion_l2(prediction[0], target) 
            # loss_gdl = criterion_gdl(prediction[1], target) + criterion_gdl(prediction[2], target)

            epoch_loss += loss_car.item()
            loss_car.backward()
            
            if((iteration+1)%accumulation_steps)==0:
                optimizer.step()
                ema.update(model.parameters())

            # print('===> Epoch[{}]({}/{})'.format(epoch, iteration, len(training_loader)))

            print('===> Epoch[{}]({}/{}): Loss: {:.6f}'.format(epoch, iteration, len(training_loader), loss_car.item()))

        print('===> Epoch {} Complete, Avg. Loss: {:.6f}'.format(epoch, epoch_loss / len(training_loader)))
        logging.info('Epoch Avg. Loss : {:.6f}'.format(epoch_loss / len(training_loader)))


    def test(e=0):
        cost_time = 0
        psnr_lst = []
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        model.eval()
        with torch.no_grad():
            for batch in tqdm(testing_loader):
                inputs, target, info = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                torch.cuda.synchronize()
                begin = time.time()

                prediction = model(inputs, info)

                torch.cuda.synchronize()
                end = time.time()
                cost_time += end - begin

                mse = MSE(prediction[0], target)
                # 0, 1, 2, 3, 4, 5, 6
                # mse = MSE(inputs[:, 1, :, :, :], target)

                psnr = 10 * log10(1 / mse.item())
                psnr_lst.append(psnr)
                print('psnr: {:.2f}'.format(psnr))

            #   ---------------------------------------------------------------

        psnr_var = np.var(psnr_lst)
        psnr_sum = np.sum(psnr_lst)
        # wandb.log({"Avg. PSNR": avg_psnr / len(testing_data_loader)})
     
        print('===> Avg. PSNR: {:.4f} dB, per frame use {} ms'.format( \
        psnr_sum / len(testing_loader) , cost_time * 1000 / len(testing_loader)))

        logging.info('frames avg. psnr {:.4f} dB with var{:.2f}'.format(psnr_sum / len(testing_loader), psnr_var))


        return psnr_sum / len(testing_loader), psnr_var

    def checkpoint(epoch):
        model_path = os.path.join('.', 'experiment', 'latestckp', 'latest.pth')
        torch.save(model.state_dict(), model_path)
        print('Checkpoint saved to {}'.format(model_path))



    # 记录实验时间
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info('\n experiment in {}'.format(nowTime))

    lr = opt.lr
    best_psnr = 25.0
    # psnr, var = test()

    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        logging.info('===> in {}th epochs'.format(epoch))
        psnr, var = test(e=epoch)

        scheduler.step(psnr)

        if lr != optimizer.param_groups[0]['lr']:
            lr = optimizer.param_groups[0]['lr']
            logging.info('reducing lr of group 0 to {:.3e}'.format(lr))


        if psnr > best_psnr:
            best_psnr = psnr
            logging.info('===> save the best Model: reach {:.2f}dB PSNR'.format(best_psnr))


        model_path = os.path.join('.', 'experiment', 'e{}_{:.2f}dB_{:.2f}.pth'.format(epoch, psnr, var))

        optimizer_path = os.path.join('.', 'experiment', 'e{}_optimizer_{:.2f}dB.pth'.
                                      format(epoch, psnr))

        # torch.save(model, model_best_path)
        torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)
        torch.save(model.state_dict(), f'/apdcephfs/share_819798/devonn_ckp/e{epoch}_{psnr:.2f}dB_{var:.2f}.pth')
        torch.save(optimizer.state_dict(), optimizer_path)
       
        ema.restore(model.parameters())
        # checkpoint(epoch)
