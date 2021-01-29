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
# from model.EMGA import EMGA
from model.D3DGA import D3DGA
from option import opt
from tqdm import tqdm
import logging
from data.dataset import get_training_set, get_test_set
import time
from tensorboardX import SummaryWriter
from loss import L1_Charbonnier_loss, MultiScaleLoss


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'

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
    model = D3DGA().to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if not (opt.pre_train is None):
        print('load model from %s ...' % opt.pre_train)
        model.load_state_dict(torch.load(opt.pre_train))
        print('success!')




    # criterion = Loss(opt)
    # criterion = nn.L1Loss()
    # criterion = MultiScaleLoss()
    # criterion = L1_Charbonnier_loss()
    # criterion_l2 = nn.MSELoss()
    MSE = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.7, verbose=True, patience=7)

    def train(epoch):
        epoch_loss = 0
        for iteration, batch in enumerate(training_loader, 1):
            inputs, target, info, pqf = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            optimizer.zero_grad()
            # detect inplace-error
            # torch.autograd.set_detect_anomaly(True)

            prediction = model(inputs, info)

            loss_car = criterion(prediction, target)
            epoch_loss += loss_car.item()
            loss_car.backward()
            optimizer.step()

            niter = epoch * len(training_loader) + iteration
            tb_logger.add_scalars('D3DGA', {'train_loss': loss_car.data.item()}, niter)

            print(
                '===> Epoch[{}]({}/{}): Loss: {:.6f}'.format(epoch, iteration, len(training_loader), loss_car.item()))
        print('===> Epoch {} Complete, Avg. Loss: {:.6f}'.format(epoch, epoch_loss / len(training_loader)))
        logging.info('Epoch Avg. Loss : {:.6f}'.format(epoch_loss / len(training_loader)))


    def test():
        cost_time = 0
        psnr_lst = []

        with torch.no_grad():
            for batch in tqdm(testing_loader):
                inputs, target, info, pqf = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                torch.cuda.synchronize()
                begin = time.time()

                prediction = model(inputs, info)

                torch.cuda.synchronize()
                end = time.time()
                cost_time += end - begin

                mse = MSE(prediction[4], target)
                # 0, 1, 2, 3, 4, 5, 6
                # mse = MSE(inputs[:, 4, :, :, :], target)

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
    w = [1.0 / 2.0, 1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0]


    for epoch in range(1, opt.nEpochs + 1):
        if epoch % 10 == 0:
            np.sum([np.array(w), np.array([-0.04, -0.02, 0, 0.02, 0.04])], axis=0)
            w = w.tolist()
            w = [i if i > 0.03 else 0.03 for i in w]
            print('====> changing loss weights to', w)

        criterion = MultiScaleLoss(weights = w)
        train(epoch)
        logging.info('===> in {}th epochs'.format(epoch))
        psnr, var = test()

        scheduler.step(psnr)

        if lr != optimizer.param_groups[0]['lr']:
            lr = optimizer.param_groups[0]['lr']
            logging.info('reducing lr of group 0 to {:.3e}'.format(lr))


        if psnr > best_psnr:
            best_psnr = psnr
            model_path = os.path.join('.', 'experiment', 'D3D_{:.2f}dB_{:.2f}.pth'.format(psnr, var))
            logging.info('===> save the best Model: reach {:.2f}dB PSNR'.format(best_psnr))
            # torch.save(model, model_best_path)
            torch.save(model.state_dict(), model_path)

        checkpoint(epoch)

