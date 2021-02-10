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
from data.dataset import get_training_set, get_test_set, get_training_hqf, get_training_lqf, get_test_hqf, get_test_lqf
import time
from tensorboardX import SummaryWriter
from loss import L1_Charbonnier_loss, MultiScaleLoss
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{opt.local_rank}'
    local_rank = opt.local_rank

    if local_rank == 0:
        logging.basicConfig(filename='./LOG/' + 'LatestVersion' + '.log', level=logging.INFO)
        tb_logger = SummaryWriter('./LOG/')

    opt = opt
    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)

    device = torch.device("cuda" if opt.cuda else "cpu")

    print('===> Loading datasets')

# initial backend as NCCL
    dist.init_process_group(backend='nccl')
    ngpu_per_node = torch.cuda.device_count()
    # batchsize_per_node = int(opt.batchSize / ngpu_per_node)
    batchsize_per_node = opt.batchSize

    train_hqf = get_training_hqf()
    test_hqf = get_test_hqf()

    train_lqf = get_training_lqf()
    test_lqf = get_test_lqf()

    train_sampler_hqf = torch.utils.data.distributed.DistributedSampler(train_hqf)
    train_sampler_lqf = torch.utils.data.distributed.DistributedSampler(train_lqf)

    training_loader_hqf = DataLoader(dataset=train_hqf, num_workers=opt.threads, batch_size=batchsize_per_node,
                                     pin_memory=True, shuffle=(train_sampler_hqf is None), sampler=train_sampler_hqf)
    training_loader_lqf = DataLoader(dataset=train_lqf, num_workers=opt.threads, batch_size=batchsize_per_node,
                                     pin_memory=True, shuffle=(train_sampler_lqf is None), sampler=train_sampler_lqf)
    testing_loader_hqf = DataLoader(dataset=test_hqf, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                     shuffle=False)
    testing_loader_lqf = DataLoader(dataset=test_lqf, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                    shuffle=False)


    print('===> Building model')
    model_hqf = EMGA().to(device)
    model_lqf = EMGA().to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # model = nn.DataParallel(model)
        model_hqf = nn.parallel.DistributedDataParallel(model_hqf, find_unused_parameters=True)
        model_hqf = nn.SyncBatchNorm.convert_sync_batchnorm(model_hqf)
        model_lqf = nn.parallel.DistributedDataParallel(model_lqf, find_unused_parameters=True)
        model_lqf = nn.SyncBatchNorm.convert_sync_batchnorm(model_lqf)

    if not (opt.pre_train_hqf is None):
        print('load model from %s ...' % opt.pre_train_hqf)
        # model.load_state_dict(torch.load(opt.pre_train))
        model_hqf.load_state_dict({k.replace('module.',''):v for k,v in torch.load(opt.pre_train_hqf).items()})
        print('success!')

    if not (opt.pre_train_lqf is None):
        print('load model from %s ...' % opt.pre_train_lqf)
        # model.load_state_dict(torch.load(opt.pre_train))
        model_lqf.load_state_dict({k.replace('module.',''):v for k,v in torch.load(opt.pre_train_lqf).items()})
        print('success!')




    # criterion = Loss(opt)
    # criterion = nn.L1Loss()
    # criterion = MultiScaleLoss()
    # criterion = L1_Charbonnier_loss()
    # criterion_l2 = nn.MSELoss()
    MSE = nn.MSELoss()

    optimizer_hqf = optim.Adam(model_hqf.parameters(), lr=opt.lr)
    scheduler_hqf = optim.lr_scheduler.ReduceLROnPlateau(optimizer_hqf, 'max', factor=0.7, verbose=True, patience=7)

    optimizer_lqf = optim.Adam(model_lqf.parameters(), lr=opt.lr)
    scheduler_lqf = optim.lr_scheduler.ReduceLROnPlateau(optimizer_lqf, 'max', factor=0.7, verbose=True, patience=7)

    def train_hqf(epoch):
        epoch_loss = 0
        for iteration, batch in enumerate(training_loader_hqf, 1):
            inputs, target, info, pqf = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            optimizer_hqf.zero_grad()
            # detect inplace-error
            # torch.autograd.set_detect_anomaly(True)

            prediction = model_hqf(inputs, info, pqf)

            loss_car = criterion(prediction, target)
            epoch_loss += loss_car.item()
            loss_car.backward()
            optimizer_hqf.step()
            dist.barrier()

            niter = epoch * len(training_loader_hqf) + iteration

            if local_rank == 0:
                tb_logger.add_scalars('HQF', {'train_loss': loss_car.data.item()}, niter)

            print(
                '===> Epoch[{}]({}/{}): Loss: {:.6f}'.format(epoch, iteration, len(training_loader_hqf), loss_car.item()))
        print('===> Epoch {} Complete, Avg. HQF Loss: {:.6f}'.format(epoch, epoch_loss / len(training_loader_hqf)))
        if local_rank == 0:
            logging.info('Epoch Avg. HQF Loss : {:.6f}'.format(epoch_loss / len(training_loader_hqf)))

    def train_lqf(epoch):
        epoch_loss = 0
        for iteration, batch in enumerate(training_loader_lqf, 1):
            inputs, target, info, pqf = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            optimizer_lqf.zero_grad()
            # detect inplace-error
            # torch.autograd.set_detect_anomaly(True)

            prediction = model_lqf(inputs, info, pqf)

            loss_car = criterion(prediction, target)
            epoch_loss += loss_car.item()
            loss_car.backward()
            optimizer_lqf.step()
            dist.barrier()

            niter = epoch * len(training_loader_lqf) + iteration

            if local_rank == 0:
                tb_logger.add_scalars('LQF', {'train_loss': loss_car.data.item()}, niter)

            print(
                '===> Epoch[{}]({}/{}): Loss: {:.6f}'.format(epoch, iteration, len(training_loader_lqf), loss_car.item()))
        print('===> Epoch {} Complete, Avg. LQF Loss: {:.6f}'.format(epoch, epoch_loss / len(training_loader_lqf)))
        if local_rank == 0:
            logging.info('Epoch Avg. LQF Loss : {:.6f}'.format(epoch_loss / len(training_loader_lqf)))


    def test():
        cost_time = 0
        psnr_lst = []

        psnr_hqf = []
        psnr_lqf = []

        with torch.no_grad():
            #   ----------------------------------------------------------------
            for batch in tqdm(testing_loader_hqf):
                inputs, target, info, pqf = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                torch.cuda.synchronize()
                begin = time.time()

                prediction = model_hqf(inputs, info, pqf)

                torch.cuda.synchronize()
                end = time.time()
                cost_time += end - begin

                mse = MSE(prediction[4], target)
                # 0, 1, 2, 3, 4, 5, 6
                # mse = MSE(inputs[:, 4, :, :, :], target)

                psnr = 10 * log10(1 / mse.item())
                psnr_lst.append(psnr)
                psnr_hqf.append(psnr)
                print('psnr: {:.2f}'.format(psnr))

            #   ---------------------------------------------------------------

            for batch in tqdm(testing_loader_lqf):
                inputs, target, info, pqf = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                torch.cuda.synchronize()
                begin = time.time()

                prediction = model_lqf(inputs, info, pqf)

                torch.cuda.synchronize()
                end = time.time()
                cost_time += end - begin

                mse = MSE(prediction[4], target)
                # 0, 1, 2, 3, 4, 5, 6
                # mse = MSE(inputs[:, 4, :, :, :], target)

                psnr = 10 * log10(1 / mse.item())
                psnr_lst.append(psnr)
                psnr_lqf.append(psnr)
                print('psnr: {:.2f}'.format(psnr))

        psnr_var = np.var(psnr_lst)
        psnr_sum = np.sum(psnr_lst)

        psnr_hqf_var = np.var(psnr_hqf)
        psnr_hqf_sum = np.sum(psnr_hqf)

        psnr_lqf_var = np.var(psnr_lqf)
        psnr_lqf_sum = np.sum(psnr_lqf)

        # wandb.log({"Avg. PSNR": avg_psnr / len(testing_data_loader)})

        print('===> Avg. PSNR: {:.4f} dB, per frame use {} ms'.format( \
        psnr_sum / (len(testing_loader_hqf) + len(testing_loader_lqf)) , cost_time * 1000 / (len(testing_loader_hqf) + len(testing_loader_lqf))))
        if local_rank == 0:
            logging.info('frames avg. psnr {:.4f} dB with var{:.2f}'.format(psnr_sum / (len(testing_loader_hqf) + len(testing_loader_lqf)), psnr_var))
            logging.info('frames avg. hqf psnr {:.4f} dB with var{:.2f}'.format(psnr_hqf_sum / len(testing_loader_hqf), psnr_hqf_var))
            logging.info('frames avg. lqf psnr {:.4f} dB with var{:.2f}'.format(psnr_lqf_sum / len(testing_loader_lqf), psnr_lqf_var))

        return psnr_hqf_sum / len(testing_loader_hqf), psnr_hqf_var, psnr_lqf_sum / len(testing_loader_lqf), psnr_lqf_var

    def checkpoint(epoch):
        model_hqf_path = os.path.join('.', 'checkpoint_all', 'latestckp', 'hqf.pth')
        model_lqf_path = os.path.join('.', 'checkpoint_all', 'latestckp', 'lqf.pth')
        torch.save(model_hqf.state_dict(), model_hqf_path)
        torch.save(model_lqf.state_dict(), model_lqf_path)
        print('Checkpoint saved to {}'.format(model_hqf_path))



    # 记录实验时间
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if local_rank == 0:
        logging.info('\n experiment in {}'.format(nowTime))

    lr_hqf = opt.lr
    lr_lqf = opt.lr
    best_psnr_hqf = 25.0
    best_psnr_lqf = 25.0
    # psnr, var = test()
    # w = [1.0 / 2.0, 1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0]
    w = [0.28, 0.17, 0.125, 0.16, 0.28]

    # if local_rank ==0 :
        # psnr_hqf, var_hqf, psnr_lqf, var_lqf = test()


    for epoch in range(1, opt.nEpochs + 1):
        if epoch % 10 == 0:
            w = np.sum([np.array(w), np.array([-0.02, -0.015, 0, 0.015, 0.02])], axis=0)
            w = w.tolist()
            w = [i if i > 0.03 else 0.03 for i in w]
            print('====> changing loss weights to', w)
            if local_rank == 0:
                logging.info(f'====> changing loss weights to {w}')

        criterion = MultiScaleLoss(weights = w)
        train_hqf(epoch)
        train_lqf(epoch)

        if local_rank == 0:

            logging.info('===> in {}th epochs'.format(epoch))

            psnr_hqf, var_hqf, psnr_lqf, var_lqf = test()

            scheduler_hqf.step(psnr_hqf)
            scheduler_lqf.step(psnr_lqf)

            if lr_hqf != optimizer_hqf.param_groups[0]['lr']:
                lr_hqf = optimizer_hqf.param_groups[0]['lr']
                logging.info('reducing lr of group 0 to {:.3e}'.format(lr_hqf))

            if lr_lqf != optimizer_lqf.param_groups[0]['lr']:
                lr_lqf = optimizer_lqf.param_groups[0]['lr']
                logging.info('reducing lr of group 0 to {:.3e}'.format(lr_lqf))


            if psnr_hqf > best_psnr_hqf:
                best_psnr_hqf = psnr_hqf
                logging.info('===> save the best Model: reach {:.2f}dB PSNR'.format(best_psnr_hqf))

            model_path = os.path.join('.', 'checkpoint_all', 'HQF_{:.2f}dB_{:.2f}.pth'.format(psnr_hqf, var_hqf))
            # torch.save(model, model_best_path)
            torch.save(model_hqf.state_dict(), model_path)


            if psnr_lqf > best_psnr_lqf:
                best_psnr_lqf = psnr_lqf
                logging.info('===> save the best Model: reach {:.2f}dB PSNR'.format(best_psnr_lqf))

            model_path = os.path.join('.', 'checkpoint_all', 'LQF_{:.2f}dB_{:.2f}.pth'.format(psnr_lqf, var_lqf))
            # torch.save(model, model_best_path)
            torch.save(model_lqf.state_dict(), model_path)


            # checkpoint(epoch)

