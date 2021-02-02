# -*- coding:utf8 -*-
from __future__ import print_function
import datetime
import argparse
from math import log10
import os, fnmatch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.EMGA import EMGA
from model.PatchD import Discriminator
from option import opt
from tqdm import tqdm
import logging
from data.dataset import get_training_set, get_test_set
import time
from tensorboardX import SummaryWriter
import loss
from loss import L1_Charbonnier_loss
import lpips
import kornia
from PIL import Image

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'

    logging.basicConfig(filename='./LOG/' + 'GAN' + '.log', level=logging.INFO)
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
    netG = EMGA().to(device)
    netD = Discriminator().to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    if not (opt.pre_train is None):
        print('load model from %s ...' % opt.pre_train)
        netG.load_state_dict(torch.load(opt.pre_train))
        print('success!')


    criterion_p = loss.Loss(opt)
    criterion_c = L1_Charbonnier_loss()
    criterion_adv = nn.BCEWithLogitsLoss().to(device)

    lpips_fn_alex = lpips.LPIPS(net='alex').to(device)

    # criterion = Loss(opt)
    # criterion = nn.L1Loss()
    # criterion = MultiScaleLoss()
    # criterion = L1_Charbonnier_loss()
    # criterion_l2 = nn.MSELoss()
    # MSE = nn.MSELoss()

    optimizer_netG = optim.Adam(netG.parameters(), lr=opt.lr)
    optimizer_netD = optim.Adam(netD.parameters(), lr=opt.lr * 0.08)
    # lpips 不降，则减小学习率
    scheduler_netG = optim.lr_scheduler.ReduceLROnPlateau(optimizer_netG, 'min', factor=0.8, verbose=True, patience=5)
    scheduler_netD = optim.lr_scheduler.ReduceLROnPlateau(optimizer_netD, 'min', factor=0.8, verbose=True, patience=3)


    def train(epoch):
        epoch_loss_g = 0
        epoch_loss_d = 0
        for iteration, batch in enumerate(training_loader, 1):
            inputs, target, info, pqf = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)

            real_labels = torch.ones((target.size(0), 1, 4, 4)).to(device)
            fake_labels = torch.zeros((target.size(0), 1, 4, 4)).to(device)

            optimizer_netG.zero_grad()

            ##########################
            #   training generator   #
            ##########################

            # LR to HR
            i4, i3, i2, i1, i0 = netG(inputs, info, pqf)

            b, _, h4, w4 = i4.size()
            b, _, h3, w3 = i3.size()
            b, _, h2, w2 = i2.size()
            b, _, h1, w1 = i1.size()

            target_4 = F.interpolate(target, size=(h4, w4), mode='bilinear', align_corners=False)
            target_3 = F.interpolate(target, size=(h3, w3), mode='bilinear', align_corners=False)
            target_2 = F.interpolate(target, size=(h2, w2), mode='bilinear', align_corners=False)
            target_1 = F.interpolate(target, size=(h1, w1), mode='bilinear', align_corners=False)

            i0_lp = kornia.gaussian_blur2d(i0, (5, 5), (1.0, 1.0))
            i0_hp = i0 - i0_lp
            i0_hp = 0.5 + i0_hp * 0.5  
          
            target_lp = kornia.gaussian_blur2d(target, (5, 5), (1.0, 1.0))
            target_hp = target - target_lp
            target_hp = 0.5 + target_hp * 0.5

            loss_c1 = criterion_c(i4, target_4)
            loss_c2 = criterion_c(i3, target_3)
            loss_c3 = criterion_c(i2, target_2)
            loss_c4 = criterion_c(i1, target_1)
            loss_c5 = criterion_c(i0_lp, target_lp)
            loss_c6 = criterion_c(i0, target)
            loss_p = criterion_p(i0, target)



            score_real = netD(target_hp)
            score_fake = netD(i0_hp)
            discriminator_rf = score_real - score_fake.mean()
            discriminator_fr = score_fake - score_real.mean()
            adversarial_loss_rf = criterion_adv(discriminator_rf, fake_labels)
            adversarial_loss_fr = criterion_adv(discriminator_fr, real_labels)
            loss_adv = (adversarial_loss_fr + adversarial_loss_rf) / 2

            netG_loss = loss_c1 * 0.002 + loss_c2 * 0.004 + loss_c3 * 0.008 + loss_c4 * 0.01 + loss_c5 * 0.01 + loss_c6 * 0.001 + loss_p + loss_adv * 0.002


            epoch_loss_g += netG_loss.item()

            netG_loss.backward()
            optimizer_netG.step()

            print('===========> G_Epoch[{}]({}/{}): Loss: {:.6f}'.format(epoch, iteration, len(training_loader), netG_loss))

            ##########################
            # training discriminator #
            ##########################

            optimizer_netD.zero_grad()

            score_real = netD(target_hp)
            score_fake = netD(i0_hp.detach())
            discriminator_rf = score_real - score_fake.mean()
            discriminator_fr = score_fake - score_real.mean()
            adversarial_loss_rf = criterion_adv(discriminator_rf, real_labels)
            adversarial_loss_fr = criterion_adv(discriminator_fr, fake_labels)
            netD_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

            epoch_loss_d += netD_loss.item()
            netD_loss.backward()
            optimizer_netD.step()

            print('===> D_Epoch[{}]({}/{}): Loss: {:.6f}'.format(epoch, iteration, len(training_loader), netD_loss.item()))

        print('===> Epoch {} Complete: Avg. G_Loss: {:.6f}'.format(epoch, epoch_loss_g / len(training_loader)))
        print('===> Epoch {} Complete: Avg. D_Loss: {:.6f}'.format(epoch, epoch_loss_d / len(training_loader)))

        logging.info('Epoch Avg. GLoss : {:.6f}'.format(epoch_loss_g / len(training_loader)))
        logging.info('with DLoss : {:.6f}'.format(epoch_loss_d / len(training_loader)))


    def test():
        cost_time = 0
        avg_lpips = 0
        with torch.no_grad():
            for batch in tqdm(testing_loader):
                inputs, target, info, pqf = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                torch.cuda.synchronize()
                begin = time.time()

                prediction = netG(inputs, info, pqf)

                torch.cuda.synchronize()
                end = time.time()
                cost_time += end - begin

                inputs_lpips = inputs[:, 4, :, :, :] * 2 - 1
                pre_lpips = prediction[4] * 2 - 1
                target_lpips = target * 2 - 1
                lpips_score = lpips_fn_alex(pre_lpips, target_lpips).item()

                # 0, 1, 2, 3, 4, 5, 6
                # lpips_score = lpips_fn_alex(inputs_lpips, target_lpips).item()

                avg_lpips += lpips_score
                print('lpips : {:.3f}'.format(lpips_score))

            #   ---------------------------------------------------------------

        # wandb.log({"Avg. PSNR": avg_psnr / len(testing_data_loader)})

        print('===> Avg. LPIPS: {:.4f} , per frame use {} ms'.format( \
        avg_lpips / len(testing_loader) , cost_time * 1000 / len(testing_loader)))

        logging.info('Avg. LPIPS {:.4f} '.format(avg_lpips / len(testing_loader)))


        return avg_lpips / len(testing_loader)

    def visual(lpips, epoch=0):
        img_list = fnmatch.filter(os.listdir("./visual/inputs/288"), '*.png')
        img_list.sort(key=lambda x: int(x[:-4]))
        with torch.no_grad():
            list_mf = []
            for i in img_list:
                img_path = os.path.join("./visual/inputs/288", i)
                img = Image.open(img_path).convert('RGB')
                img = np.asarray(img)
                list_mf.append(img)

            inputs = np.stack(list_mf, axis=0)
            inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).float() / 255
            # B N C H W
            inputs = torch.unsqueeze(inputs, 0)
            inputs = inputs.cuda()

            info_path = os.path.join("./visual/inputs", "19.tuLayer.png")
            info = Image.open(info_path).convert('RGB')
            info = np.asarray(info)[..., 0:1]
            info = torch.from_numpy(info).permute(2, 0, 1).float() / 255
            info = torch.unsqueeze(info, 0)
            info = info.cuda()
            pqf = np.array([0.9, 1, 0.9, 0.9, 1, 1, 0.9, 0.9, 0.9])
            pqf = torch.from_numpy(pqf).float()
            pqf = torch.unsqueeze(pqf, 0)

            i4, i3, i2, i1, i0 = netG(inputs, info, pqf)

            out = torch.squeeze(i0, 0)
            out = out.cpu()
            out = out.detach().numpy() * 255.0
            out = out.clip(0, 255).transpose(1, 2, 0)
            out_img = Image.fromarray(out.astype(np.uint8), mode='RGB')

            output_path = os.path.join("./visual/outputs", f"e{epoch}_lpips{lpips:.4f}.png")
            out_img.save(output_path)

    def checkpoint(epoch):
        model_path = os.path.join('.', 'experiment', 'latestckp', 'latest_gan.pth')
        torch.save(netG.state_dict(), model_path)
        print('Checkpoint saved to {}'.format(model_path))



    # 记录实验时间
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info('\n experiment in {}'.format(nowTime))

    lr = opt.lr
    best_lpips = 0.4
    # lpips = test()
    # visual(lpips)

    for epoch in range(1, opt.nEpochs + 1):

        train(epoch)
        logging.info('===> in {}th epochs'.format(epoch))
        lpips = test()
        visual(lpips, epoch)

        scheduler_netG.step(lpips)
        scheduler_netD.step(lpips)

        if lr != optimizer_netG.param_groups[0]['lr']:
            lr = optimizer_netG.param_groups[0]['lr']
            logging.info('netG reducing lr of group 0 to {:.3e}'.format(lr))

        if lr != optimizer_netD.param_groups[0]['lr']:
            lr = optimizer_netD.param_groups[0]['lr']
            logging.info('netD reducing lr of group 0 to {:.3e}'.format(lr))

        if lpips <= best_lpips:
            best_lpips = lpips
            model_best_path = os.path.join('.', 'experiment', f'G_{lpips:.3f}.pth')
            logging.info(f'===> save the best model: reach {best_lpips:.3f} LPIPS')
            torch.save(netG.state_dict(), model_best_path)

            model_D = os.path.join('.', 'experiment', f'D_{lpips:.3f}.pth')
            torch.save(netD.state_dict(), model_D)

        checkpoint(epoch)
