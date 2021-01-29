#!/usr/bin/env python

from __future__ import print_function
import os, fnmatch
import argparse
import torch
import torch.nn as nn
from PIL import Image
from util import common
import numpy as np
import time
from tqdm import tqdm
import math
from model.PTLSTM import PTLSTM
from contextlib import contextmanager

# solver settings
parser = argparse.ArgumentParser(description='PyTorch NTIRE Solver')
parser.add_argument('--input_video', type=str, default='/cfs_data/devonnhou/NTIRE2021/Dataset/FixedQP/valuation_fixed-QP/fixedqp_png/002',
                    help='input video to use')
parser.add_argument('--input_info', type=str, default='/cfs_data/devonnhou/NTIRE2021/Dataset/FixedQP/valuation_fixed-QP/fixedqp_info/002',
                    help='side information to use')
parser.add_argument('--ckp', type=str, default='./experiment/latestckp/PTNet_31.35dB_version1.4.0.pth', help='model file to use')
parser.add_argument('--output_video', type=str, default='/cfs_data/devonnhou/NTIRE2021/Dataset/FixedQP/valuation_fixed-QP/ptnet_out/002',
                    help='where to save the output video')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--ensemble', action='store_true', help='use self-ensemble method for test')
# parser.add_argument('--sampling', action='store_true', help='results with uniform sampling each 10th frame.')
args = parser.parse_args()
print(args)


@contextmanager
def timer(name):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    yield
    end.record()

    torch.cuda.synchronize()
    print(f'[{name}] done in {start.elapsed_time(end):.3f} ms')
    print('per frame consume {} ms'.format(start.elapsed_time(end)))


def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        multi_frames = len(img.shape) == 4
        if multi_frames:
            if hflip: img = img[:, :, ::-1, :]
            if vflip: img = img[:, ::-1, :, :]
            if rot90: img = img.transpose(0, 2, 1, 3)
        else:
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(_l) for _l in l]


class Evaluator():
    def __init__(self, args, my_model):
        self.args = args
        self.ckp = args.ckp
        self.video_i = args.input_video
        self.info = args.input_info
        self.video_o = args.output_video
        self.model = my_model

    def mf_test(self):
        img_list = fnmatch.filter(os.listdir(self.video_i), '*.png')
        img_list.sort(key=lambda x: int(x[:-4]))
        # avg_psnr = 0
        with torch.no_grad():
            for index in tqdm(range(len(img_list))):
                # if ((index + 1) % 10):
                  #  continue

                list_mf = []
                for i in [index - 3 if (index - 3) > 0 else 0,
                          index - 2 if (index - 2) > 0 else 0,
                          index - 1 if (index - 1) > 0 else 0,
                          index,
                          index + 1 if (index + 1) < len(img_list) else len(img_list) - 1,
                          index + 2 if (index + 2) < len(img_list) else len(img_list) - 1,
                          index + 3 if (index + 3) < len(img_list) else len(img_list) - 1]:

                    img_path = os.path.join(self.video_i, img_list[i])
                    img = Image.open(img_path).convert('RGB')
                    img = np.asarray(img)
                    list_mf.append(img)

                input = np.stack(list_mf, axis=0)
                # print(input.shape)
                # N H W C
                # print(torch.from_numpy(img).shape)
                input = torch.from_numpy(input).permute(0, 3, 1, 2).float() / 255
                # B N C H W
                input = torch.unsqueeze(input, 0)
                input = input.cuda()

                info_filename = str(int(img_list[index].split('.')[0]) - 1) + '.tuLayer.png'
                info_path = os.path.join(self.info, info_filename)
                info = Image.open(info_path).convert('RGB')
                info = np.asarray(info)[..., 0:1]
                info = torch.from_numpy(info).permute(2, 0, 1).float() / 255
                info = torch.unsqueeze(info, 0)
                info = info.cuda()

                model = self.model

                with timer('PTLSTM'):
                    out = model(input, info)

                out = torch.squeeze(out, 0)
                out = out.cpu()
                out = out.detach().numpy() * 255.0
                out = out.clip(0, 255).transpose(1, 2, 0)
                out_img = Image.fromarray(out.astype(np.uint8), mode='RGB')

                output_path = os.path.join(self.video_o, img_list[index])
                out_img.save(output_path)


    def mf_ensemble_test(self):
        img_list = fnmatch.filter(os.listdir(self.video_i), '*.png')
        img_list.sort(key=lambda x: int(x[:-4]))
        with torch.no_grad():
            for index in tqdm(range(len(img_list))):
                if ((index + 1) % 10):
                    continue

                list_mf = []
                for i in [index - 3 if (index - 3) > 0 else 0,
                          index - 2 if (index - 2) > 0 else 0,
                          index - 1 if (index - 1) > 0 else 0,
                          index,
                          index + 1 if (index + 1) < len(img_list) else len(img_list) - 1,
                          index + 2 if (index + 2) < len(img_list) else len(img_list) - 1,
                          index + 3 if (index + 3) < len(img_list) else len(img_list) - 1]:

                    img_path = os.path.join(self.video_i, img_list[i])
                    img = Image.open(img_path).convert('RGB')
                    img = np.asarray(img)
                    list_mf.append(img)

                info_filename = str(int(img_list[index].split('.')[0]) - 1) + '.tuLayer.png'
                info_path = os.path.join(self.info, info_filename)
                info = Image.open(info_path).convert('RGB')
                info = np.asarray(info)[..., 0:1]

                input_root = np.stack(list_mf, axis=0)
                info_root = info
                # rot ?
                input_1F = np.ascontiguousarray(input_root)
                input_1T = np.ascontiguousarray(input_root.transpose(0, 2, 1, 3))

                info_1F = np.ascontiguousarray(info_root)
                info_1T = np.ascontiguousarray(info_root.transpose(1, 0, 2))

                # rot_F hflip ? vflip ?
                input_1F_2F = input_1F
                input_1F_2F_3F = input_1F_2F
                input_1F_2F_3T = np.ascontiguousarray(input_1F_2F[:, ::-1, :, :])

                input_1F_2T = np.ascontiguousarray(input_1F[:, :, ::-1, :])
                input_1F_2T_3F = input_1F_2T
                input_1F_2T_3T = np.ascontiguousarray(input_1F_2T[:, ::-1, :, :])

                info_1F_2F = info_1F
                info_1F_2F_3F = info_1F_2F
                info_1F_2F_3T = np.ascontiguousarray(info_1F_2F[::-1, :, :])

                info_1F_2T = np.ascontiguousarray(info_1F[:, ::-1, :])
                info_1F_2T_3F = info_1F_2T
                info_1F_2T_3T = np.ascontiguousarray(info_1F_2T[::-1, :, :])



                # rot_T hflip ? vflip ?
                input_1T_2F = input_1T
                input_1T_2F_3F = input_1T_2F
                input_1T_2F_3T = np.ascontiguousarray(input_1T_2F[:, ::-1, :, :])

                input_1T_2T = np.ascontiguousarray(input_1T[:, :, ::-1, :])
                input_1T_2T_3F = input_1T_2T
                input_1T_2T_3T = np.ascontiguousarray(input_1T_2T[:, ::-1, :, :])

                info_1T_2F = info_1T
                info_1T_2F_3F = info_1T_2F
                info_1T_2F_3T = np.ascontiguousarray(info_1T_2F[::-1, :, :])

                info_1T_2T = np.ascontiguousarray(info_1T[:, ::-1, :])
                info_1T_2T_3F = info_1T_2T
                info_1T_2T_3T = np.ascontiguousarray(info_1T_2T[::-1, :, :])


                # print(input.shape)
                # N H W C
                # print(torch.from_numpy(img).shape)

                input_1F_2F_3F = torch.from_numpy(input_1F_2F_3F).permute(0, 3, 1, 2).float() / 255
                input_1F_2F_3T = torch.from_numpy(input_1F_2F_3T).permute(0, 3, 1, 2).float() / 255
                input_1F_2T_3F = torch.from_numpy(input_1F_2T_3F).permute(0, 3, 1, 2).float() / 255
                input_1F_2T_3T = torch.from_numpy(input_1F_2T_3T).permute(0, 3, 1, 2).float() / 255

                input_1T_2F_3F = torch.from_numpy(input_1T_2F_3F).permute(0, 3, 1, 2).float() / 255
                input_1T_2F_3T = torch.from_numpy(input_1T_2F_3T).permute(0, 3, 1, 2).float() / 255
                input_1T_2T_3F = torch.from_numpy(input_1T_2T_3F).permute(0, 3, 1, 2).float() / 255
                input_1T_2T_3T = torch.from_numpy(input_1T_2T_3T).permute(0, 3, 1, 2).float() / 255


                info_1F_2F_3F = torch.from_numpy(info_1F_2F_3F).permute(2, 0, 1).float() / 255
                info_1F_2F_3T = torch.from_numpy(info_1F_2F_3T).permute(2, 0, 1).float() / 255
                info_1F_2T_3F = torch.from_numpy(info_1F_2T_3F).permute(2, 0, 1).float() / 255
                info_1F_2T_3T = torch.from_numpy(info_1F_2T_3T).permute(2, 0, 1).float() / 255

                info_1T_2F_3F = torch.from_numpy(info_1T_2F_3F).permute(2, 0, 1).float() / 255
                info_1T_2F_3T = torch.from_numpy(info_1T_2F_3T).permute(2, 0, 1).float() / 255
                info_1T_2T_3F = torch.from_numpy(info_1T_2T_3F).permute(2, 0, 1).float() / 255
                info_1T_2T_3T = torch.from_numpy(info_1T_2T_3T).permute(2, 0, 1).float() / 255


                # B N C H W
                input_norot = [torch.unsqueeze(input_1F_2F_3F, 0), torch.unsqueeze(input_1F_2F_3T, 0),
                               torch.unsqueeze(input_1F_2T_3F, 0), torch.unsqueeze(input_1F_2T_3T, 0)]
                input_rot = [torch.unsqueeze(input_1T_2F_3F, 0), torch.unsqueeze(input_1T_2F_3T, 0),
                             torch.unsqueeze(input_1T_2T_3F, 0), torch.unsqueeze(input_1T_2T_3T, 0)]

                input_norot = torch.cat(input_norot, 0).cuda()

                input_rot = torch.cat(input_rot, 0).cuda()


                info_norot = [torch.unsqueeze(info_1F_2F_3F, 0), torch.unsqueeze(info_1F_2F_3T, 0),
                               torch.unsqueeze(info_1F_2T_3F, 0), torch.unsqueeze(info_1F_2T_3T, 0)]
                info_rot = [torch.unsqueeze(info_1T_2F_3F, 0), torch.unsqueeze(info_1T_2F_3T, 0),
                             torch.unsqueeze(info_1T_2T_3F, 0), torch.unsqueeze(info_1T_2T_3T, 0)]

                info_norot = torch.cat(info_norot, 0).cuda()

                info_rot = torch.cat(info_rot, 0).cuda()

                model = self.model

                with timer('PTLSTM'):
                    # 4, C, H, W
                    out = model(input_norot, info_norot)
                    out_rot = model(input_rot, info_rot)

                out_0, out_1, out_2, out_3 = out[0], out[1], out[2], out[3]
                out_4, out_5, out_6, out_7 = out_rot[0], out_rot[1], out_rot[2], out_rot[3]

                out_x4 = out_0 + torch.flip(out_1, [1, ]) + torch.flip(out_2, [2, ]) + torch.flip(out_3, [1, 2])

                out_rot_x4 = out_4 + torch.flip(out_5, [1, ]) + torch.flip(out_6, [2, ]) + torch.flip(out_7, [1, 2])
                out_rot_x4 = out_rot_x4.permute(0, 2, 1)  # 注意顺序，input先rot，output则最后rot.

                out = (out_x4 + out_rot_x4) / 8.0

                out = out.cpu()
                out = out.detach().numpy() * 255.0
                out = out.clip(0, 255).transpose(1, 2, 0)
                out_img = Image.fromarray(out.astype(np.uint8), mode='RGB')

                output_path = os.path.join(self.video_o, img_list[index])
                out_img.save(output_path)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
    print('===> Loading pretrained model')
    device = torch.device("cuda")

    model = PTLSTM().to(device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.ckp).items()})
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    t = Evaluator(args, model)
    if args.ensemble:
        t.mf_ensemble_test()
    else:
        t.mf_test()
