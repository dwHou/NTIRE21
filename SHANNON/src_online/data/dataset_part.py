import torch.utils.data as data
import torch
from util import common
import numpy as np
import pickle
import random
from option import opt
from PIL import Image


def get_training_010():
    train_txt = '/dockerdata/devonn/NTIRE2021/train_010.txt'
    return ntire_dataset(train_txt, 'train')


def get_test_010():
    test_txt = '/dockerdata/devonn/NTIRE2021/val_010.txt'
    return ntire_dataset(test_txt, 'test')

def get_training_hqf():
    train_txt = '/dockerdata/devonn/NTIRE2021/''train_hqf_pairs.txt'
    return ntire_dataset(train_txt, 'train')


def get_test_hqf():
    test_txt = '/dockerdata/devonn/NTIRE2021/''val_hqf_pairs.txt'
    return ntire_dataset(test_txt, 'test')

def get_training_lqf():
    train_txt = '/dockerdata/devonn/NTIRE2021/''train_lqf_pairs.txt'
    return ntire_dataset(train_txt, 'train')


def get_test_lqf():
    test_txt = '/dockerdata/devonn/NTIRE2021/''val_lqf_pairs.txt'
    return ntire_dataset(test_txt, 'test')


# NTIRE Dataset + Additional Dataset
class ntire_dataset(data.Dataset):
    def __init__(self, txt_path, flag):
        super(ntire_dataset, self).__init__()

        self.flag = flag

        with open(txt_path, 'r') as fh:
            pairs = []

            for line in fh:
                line = line.rstrip()
                words = line.split()
                pairs.append((words[0], words[1], words[2]))
            self.pairs = pairs



    def __getitem__(self, index):
        self.index = index
        np_in, np_tar, np_info = self._load_png(self.index)

        if self.flag == 'train':
            self.patch_size = opt.patchSize
            patch_in, patch_tar, patch_info = common.get_patch(np_in, np_tar, np_info, opt.patchSize)
            # data augment
            patch_in, patch_tar, patch_info = common.augment([patch_in, patch_tar, patch_info])

        elif self.flag == 'test':
            patch_in, patch_tar, patch_info = np_in, np_tar, np_info

        patch_in, patch_tar, patch_info = common.np2Tensor([patch_in, patch_tar, patch_info], 255)

        # in: B5C=3HW tar: B3C=3HW info: BC=1HW mode: torch.tensor(0.) or (1.)
        return patch_in, patch_tar, patch_info


    def __len__(self):
        self.length= len(self.pairs)
        if self.flag == 'train':
            return self.length // 3  
        elif self.flag == 'test':
            return self.length


    def _load_png(self, index):
        if self.flag == 'train':
            center, target, info = self.pairs[index * 3]
        elif self.flag == 'test':
            center, target, info = self.pairs[index]

        center_num = int(center[-8:-4])

        mod = (center_num - 1) % 4

        idx_s = info.find('info') + 9
        idx_e = info.find('.tuLayer')

        input0_ = str(10000 + center_num - 4)[1:]
        input1_ = str(10000 + center_num - 3)[1:]
        input2_ = str(10000 + center_num - 2)[1:]
        input3_ = str(10000 + center_num - 1)[1:]
        input4_ = str(10000 + center_num)[1:]
        input5_ = str(10000 + center_num + 1)[1:]
        input6_ = str(10000 + center_num + 2)[1:]
        input7_ = str(10000 + center_num + 3)[1:]
        input8_ = str(10000 + center_num + 4)[1:]

        input0 = center.replace(center[-8:-4], input0_)
        input1 = center.replace(center[-8:-4], input1_)
        input2 = center.replace(center[-8:-4], input2_)
        input3 = center.replace(center[-8:-4], input3_)
        input4 = center.replace(center[-8:-4], input4_)
        input5 = center.replace(center[-8:-4], input5_)
        input6 = center.replace(center[-8:-4], input6_)
        input7 = center.replace(center[-8:-4], input7_)
        input8 = center.replace(center[-8:-4], input8_)


        info3 = info[:idx_s] + str(int(info[idx_s:idx_e]) - 1) + info[idx_e:]
        info4 = info
        info5 = info[:idx_s] + str(int(info[idx_s:idx_e]) + 1) + info[idx_e:]

        '''
        info3 = info.replace(info[idx_s:idx_e], str(int(info[idx_s:idx_e]) - 1))
        info4 = info
        info5 = info.replace(info[idx_s:idx_e], str(int(info[idx_s:idx_e]) + 1))

        info3_str = info3
        info4_str = info4
        info5_str = info5
        '''
        # input (0 1 2 3) 4 (5 6 7 8)

        input4 = Image.open('/dockerdata/devonn/NTIRE2021/' + input4).convert('RGB')
        info4 = Image.open('/dockerdata/devonn/NTIRE2021/training_fixed-QP/' + info4).convert('RGB')

        # left
        try:
            input3 = Image.open('/dockerdata/devonn/NTIRE2021/' + input3).convert('RGB')
            # info3 = Image.open('/dockerdata/devonn/NTIRE2021/''training_fixed-QP/' + info3).convert('RGB')

            try:
                input2 = Image.open('/dockerdata/devonn/NTIRE2021/' + input2).convert('RGB')
                try:
                    input1 = Image.open('/dockerdata/devonn/NTIRE2021/' + input1).convert('RGB')
                    try:
                        input0 = Image.open('/dockerdata/devonn/NTIRE2021/' + input0).convert('RGB')
                    except Exception:
                        input0 = input1

                except Exception:
                    input1 = input2
                    input0 = input1

            except Exception:
                input2 = input3
                input1 = input2
                input0 = input1

        except Exception:
            input3 = input4
            # info3 = info4
            input2 = input3
            input1 = input2
            input0 = input1


        # right
        try:
            input5 = Image.open('/dockerdata/devonn/NTIRE2021/' + input5).convert('RGB')
            # info5 = Image.open('/dockerdata/devonn/NTIRE2021/''training_fixed-QP/' + info5).convert('RGB')

            try:
                input6 = Image.open('/dockerdata/devonn/NTIRE2021/' + input6).convert('RGB')
                try:
                    input7 = Image.open('/dockerdata/devonn/NTIRE2021/' + input7).convert('RGB')
                    try:
                        input8 = Image.open('/dockerdata/devonn/NTIRE2021/' + input8).convert('RGB')
                    except Exception:
                        input8 = input7

                except Exception:
                    input7 = input6
                    input8 = input7

            except Exception:
                input6 = input5
                input7 = input6
                input8 = input7

        except Exception:
            input5 = input4
            # info5 = info4
            input6 = input5
            input7 = input6
            input8 = input7


        if mod == 0:
            # pqf = np.array([1, 0.9, 0.9, 0.9, 1.1, 0.9, 0.9, 0.9, 1])
            # list_in = [np.asarray(input0), np.asarray(input4), np.asarray(input8)]
            pass
        elif mod == 1:
            pqf = np.array([0.9, 0.9, 0.9, 1, 1, 0.9, 0.9, 1, 0.9])
            list_in = [np.asarray(input3), np.asarray(input4), np.asarray(input7)]

            # pqf_l = input3
            # pqf_r = input7

        elif mod == 2:
            # pqf = np.array([0.9, 0.9, 1, 0.9, 1, 0.9, 1, 0.9, 0.9])
            # list_in = [np.asarray(input2), np.asarray(input4), np.asarray(input6)]
            pass
            # pqf_l = input2
            # pqf_r = input6

        elif mod == 3:
            pqf = np.array([0.9, 1, 0.9, 0.9, 1, 1, 0.9, 0.9, 0.9])
            list_in = [np.asarray(input5), np.asarray(input4), np.asarray(input1)]

            # pqf_l = input1
            # pqf_r = input5

        else:
            pass


        img_tar = Image.open('/dockerdata/devonn/NTIRE2021/' + target).convert('RGB')

        # list_in = [np.asarray(input0), np.asarray(input1), np.asarray(input2), np.asarray(input3), np.asarray(input4), np.asarray(input5), np.asarray(input6), np.asarray(input7), np.asarray(input8)]

        np_info = np.asarray(info4)[..., 0:1]

        np_in = np.stack(list_in, axis=0)

        # torch_pqf = torch.from_numpy(pqf).float()

        np_tar = np.asarray(img_tar)
        # 9 3 H W; 3 H W; 3 1 H W; 2 3 H W
        return np_in, np_tar, np_info
