import torch.utils.data as data
import torch
from util import common
import numpy as np
import pickle
import random
from option import opt
from PIL import Image
import lmdb

def get_training_set():
    train_txt = '/cfs_data/devonnhou/NTIRE2021/Dataset/FixedQP/train_pairs.txt'
    train_db = '/cfs_data/devonnhou/NTIRE2021/LMDB/train_lmdb'
    return ntire_dataset(train_txt, train_db, 'train')


def get_test_set():
    test_txt = '/cfs_data/devonnhou/NTIRE2021/Dataset/FixedQP/val_pairs.txt'
    test_db = '/cfs_data/devonnhou/NTIRE2021/LMDB/val_lmdb'
    return ntire_dataset(test_txt, test_db, 'test')



# NTIRE Dataset + Additional Dataset
class ntire_dataset(data.Dataset):
    def __init__(self, txt_path, db_path, flag):
        super(ntire_dataset, self).__init__()

        self.flag = flag

        env = lmdb.open(db_path)
        txn = env.begin()
        self.txn = txn

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
        return len(self.pairs)


    def _load_png(self, index):
        center, target, info = self.pairs[index]

        center_num = int(center[-8:-4])

        mod = (center_num - 1) % 4

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

        pairs = self.txn.get('{}'.format(input4).encode())
        np_pairs = pickle.loads(pairs)
        np_in_c, np_tar, np_info = np_pairs[0], np_pairs[1], np_pairs[2]

        if mod == 0:
            # pqf = np.array([1, 0.9, 0.9, 0.9, 1.1, 0.9, 0.9, 0.9, 1])
            input_l = input0
            input_r = input8

        elif mod == 1:
            input_l = input3
            input_r = input7


        elif mod == 2:
            input_l = input2
            input_r = input6


        elif mod == 3:
            input_l = input1
            input_r = input5

        else:
            pass

        pairs_l = self.txn.get('{}'.format(input_l).encode())
        if pairs_l == None:
            np_in_l = np_in_c.copy()
        else:
            np_pairs_l = pickle.loads(pairs_l)
            np_in_l = np_pairs_l[0]

        pairs_r = self.txn.get('{}'.format(input_r).encode())
        if pairs_r == None:
            np_in_r = np_in_c.copy()
        else:
            np_pairs_r = pickle.loads(pairs_r)
            np_in_r = np_pairs_r[0]

        list_in = [np_in_l, np_in_c, np_in_r]

        np_in = np.stack(list_in, axis=0)

        return np_in, np_tar, np_info[..., 0:1]
