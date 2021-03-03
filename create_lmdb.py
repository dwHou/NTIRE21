#!/usr/bin/env python

import os
import numpy as np
import lmdb
from PIL import Image
from tqdm import tqdm
# import msgpack
import pickle

# conda install -c conda-forge python-lmdb

# https://www.jianshu.com/p/694b8639f42f
# 廖雪峰 https://www.liaoxuefeng.com/wiki/1016959663602400/1017624706151424
# 注意dumps与dump区别
# png是无损压缩过的 LMDB不经过压缩 而且还添加了序列化的数据结构

'''
    为数据集生成对应的lmdb存储文件
'''

train_db_path = os.path.join('LMDB', 'train_lmdb')
train_txt = '/cfs_data/devonnhou/NTIRE2021/Dataset/FixedQP/train_pairs.txt'

valid_db_path = os.path.join('LMDB', 'val_lmdb')
val_txt = '/cfs_data/devonnhou/NTIRE2021/Dataset/FixedQP/val_pairs.txt'


def create_lmdb(env_path, txt_path):
    env = lmdb.open(env_path, map_size=1099511627776)  # 1024^4 即1TB
    txn = env.begin(write=True)


    with open(txt_path, 'r') as fh:
        pairs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            pairs.append((words[0], words[1], words[2]))

    # key =0 # key = 0 开始更好，能和__getitem__的index对应上。
    # 这段逻辑同gen_txt.py
    for index in range(len(pairs)):
        input, target, info = pairs[index]
        
        key = info
        
        img_in = Image.open('/cfs_data/devonnhou/NTIRE2021/Dataset/FixedQP/' + input).convert('RGB')
        img_tar = Image.open('/cfs_data/devonnhou/NTIRE2021/Dataset/Label/training_raw/' + target).convert('RGB')
        img_info = Image.open('/cfs_data/devonnhou/NTIRE2021/Dataset/FixedQP/training_fixed-QP/' + info).convert('RGB')

        np_in = np.asarray(img_in)
        np_tar = np.asarray(img_tar)
        np_info = np.asarray(img_info)
        # print(np_in.shape, np_tar.shape, np_info.shape)
        np_pair = np.array([np_in, np_tar, np_info])

        pair_b = pickle.dumps(np_pair)

        txn.put('{}'.format(key).encode(), pair_b)
        
        if index % 500 == 0:
            print("===>commit interval")
            txn.commit()
            # commit 之后需要再次 begin
            txn = env.begin(write=True)

    txn.commit()
    txn = env.begin()

    print(txn.stat()['entries'])
    env.close()
    print('Finish writing lmdb.')


if __name__ == "__main__":
    create_lmdb(train_db_path, train_txt)
    create_lmdb(valid_db_path, val_txt)
