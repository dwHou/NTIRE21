import torch
from collections import OrderedDict
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
net_ori = torch.load('e33_29.78dB_0.17.pth')


net_rename = OrderedDict()
for k, v_A in net_ori.items():
    k_new = k.replace("MPRNet", "ShannonNet")
    net_rename[k_new] = v_A
torch.save(net_rename, 'model.pth', _use_new_zipfile_serialization=False)
