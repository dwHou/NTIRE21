import torch
from collections import OrderedDict
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
net_A = torch.load('e16_29.71dB_0.20.pth')

net_B = torch.load('e15_29.70dB_0.20.pth')

net_C = torch.load('e14_29.69dB_0.20.pth')

net_D = torch.load('e12_29.68dB_0.21.pth')

net_E = torch.load('e11_29.68dB_0.21.pth')

net_interp = OrderedDict()
for k, v_A in net_A.items():
    v_B = net_B[k]
    v_C = net_C[k]
    v_D = net_D[k]
    v_E = net_E[k]
    # net_interp[k] = 0.4 * v_D + 0.3 * v_C + 0.2 * v_B + 0.1 * v_A
    # net_interp[k] = 0.25 * v_D + 0.25 * v_C + 0.25 * v_B + 0.25 * v_A
    net_interp[k] = 0.25 * v_A + 0.25 * v_B + 0.2 * v_C + 0.15 * v_D + 0.15 * v_E
    # net_interp[k] = 0.7 * v_D + 0.3 * v_C
torch.save(net_interp, 'interp_2971.pth')
