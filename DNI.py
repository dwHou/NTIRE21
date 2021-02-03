import torch
from collections import OrderedDict

alpha = 0.3
net_A = torch.load('./M_29.38dB_version1.4.0.pth')
net_B = torch.load('G_0.106.pth')

net_interp = OrderedDict()
for k, v_A in net_A.items():
    v_B = net_B[k]
    net_interp[k] = alpha * v_A + (1 - alpha) * v_B
torch.save(net_interp, './interp.pth')
