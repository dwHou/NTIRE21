import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class mi_sequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input

class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=3, stride=1):
        super(ConvNorm, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x):
        out = self.conv(x)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=1, padding=padding, dilation=padding, bias=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, dilation=1, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = (self.conv2(out))
        out = x + out
        out = F.relu(out)
        return out

## Channel Attention (CA) Layer

class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        # B C 1 1
        y = self.conv_du(y)
        # CNN repalce FC
        return x * y

## Enhanced Spatial Attention (ESA) Layer

class SA(nn.Module):
    def __init__(self, n_feats):
        super(SA, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    # 和官方实现有歧义
    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        # v_max = self.pooling(c1)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

class ESA(nn.Module):
    def __init__(self, n_feats):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(2 * f, 2 * f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2 * f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, 2 * f, kernel_size=1)

        self.conv_info = nn.Conv2d(16, f, kernel_size=1)
        self.conv_cat = nn.Conv2d(2 * f, 2 * f, kernel_size=1)

        self.conv4_ = nn.Conv2d(2 * f, n_feats, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    # 和官方实现有歧义
    def forward(self, x, info_1):
        info_1 = self.relu(self.conv_info(info_1))
        c1_ = self.relu(self.conv1(x))
        c1_ = self.conv_cat(torch.cat( [c1_, info_1], dim=1))

        c1 = self.conv2(c1_)
        # v_max = self.pooling(c1)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        c3 = self.conv4(c3)
        cf = self.conv_f(c1_)
        c4 = self.relu(self.conv4_(c3 + cf))

        m = self.sigmoid(c4)

        return x * m, m

## Residual Block with ESA


class EHDC(nn.Module):
    __constants__ = ['branch1', 'branch2', 'branch3', 'branch5']

    def __init__(self, nf):
        super(EHDC, self).__init__()

        act = nn.LeakyReLU(0.01)

        self.branch1 = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(nf, nf // 8, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 8, nf * 3 // 16, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            act,
            nn.Conv2d(nf * 3 // 16, nf // 4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=5, bias=True, dilation=5)
        )

        self.ConvLinear = nn.Conv2d(nf, nf, kernel_size=1, stride=1)
        self.ShortPath = nn.Conv2d(nf, nf, kernel_size=1, stride=1)
        self.lrelu = nn.LeakyReLU(0.01, False)

        self.sa = ESA(nf)

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x, info_1):
        res = x
        res = self.lrelu(self.ShortPath(res))
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        outputs = self.lrelu(self.ConvLinear(outputs))

        outputs, mask = self.sa(outputs, info_1)

        rf = outputs
        outputs += res

        return outputs, rf

class RB(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, act=nn.LeakyReLU(0.01)):
        super(RB, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=1),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1),
            SA(out_feat)
        )

    def forward(self, x):
        res = x
        out = self.body(x)
        rf = out
        out += res
        return out, rf


class RB_last(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, act=nn.LeakyReLU(0.01)):
        super(RB_last, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size, stride=1),
            act,
            ConvNorm(out_feat, out_feat, kernel_size, stride=1),
            SA(out_feat)
        )

    def forward(self, x):
        # res = x
        out = self.body(x)
        # out += res
        return out


class ERB_last(nn.Module):
    __constants__ = ['branch1', 'branch2', 'branch3', 'branch5']

    def __init__(self, nf):
        super(ERB_last, self).__init__()

        act = nn.LeakyReLU(0.01)

        self.branch1 = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(nf, nf // 8, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 8, nf * 3 // 16, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            act,
            nn.Conv2d(nf * 3 // 16, nf // 4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=5, bias=True, dilation=5)
        )

        self.ConvLinear = nn.Conv2d(nf, nf, kernel_size=1, stride=1)
        self.ShortPath = nn.Conv2d(nf, nf, kernel_size=1, stride=1)
        self.lrelu = nn.LeakyReLU(0.01, False)

        self.sa = ESA(nf)

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x, info_1):
        # the inside of RB_last do not use residual learning
        # res = x
        # res = self.lrelu(self.ShortPath(res))
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        outputs = self.lrelu(self.ConvLinear(outputs))
        outputs, mask = self.sa(outputs, info_1)
        # outputs += res
        return outputs


## Residual Group (RG)
class BM(nn.Module):
    def __init__(self, n_feat, kernel_size):
        super(BM, self).__init__()

        self.RB1 = RB(n_feat, n_feat, kernel_size)

        self.RB2 = RB(n_feat, n_feat, kernel_size)

        self.RB3 = RB(n_feat, n_feat, kernel_size)

        self.RB4 = RB_last(n_feat, n_feat, kernel_size)

        self.Aggre = ConvNorm(n_feat * 4, n_feat, kernel_size=1, stride=1)

    def forward(self, x):
        out, rf1 = self.RB1(x)
        out, rf2 = self.RB2(out)
        out, rf3 = self.RB3(out)

        out = self.RB4(out)
        # B, C, H, W
        out = torch.cat([out, rf1, rf2, rf3], 1)
        out = self.Aggre(out)
        out += x
        return out


class EBM(nn.Module):
    def __init__(self, n_feat, kernel_size):
        super(EBM, self).__init__()

        self.nf = n_feat

        self.EHDC1 = EHDC(self.nf)

        self.EHDC2 = EHDC(self.nf)

        self.EHDC3 = EHDC(self.nf)

        self.RB4 = ERB_last(self.nf)

        self.act = nn.LeakyReLU(0.01, inplace=True)

        # self.sft0 = SFTLayer_torch()

        # self.sft1 = SFTLayer_torch()

        self.Aggre = ConvNorm(n_feat * 4, n_feat, kernel_size=1, stride=1)

    def forward(self, x, info_1):
        # out = self.act(self.sft0(x, info_0))
        out, rf1 = self.EHDC1(x, info_1)
        out, rf2 = self.EHDC2(out, info_1)
        # out = self.act(self.sft1(x, info_0))
        out, rf3 = self.EHDC3(out, info_1)
        out = self.RB4(out, info_1)
        # B, C, H, W
        # print(out.shape, rf1.shape, rf2.shape, rf3.shape)
        out = torch.cat([out, rf1, rf2, rf3], 1)
        out = self.Aggre(out)
        out += x
        return out, info_1


# RIR : n_basemodules = 30
class RIR(nn.Module):
    def __init__(self, n_basemodules, n_feats):
        super(RIR, self).__init__()

        self.headConv = ConvNorm(n_feats, n_feats, kernel_size=3, stride=1)
        # define modules: head, body, tail
        modules_body = [
            BM(
                n_feat=n_feats,
                kernel_size=3)
            for _ in range(n_basemodules)]
        self.body = nn.Sequential(*modules_body)

        self.tailConv = ConvNorm(n_feats, n_feats, kernel_size=3, stride=1)

    def forward(self, x):
        # Build input tensor B, C, H, W
        x = self.headConv(x)
        res = self.body(x)
        res += x
        out = self.tailConv(res)
        return out

class ERIR(nn.Module):
    def __init__(self, n_basemodules, n_feats):
        super(ERIR, self).__init__()

        self.headConv = ConvNorm(n_feats, n_feats, kernel_size=3, stride=1)
        # define modules: head, body, tail
        modules_body = [
            EBM(
                n_feat=n_feats,
                kernel_size=3)
            for _ in range(n_basemodules)]
        self.body = mi_sequential(*modules_body)

        self.tailConv = ConvNorm(n_feats, n_feats, kernel_size=3, stride=1)

    def forward(self, x, info_1):
        # Build input tensor B, C, H, W
        x = self.headConv(x)
        res, _ = self.body(x, info_1)
        res += x
        out = self.tailConv(res)
        return out


class DiffRefiner(nn.Module):
    __constants__ = ['branch1', 'branch2', 'branch3', 'branch5']

    def __init__(self, nf):
        super(DiffRefiner, self).__init__()

        act = nn.LeakyReLU(0.01)

        self.branch1 = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(nf, nf // 8, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 8, nf * 3 // 16, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            act,
            nn.Conv2d(nf * 3 // 16, nf // 4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=5, bias=True, dilation=5)
        )

        self.ConvLinear = nn.Conv2d(nf, nf, kernel_size=1, stride=1)
        self.ShortPath = nn.Conv2d(nf, nf, kernel_size=1, stride=1)
        self.lrelu = nn.LeakyReLU(0.01, False)

        self.sa = SA(nf)

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        res = x
        res = self.lrelu(self.ShortPath(res))
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        outputs = self.lrelu(self.ConvLinear(outputs))

        outputs = self.sa(outputs)
        outputs += res

        return outputs

class EHDCwoSA(nn.Module):
    __constants__ = ['branch1', 'branch2', 'branch3', 'branch5']

    def __init__(self, nf):
        super(EHDCwoSA, self).__init__()

        act = nn.LeakyReLU(0.01)

        self.branch1 = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(nf, nf // 4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(nf, nf // 8, kernel_size=1, stride=1),
            act,
            nn.Conv2d(nf // 8, nf * 3 // 16, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            act,
            nn.Conv2d(nf * 3 // 16, nf // 4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            act,
            nn.Conv2d(nf // 4, nf // 4, kernel_size=3, stride=1, padding=5, bias=True, dilation=5)
        )

        self.ConvLinear = nn.Conv2d(nf, nf, kernel_size=1, stride=1)
        self.ShortPath = nn.Conv2d(nf, nf, kernel_size=1, stride=1)
        self.lrelu = nn.LeakyReLU(0.01, False)

        self.ca = CALayer(nf)

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        res = x
        res = self.lrelu(self.ShortPath(res))
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)

        outputs = self.lrelu(self.ConvLinear(outputs))
        outputs = self.ca(outputs)
        outputs += res

        return outputs


