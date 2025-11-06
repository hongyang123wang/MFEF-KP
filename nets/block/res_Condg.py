from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x  # [B,C,H,W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)


class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class Upsample(nn.Module):
    """ 最近邻上采样 2×，再接 3×3 卷积 """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)  # 3×3 消除混叠
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class AuxiliaryHead(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        # 前景：16×16 → 32×32 → 64×64 → 128×128
        self.stem_fg = nn.Sequential(
            Upsample(in_c, 64),
            Upsample(64, 32),
            Upsample(32, 32),
        )

        self.head_fg = nn.Conv2d(32, 1, 1)  # 128×128

    def forward(self, f_fg):
        h_fg = self.stem_fg(f_fg)  # [B,32,128,128]
        hm_fg = self.head_fg(h_fg)  # [B,1,128,128]
        return hm_fg


class UpAndAdd(nn.Module):
    def __init__(self, in_c_low, in_c_high, out_c):
        """
        将低分辨率特征图 in_c_low 上采样 2 倍，
        与高分辨率特征图 in_c_high 逐元素相加，
        并用 1×1 卷积统一通道数到 out_c。
        """
        super().__init__()
        # 转置卷积：stride=2 即可放大 2 倍
        self.up = nn.ConvTranspose2d(in_c_low, in_c_high,
                                     kernel_size=4, stride=2, padding=1,
                                     bias=False)
        # 如果两路通道不同，再做一次 1×1 融合
        if in_c_high != out_c:
            self.fuse = nn.Conv2d(in_c_high, out_c, 1, bias=False)
        else:
            self.fuse = nn.Identity()

    def forward(self, x_high, x_low):
        # x_high: 32×32×512, x_low: 16×16×1024
        x_up = self.up(x_low)  # 16×16×1024 -> 32×32×512
        x_up = x_up + x_high  # 逐元素相加
        return self.fuse(x_up)  # 32×32×out_c


class DecoupleLayer(nn.Module):
    def __init__(self, in_c=1024, out_c=128):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            CBR(256, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
    def forward(self, x):
        f_fg = self.cbr_fg(x)
        return f_fg




class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, scale=2):
        super().__init__()
        self.scale = scale
        self.relu = nn.ReLU()

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.c1 = CBR(in_c + out_c, out_c, kernel_size=1, padding=0)
        self.c2 = CBR(out_c, out_c, act=False)
        self.c3 = CBR(out_c, out_c, act=False)
        self.c4 = CBR(out_c, out_c, kernel_size=1, padding=0, act=False)
        self.ca = channel_attention(out_c)
        self.sa = spatial_attention()

    def _save_sfd_compare(self, plain, enhance, prefix, indx=0):
        import os, matplotlib.pyplot as plt
        os.makedirs(prefix, exist_ok=True)
        B = plain.size(0)
        for i in range(min(B, 3)):  # 最多存 3 张
            p = plain[i].mean(dim=0).cpu()
            e = enhance[i].mean(dim=0).cpu()

            # 归一化处理，保留相对关系
            p = (p - p.min()) / (p.max() - p.min() + 1e-8)
            e = (e - e.min()) / (e.max() - e.min() + 1e-8)

            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            ax[0].imshow(p, cmap='jet')
            ax[0].set_title('w/o SFD')
            ax[0].axis('off')
            ax[1].imshow(e, cmap='jet')
            ax[1].set_title('w/ SFD')
            ax[1].axis('off')
            plt.tight_layout()
            plt.savefig(f'{prefix}/sample{indx}_{i}.png', dpi=300)  # 添加索引避免覆盖
            plt.close()

    def forward(self, x, skip, draw=False):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)

        x = self.c1(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)
        s3 = x

        x = self.c4(x)
        x = self.relu(x + s3 + s2 + s1)

        x1 = self.ca(x)
        x = self.sa(x1)
        return x


class output_block(nn.Module):

    def __init__(self, in_c, out_c=64):
        super().__init__()

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.fuse = CBR(in_c * 7, 256, kernel_size=3, padding=1)
        # self.c1 = CBR(512, 256, kernel_size=3, padding=1)
        self.c2 = CBR(256, 128, kernel_size=1, padding=0)
        self.c3 = nn.Conv2d(128, out_c, kernel_size=1, padding=0)
        # self.sig = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        x2 = self.up_2x2(x2)
        x3 = self.up_4x4(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fuse(x)

        # x = self.up_2x2(x)
        # x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        # x = self.sig(x)
        return x
