from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import torch.nn as nn

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


#
# class CoordAtt(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super().__init__()
#         self.h_avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # 水平方向池化 [B,C,H,1]
#         self.w_avg_pool = nn.AdaptiveAvgPool2d((1, None))  # 垂直方向池化 [B,C,1,W]
#
#         # 共享的1x1卷积（替代全连接层）
#         mid_channels = max(in_channels // reduction, 4)  # 确保最小通道数
#         self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
#         self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
#         self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # X方向注意力
#         x_h = self.h_avg_pool(x)  # [B,C,H,1]
#         x_h = self.conv1(x_h)  # [B,C//r,H,1]
#         x_h = self.conv_h(x_h)  # [B,C,H,1]
#
#         # Y方向注意力
#         x_w = self.w_avg_pool(x)  # [B,C,1,W]
#         x_w = self.conv1(x_w)  # [B,C//r,1,W]
#         x_w = self.conv_w(x_w)  # [B,C,1,W]
#
#         # 融合并生成注意力权重（直接相加）
#         attn = self.sigmoid(x_h + x_w)  # [B,C,H,W]
#         return x * attn
#
#
# class MonaDepthAdapter(nn.Module):
#     def __init__(self, in_channels, reduction=4):
#         super().__init__()
#         # 多尺度深度卷积组
#         self.dwconv3 = nn.Conv2d(in_channels // reduction, in_channels // reduction,
#                                  kernel_size=3, padding=1, groups=in_channels // reduction)
#         self.dwconv5 = nn.Conv2d(in_channels // reduction, in_channels // reduction,
#                                  kernel_size=5, padding=2, groups=in_channels // reduction)
#         self.dwconv7 = nn.Conv2d(in_channels // reduction, in_channels // reduction,
#                                  kernel_size=7, padding=3, groups=in_channels // reduction)
#
#         # 动态归一化
#         self.norm = nn.LayerNorm(in_channels)
#         self.scale = nn.Parameter(torch.ones(1))
#
#         # 投影层
#         self.down = nn.Linear(in_channels, in_channels // reduction)
#         self.up = nn.Linear(in_channels // reduction, in_channels)
#
#     def forward(self, x):
#         # 动态归一化
#         x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * self.scale
#
#         # 下采样
#         x_down = self.down(x_norm.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#
#         # 多尺度处理
#         x3 = self.dwconv3(x_down)
#         x5 = self.dwconv5(x_down)
#         x7 = self.dwconv7(x_down)
#
#         # 特征聚合
#         x_fused = x3 + x5 + x7
#         x_out = self.up(x_fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#
#         return x + 0.3 * x_out


class FeatureFusion(nn.Module):
    def __init__(self, in_channels=[128, 256, 512, 1024], out_channels=64):
        super().__init__()
        up_sample = 2
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, out_channels, 1) for ch in in_channels
        ])

        self.context_blocks = nn.ModuleDict({
            'fg': nn.Sequential(
                nn.Conv2d(256, out_channels * up_sample, 1),
                GlobalContextBlock(out_channels * up_sample)
            ),
            'bg': nn.Sequential(
                nn.Conv2d(256, out_channels * up_sample, 1),
                GlobalContextBlock(out_channels * up_sample)
            ),
            'uc': nn.Sequential(
                nn.Conv2d(256, out_channels * up_sample, 1),
                GlobalContextBlock(out_channels * up_sample)
            )
        })

        # 特征金字塔融合
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for _ in range(3)
        ])

        self.final_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 8, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1)
        )

        self.expand_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=1)

    def _upsample_add(self, x, lateral):
        return F.interpolate(x, size=lateral.shape[2:], mode='bilinear') + lateral

    def forward(self, x1, x2, x3, x4, f_fg, f_bg, f_uc):
        # 对齐多尺度特征
        laterals = [
            self.lateral_convs[0](x1),  # 64 128*128
            self.lateral_convs[1](x2),  # 64 64*64
            self.lateral_convs[2](x3),  # 64 32*32
            self.lateral_convs[3](x4)  # 64 16*16
        ]

        # 自顶向下融合
        fused = laterals[-1]
        for i in range(2, -1, -1):
            fused = self._upsample_add(fused, laterals[i])
            fused = self.fusion_convs[i](fused)  # [B,128,H,W]
        fused = self.expand_conv(fused)  # 扩展维度为 128  b*128*128*128

        # 处理解耦特征（确保输出256通道）
        f_fg = F.interpolate(self.context_blocks['fg'](f_fg), scale_factor=8, mode='bilinear')
        f_bg = F.interpolate(self.context_blocks['bg'](f_bg), scale_factor=8, mode='bilinear')
        f_uc = F.interpolate(self.context_blocks['uc'](f_uc), scale_factor=8, mode='bilinear')

        # 最终融合（现在应该是256 * 4=1024通道）
        fused_features = torch.cat([
            fused,  # [B,128,128,128]
            f_fg,  # [B,128,128,128]
            f_bg,  # [B,128,128,128]
            f_uc  # [B,128,128,128]
        ], dim=1)  # [B,512,128,128]

        # 处理逻辑保持不变... b*64*128*128
        output = self.final_fusion(fused_features)
        assert output.size(1) == 64, f"Output channels should be 64, got {output.size(1)}"
        return output


class GlobalContextBlock(nn.Module):
    """全局上下文模块（自动适应输入通道，支持轻量化与残差连接）"""

    def __init__(self, in_ch, reduction=8, min_mid_ch=1, use_depthwise=False, act_type='relu', use_residual=False):
        """
        Args:
            in_ch (int): 输入通道数
            reduction (int): 通道缩减比例，默认16
            min_mid_ch (int): 中间通道数的最小值，避免过小导致数值不稳定
            use_depthwise (bool): 是否使用深度可分离卷积，减少参数量
            act_type (str): 激活函数类型，支持 'relu'、'leaky_relu'、'swish'
            use_residual (bool): 是否添加残差连接
        """
        super().__init__()
        # 计算中间通道数
        mid_ch = max(in_ch // reduction, min_mid_ch)

        # 动态选择激活函数
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act_type == 'swish':
            self.act = nn.SiLU(inplace=True)  # PyTorch 1.7+ 支持 SiLU
        else:
            raise ValueError(f"Unsupported activation type: {act_type}")

        # 构建模块
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            # 降维卷积（可选深度可分离）
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False,
                      groups=in_ch if use_depthwise else 1),
            nn.BatchNorm2d(mid_ch),
            self.act,
            # 还原通道（可选深度可分离）
            nn.Conv2d(mid_ch, in_ch, kernel_size=1, bias=False,
                      groups=mid_ch if use_depthwise else 1),
            nn.Sigmoid()
        )

        # 残差连接（输入与输出维度一致）
        self.use_residual = use_residual
        if use_residual:
            self.residual = nn.Identity()  # 保持输入不变

    def forward(self, x):
        # 生成注意力权重并广播到原始输入
        attn = self.conv(x)  # [B,C,1,1]
        out = x * attn

        # 添加残差连接
        if self.use_residual:
            out += self.residual(x)  # 残差连接

        return out


#
# class HybridAttention(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.ca = CoordAtt(channels, reduction)
#         self.mona = MonaDepthAdapter(channels, reduction=4)
#
#         # 动态权重学习
#         self.weight = nn.Parameter(torch.tensor([0.5, 0.5]))
#
#     def forward(self, x):
#         ca_out = self.ca(x)
#         mona_out = self.mona(x)
#
#         # 软权重分配
#         w = torch.softmax(self.weight, dim=0)
#         return w[0] * ca_out + w[1] * mona_out

# ---------------- 基础模块（不变）----------------
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# 1. 频域分支：稳定版 FreBlock
# --------------------------------------------------
class StableFreBlock(nn.Module):
    """
    频域增强，修复：
      1. |x|==0 时 angle 产生 NaN
      2. mag/pha 网络输出无界
    """

    def __init__(self, c, eps=1e-8):
        super().__init__()
        self.mag = nn.Sequential(
            nn.Conv2d(c, c, 1), nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(c, c, 1)
        )
        self.pha = nn.Sequential(
            nn.Conv2d(c, c, 1), nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(c, c, 1)
        )
        self.eps = eps
        # 把最后一层权重初始化为 0，让初始阶段 mag/pha 网络输出≈0
        nn.init.zeros_(self.mag[-1].weight)
        nn.init.zeros_(self.mag[-1].bias)
        nn.init.zeros_(self.pha[-1].weight)
        nn.init.zeros_(self.pha[-1].bias)

        # self.freq_bn = nn.BatchNorm2d(c, affine=True, eps=eps)

    def forward(self, x):
        mag = torch.abs(x)
        pha = torch.angle(x + self.eps)  # 防止 0 输入
        mag = self.mag(mag.clamp_min(self.eps))  # 防止 log(0)
        pha = self.pha(pha)
        # print("pha min: ", pha.min(), "pha max: ", pha.max())
        # print("mag min: ", mag.min(), "mag max: ", mag.max())
        xfft = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
        return xfft


# --------------------------------------------------
# 2. 空域分支：稳定版 HybridAttention
# --------------------------------------------------
class StableMonaDepthAdapter(nn.Module):
    """
    把 LayerNorm 换成 BatchNorm2d + 输入截断，彻底避免极端值
    """

    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced_c = in_channels // reduction
        self.down = nn.Conv2d(in_channels, reduced_c, 1)
        self.dw3 = nn.Conv2d(reduced_c, reduced_c, 3, 1, 1, groups=reduced_c)
        self.dw5 = nn.Conv2d(reduced_c, reduced_c, 5, 1, 2, groups=reduced_c)
        self.dw7 = nn.Conv2d(reduced_c, reduced_c, 7, 1, 3, groups=reduced_c)
        self.up = nn.Conv2d(reduced_c, in_channels, 1)
        self.norm = nn.BatchNorm2d(in_channels)  # 稳定
        self.scale = nn.Parameter(torch.zeros(1))  # 初始 0，渐进学习

    def forward(self, x):
        x = self.norm(x)
        x = x.clamp(-10, 10)  # 硬截断
        x_down = self.down(x)
        x3 = self.dw3(x_down)
        x5 = self.dw5(x_down)
        x7 = self.dw7(x_down)
        x_out = self.up(x3 + x5 + x7)
        return x + self.scale.tanh() * x_out  # tanh 把 scale 限制在 [-1,1]


class StableCoordAtt(nn.Module):
    # 原实现已经比较稳定，只把 sigmoid 换成 softplus 防止饱和
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.h_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.w_pool = nn.AdaptiveAvgPool2d((1, None))
        mid_c = max(in_channels // reduction, 4)
        self.conv1 = nn.Conv2d(in_channels, mid_c, 1)
        self.conv_h = nn.Conv2d(mid_c, in_channels, 1)
        self.conv_w = nn.Conv2d(mid_c, in_channels, 1)

    def forward(self, x):
        h = self.conv_h(self.conv1(self.h_pool(x)))
        w = self.conv_w(self.conv1(self.w_pool(x)))
        attn = F.softplus(h + w)  # 平滑版 sigmoid
        return x * attn.clamp_max(1.0)


class StableHybridAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = StableCoordAtt(channels, reduction)
        self.mona = StableMonaDepthAdapter(channels, reduction=4)
        self.w = nn.Parameter(torch.tensor([0.6, 0.4]))

    def forward(self, x):
        ca = self.ca(x)
        mona = self.mona(x)
        w = torch.softmax(self.w, dim=0)
        # print(w[0], w[1])
        return w[0] * ca + w[1] * mona


class StableSpaBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = StableHybridAttention(c)

    def forward(self, x):
        return x + self.block(x)


# --------------------------------------------------
# 3. 最终融合模块：稳定版 SFD_HA_Block
# --------------------------------------------------
class StableSFD_HA_Block(nn.Module):
    def __init__(self, in_nc, clamp=0.8):
        super().__init__()
        self.spa = StableSpaBlock(in_nc)
        self.fre = StableFreBlock(in_nc)
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 1, 1, 0)
        # 初始 0.1，防止残差放大
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.clamp = clamp

    def forward(self, x):
        x = x - x.mean(dim=(2, 3), keepdim=True)  # 零均值
        x = x.clamp(-10, 10)  # 截断极端值

        B, C, H, W = x.shape

        # ---- 频域分支 ----
        x_fft = torch.fft.rfft2(x, norm='backward')
        x_fft = self.fre(x_fft)
        x_freq = torch.fft.irfft2(x_fft, s=(H, W), norm='backward')

        # ---- 空域分支 ----
        x_spa = self.spa(x)

        # ---- 融合 ----
        fused = self.cat(torch.cat([x_spa, x_freq], dim=1))
        # 把 scale 限制在 [0,1] 之间，彻底杜绝放大
        return x + torch.sigmoid(self.scale) * fused



class LightweightContrastGuidedRefiner(nn.Module):
    def __init__(self, feat_c, guide_c, reduction=8):
        super().__init__()
        # 1. 把 guide 压缩到低维，提取“引导语义”
        self.guide_compress = nn.Sequential(
            nn.Conv2d(guide_c, feat_c // reduction, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 2. 计算空间注意力（基于局部对比）
        self.local_contrast = nn.Sequential(
            nn.Conv2d(feat_c // reduction, 1, kernel_size=3, padding=1),  # 提取空间响应
            nn.Sigmoid()  # 输出 [0,1] 注意力掩码
        )

        # 3. 轻量残差调整器（微调特征）
        self.delta_gen = nn.Sequential(
            nn.Conv2d(feat_c, feat_c // reduction, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_c // reduction, feat_c, kernel_size=3, padding=1),
        )

        # 4. 可学习门控（控制微调强度）
        self.gate = nn.Parameter(torch.tensor(0.1))  # 默认只微调 10%

    def forward(self, feat, guide):
        B, C, H, W = feat.shape

        # 上采样 guide 到 feat 分辨率（如果需要）
        if guide.shape[2:] != feat.shape[2:]:
            guide = F.interpolate(guide, size=(H, W), mode='bilinear', align_corners=False)

        # 1. 压缩 guide 语义
        guide_comp = self.guide_compress(guide)  # [B, C//r, H, W]

        # 2. 生成“对比感知”的空间注意力（哪里是角点区域？）
        spatial_attn = self.local_contrast(guide_comp)  # [B, 1, H, W]

        # 3. 生成微调增量 Δ
        delta = self.delta_gen(feat)  # [B, C, H, W]

        # 4. 用空间注意力加权 Δ → 只在角点区域微调
        refined_delta = delta * spatial_attn  # [B, C, H, W]

        # 5. 门控残差输出
        gate = torch.sigmoid(self.gate)  # 控制幅度，防止过调
        out = feat + gate * refined_delta

        return out


class MultiStageFusionModule(nn.Module):
    def __init__(self, c1, c2, c3, c4, guide_c, reduction=8):
        super().__init__()
        # 原 refiner 保留
        self.refine_x1 = LightweightContrastGuidedRefiner(c1, guide_c, reduction)
        self.refine_x2 = LightweightContrastGuidedRefiner(c2, guide_c, reduction)
        self.refine_x3 = LightweightContrastGuidedRefiner(c3, guide_c, reduction)
        self.refine_x4 = LightweightContrastGuidedRefiner(c4, guide_c, reduction)

        # 通道压缩 1×1（必须，否则 cat 后通道翻倍）
        self.compress4 = nn.Sequential(
            nn.Conv2d(c4, c3, 1, bias=False),
            nn.GroupNorm(8, c3)  # 8 组即可
        )
        self.compress3 = nn.Sequential(
            nn.Conv2d(c3, c2, 1, bias=False),
            nn.GroupNorm(8, c2)
        )
        self.compress2 = nn.Sequential(
            nn.Conv2d(c2, c1, 1, bias=False),
            nn.GroupNorm(8, c1)
        )

        # 可学习残差缩放 α（初始 0.5，sigmoid 后 0~1）
        self.alpha4 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x1, x2, x3, x4, x3_p):
        # Step 1: refine 最深层
        x4_refined = self.refine_x4(x4, x3_p)  # [B, 512, 32, 32]

        # Step 2: 上采样 + 压缩通道 + 可学习残差融合 → x3
        x4_up = F.interpolate(x4_refined, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x4_up = self.compress4(x4_up)  # [B, 256, 64, 64]
        alpha4 = torch.sigmoid(self.alpha4)  # 0~1
        x3_fused = x3 + alpha4 * x4_up  # 可控残差
        x3_refined = self.refine_x3(x3_fused, x3_p)

        # Step 3: 融合到 x2
        x3_up = F.interpolate(x3_refined, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x3_up = self.compress3(x3_up)  # [B, 128, 128, 128]
        alpha3 = torch.sigmoid(self.alpha3)
        x2_fused = x2 + alpha3 * x3_up
        x2_refined = self.refine_x2(x2_fused, x3_p)

        # Step 4: 融合到 x1
        x2_up = F.interpolate(x2_refined, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x2_up = self.compress2(x2_up)  # [B, 64, 256, 256]
        alpha2 = torch.sigmoid(self.alpha2)
        x1_fused = x1 + alpha2 * x2_up
        x1_refined = self.refine_x1(x1_fused, x3_p)

        return x1_refined, x2_refined, x3_refined, x4_refined
