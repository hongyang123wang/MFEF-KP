import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from nets.block.res_Condg import decoder_block, \
    output_block, AuxiliaryHead, DecoupleLayer
from nets.block.res_addother import StableSFD_HA_Block, MultiStageFusionModule, StableHybridAttention

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


class Up(nn.Module):
    def __init__(self, in_ch=32, out_ch=16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 3, padding=1)  # 4 = 2×2

    def forward(self, x):
        x = self.conv(x)  # (B, 64, 256, 256)
        x = F.pixel_shuffle(x, 2)  # (B, 16, 512, 512)
        return x


class Bottleneck(nn.Module):
    # expansion = 4
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# -----------------------------------------------------------------#
#   使用Renset50作为主干特征提取网络，最终会获得一个
#   16x16x2048的有效特征层
# -----------------------------------------------------------------#
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 512,512,3 -> 256,256,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 256x256x64 -> 128x128x64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change

        # 128x128x64 -> 128x128x256
        self.layer1 = self._make_layer(block, 64, layers[0])

        # 128x128x256 -> 64x64x512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # 64x64x512 -> 32x32x1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # ====================  新增网络  开始 =====================
        self.aux_head = AuxiliaryHead(128)
        self.decouple_layer = DecoupleLayer(512, 128)

        self.sfd0 = StableSFD_HA_Block(64)  # layer1 输出通道是 128
        self.sfd1 = StableSFD_HA_Block(128)  # layer1 输出通道是 128
        self.sfd2 = StableSFD_HA_Block(256)  # layer2 输出通道是 256
        self.sfd3 = StableSFD_HA_Block(512)  # layer2 输出通道是 256

        # self.ha3 = StableHybridAttention(512)  # layer3后
        # self.ha4 = HybridAttention(1024)  # layer4后

        self.multi_scale_fusion = MultiStageFusionModule(64, 128, 256, 512, 128)

        self.decoder_small = decoder_block(512, 256, scale=2)
        self.decoder_middle = decoder_block(256, 128, scale=2)
        self.decoder_large = decoder_block(128, 64, scale=2)

        self.output_block = output_block(64, 64)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): # 3 512 512
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)  # 32 256 256

        x1 = self.maxpool(x0)  # 64 128 128

        x1 = self.layer1(x1)  # 128 128 128

        x2 = self.layer2(x1)  # 256 64 64

        x3 = self.layer3(x2)  # 512 32 32

        x0_ehance = self.sfd0(x0)

        x1_ehance = self.sfd1(x1)

        x2_ehance = self.sfd2(x2)

        x3_ehance = self.sfd3(x3)

        x3_shm = self.decouple_layer(x3)

        mask_kp = self.aux_head(x3_shm)

        # 2 64 256 256  ; 2 128 128 128 ; 2 256 64 64 ; 2 512 32 32
        x1_fused, x2_fused, x3_fused, x4_fused = self.multi_scale_fusion(x0_ehance, x1_ehance, x2_ehance, x3_ehance,
                                                                         x3_shm)

        x_large = self.decoder_large(x2_fused, x1_fused)  # b 64 256 256
        x_middle = self.decoder_middle(x3_fused, x2_fused)  # b 128 128 128
        x_small = self.decoder_small(x4_fused, x3_fused)  # b 256 64 64

        x_fused = self.output_block(x_large, x_middle, x_small)

        return x_fused, mask_kp


def resnet50(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls['resnet50'], model_dir='model_data/')
    #     model.load_state_dict(state_dict)
    #     print("pretrained model loaded!")
    # # ----------------------------------------------------------#
    # #   获取特征提取部分
    # # ----------------------------------------------------------#
    # features = list(
    #     [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4])
    # features = nn.Sequential(*features)
    return model


class resnet50_Decoder(nn.Module):
    def __init__(self, inplanes, bn_momentum=0.1):
        super(resnet50_Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.deconv_with_bias = False

        # ----------------------------------------------------------#
        #   16,16,2048 -> 32,32,256 -> 64,64,128 -> 128,128,64
        #   利用ConvTranspose2d进行上采样。
        #   每次特征层的宽高变为原来的两倍。
        # ----------------------------------------------------------#
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


class resnet50_Head(nn.Module):
    def __init__(self, inchannel=64, num_classes=1, outchannel=8, bn_momentum=0.1, poly_setting=None):
        super(resnet50_Head, self).__init__()

        self.poly = poly_setting["poly"]
        self.two_offset_branch = poly_setting["two_offset_branch"]
        self.direct_theta = poly_setting["direct_theta"]
        # 是否为物体
        self.cls_head = nn.Sequential(
            nn.Conv2d(inchannel, outchannel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, num_classes,
                      kernel_size=1, stride=1, padding=0))

        # 关键点预测部分
        self.kp_head = nn.Sequential(
            nn.Conv2d(inchannel, outchannel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, num_classes,
                      kernel_size=1, stride=1, padding=0))

        # 中心点偏移预测部分
        self.reg_head = nn.Sequential(
            nn.Conv2d(inchannel, outchannel,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, 2,
                      kernel_size=1, stride=1, padding=0))

        # 关键点偏移部分
        # 极坐标
        if self.direct_theta:
            # 与中心点的距离、角度值(rad)
            self.vertex_offset_head = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(outchannel, momentum=bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, 8, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        hm = self.cls_head(x).sigmoid_()
        ct_offset = self.reg_head(x)
        vertex_offset = self.vertex_offset_head(x)
        kp_hm = self.kp_head(x).sigmoid_()

        return hm, ct_offset, vertex_offset, kp_hm
