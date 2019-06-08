# -*- coding: utf-8 -*-
"""
Created on Thu May 23 18:05:06 2019

@author: hb2506
"""
import torch
from torch import nn

IMAGE_CHANNELS = 3
IMAGE_SIZE = 128

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=16, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 16:
            raise ValueError('BasicBlock only supports groups=1 and base_width=16')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.utils.spectral_norm(conv3x3(inplanes, planes, stride))
        self.bn1 = norm_layer(planes, 0.8)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout2d(0.25)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = self.dropout(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=16, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 3
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.layer1 = self._make_layer(block, 16, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.utils.spectral_norm(conv1x1(self.inplanes, planes * block.expansion, stride)),
                norm_layer(planes * block.expansion, 0.8),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        
        x = x.reshape(x.size(0), -1)
        
        return x

def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        
        self.resnet = _resnet('resnet', BasicBlock, [2, 2, 2, 2])
        flatten_size = 128 * BasicBlock.expansion * ((opt.img_size // 2 ** 4) ** 2)

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(flatten_size, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(flatten_size, opt.n_classes), nn.Sigmoid())


    def forward(self, img):
        out = self.resnet(img)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

