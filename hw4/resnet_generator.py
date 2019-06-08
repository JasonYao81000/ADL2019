# -*- coding: utf-8 -*-
"""
Created on Thu May 23 18:05:06 2019

@author: hb2506
"""
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1,
                 base_width=128, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 128:
            raise ValueError('BasicBlock only supports groups=1 and base_width=512')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1_upsample and self.upsample layers upsample the input when stride != 1
            
        self.conv1 = conv3x3(inplanes, planes)
#        self.conv1_upsample = nn.Upsample(scale_factor=2)
        self.bn1 = norm_layer(planes, 0.8)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, 0.8)
        self.stride = stride
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)
        
#        if self.stride != 1 :
#            out = self.conv1_upsample(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)
        
        out += identity
        out = self.lrelu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=128,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        self.upsample = nn.Upsample(scale_factor=2)
        
        self.groups = groups
        self.base_width = width_per_group
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1])

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
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                upsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                upsample = nn.Sequential(
                    nn.Upsample(scale_factor=stride),
                    conv1x1(self.inplanes, planes * block.expansion, 1),
                    norm_layer(planes * block.expansion),
                )

#        print(upsample)
        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.upsample(x)
#        print("in")
        x = self.layer1(x)
#        print("out")
        x = self.upsample(x)
        x = self.layer2(x)
#        print("out2")
        return x

def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.label_emb = nn.Linear(opt.n_classes, opt.latent_dim, bias=False)
        self.hidden_cat = nn.Linear(opt.latent_dim, opt.latent_dim, bias=False)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim + opt.n_classes, 128 * self.init_size ** 2))
        
        self.resnet = _resnet('resnet', BasicBlock, [2, 2])

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            self.resnet,
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        hidden = self.hidden_cat(noise)
        gen_input = torch.cat((labels, hidden), dim=-1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
#        print(out.size())
        img = self.conv_blocks(out)
        return img

