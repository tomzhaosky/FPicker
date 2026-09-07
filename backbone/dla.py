"""
File: backbone/dla.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2026-02-09
Description: DLA-34 Backbone adapted for FPicker.
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['dla34']

model_urls = {
    'dla34': 'http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth'
}


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x

        if residual is None:
            if self.project:
                residual = self.project(bottom)
            else:
                residual = bottom

        if self.level_root:
            children.append(bottom)

        x1 = self.tree1(x, residual)

        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, block=BasicBlock, residual_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)

        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # Projections to match ResNet50 channels: [256, 512, 1024, 2048]
        self.proj_c1 = nn.Conv2d(channels[2], 256, 1)  # 64 -> 256
        self.proj_c2 = nn.Conv2d(channels[3], 512, 1)  # 128 -> 512
        self.proj_c3 = nn.Conv2d(channels[4], 1024, 1)  # 256 -> 1024
        self.proj_c4 = nn.Conv2d(channels[5], 2048, 1)  # 512 -> 2048

        self.init_weights()

    @staticmethod
    def _make_conv_level(inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def init_weights(self):
        for m in [self.proj_c1, self.proj_c2, self.proj_c3, self.proj_c4]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            if i == 0:
                x = self.level0(x)
            elif i == 1:
                x = self.level1(x)
            elif i == 2:
                x = self.level2(x)
            elif i == 3:
                x = self.level3(x)
            elif i == 4:
                x = self.level4(x)
            elif i == 5:
                x = self.level5(x)
            y.append(x)

        c1 = self.proj_c1(y[2])
        c2 = self.proj_c2(y[3])
        c3 = self.proj_c3(y[4])
        c4 = self.proj_c4(y[5])

        return c1, c2, c3, c4


def dla34(pretrained=False, **kwargs):
    if 'return_levels' in kwargs:
        kwargs.pop('return_levels')

    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        try:
            url = model_urls['dla34']
            print(f"Loading DLA-34 pretrained weights from {url}")
            checkpoint = model_zoo.load_url(url)
            model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            print(f"Warning: Failed to load DLA pretrained weights: {e}")
    return model
