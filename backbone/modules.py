"""
File: backbone/modules.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2025-12-23
Description: Defines fundamental neural network components including DilateEncoder, SPP, CoordConv, and upsampling layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Conv(nn.Module):
    """
    Standard Convolutional Block: Conv2d + BatchNorm2d + Activation
    """

    def __init__(self, in_channels, out_channels, k, s=1, p=0, d=1, g=1, act: Optional[str] = 'relu'):
        super(Conv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
            nn.BatchNorm2d(out_channels)
        ]

        if act is None:
            pass
        elif act == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif act == 'leaky':
            layers.append(nn.LeakyReLU(0.1, inplace=True))

        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class UpSample(nn.Module):
    """
    Wrapper for torch.nn.functional.interpolate
    """

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(UpSample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corner = align_corner

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor,
                             mode=self.mode, align_corners=self.align_corner)


class ResizeConv(nn.Module):
    """
    Upsampling followed by a Convolution (to reduce aliasing compared to ConvTranspose)
    """

    def __init__(self, in_channels, out_channels, act='relu', size=None, scale_factor=None, mode='nearest',
                 align_corner=None):
        super(ResizeConv, self).__init__()
        self.upsample = UpSample(size=size, scale_factor=scale_factor, mode=mode, align_corner=align_corner)
        self.conv = Conv(in_channels, out_channels, k=1, act=act)

    def forward(self, x):
        return self.conv(self.upsample(x))


class Bottleneck(nn.Module):
    """
    Standard ResNet Bottleneck with Dilation support.
    Structure: 1x1 Conv -> 3x3 Conv -> 1x1 Conv + Residual
    """

    def __init__(self, in_channels, dilation=1, expand_ratio=0.5, act='relu'):
        super(Bottleneck, self).__init__()
        hidden_channels = int(in_channels * expand_ratio)
        self.branch = nn.Sequential(
            Conv(in_channels, hidden_channels, k=1, act=act),
            Conv(hidden_channels, hidden_channels, k=3, p=dilation, d=dilation, act=act),
            Conv(hidden_channels, in_channels, k=1, act=act)
        )

    def forward(self, x):
        return x + self.branch(x)


class DilateEncoder(nn.Module):
    """
    Encoder module using dilated bottlenecks to capture multi-scale context.
    """

    def __init__(self, in_channels, out_channels, act='relu', dilation_list=None):
        super(DilateEncoder, self).__init__()

        if dilation_list is None:
            dilation_list = [4, 8, 12, 16]

        self.projector = nn.Sequential(
            Conv(in_channels, out_channels, k=1, act=None),
            Conv(out_channels, out_channels, k=3, p=1, act=None)
        )

        encoders = []
        for d in dilation_list:
            encoders.append(Bottleneck(in_channels=out_channels, dilation=d, act=act))
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)
        return x


class SPP(nn.Module):
    """
    Spatial Pyramid Pooling (SPP) module.
    Concatenates feature maps pooled at different kernel sizes.
    """

    def __init__(self, in_channels, out_channels, expand_ratio=0.5, act='relu'):
        super(SPP, self).__init__()
        hidden_channels = int(in_channels * expand_ratio)
        self.cv1 = Conv(in_channels, hidden_channels, k=1, act=act)
        self.cv2 = Conv(hidden_channels * 4, out_channels, k=1, act=act)

    def forward(self, x):
        x = self.cv1(x)
        x_1 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = F.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)
        x = self.cv2(x)
        return x


class CoordConv(nn.Module):
    """
    Coordinate Convolution.
    Concatenates normalized (x, y) coordinates to feature maps before convolution.
    """

    def __init__(self, in_channels, out_channels, k=1, p=0, s=1, d=1, g=1, act='relu'):
        super(CoordConv, self).__init__()
        self.conv = Conv(in_channels + 2, out_channels, k=k, p=p, s=s, d=d, g=g, act=act)

    def forward(self, x):
        B, _, H, W = x.size()
        device = x.device

        # Generate coordinate grid
        # Use indexing='ij' to avoid warnings
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

        # Stack and expand to batch size: [B, 2, H, W]
        grid_xy = torch.stack([grid_x, grid_y], dim=0).float()
        grid_xy = grid_xy.unsqueeze(0).repeat(B, 1, 1, 1).to(device)

        # Normalize coordinates to range [-1, 1]
        grid_xy[:, 0, :, :] = (grid_xy[:, 0, :, :] / (W - 1)) * 2.0 - 1.0
        grid_xy[:, 1, :, :] = (grid_xy[:, 1, :, :] / (H - 1)) * 2.0 - 1.0

        # Concatenate coordinate channels
        x_coord = torch.cat([x, grid_xy], dim=1)
        y = self.conv(x_coord)

        return y
