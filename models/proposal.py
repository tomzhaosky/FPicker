"""
File: models/proposal.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2026-01-22
Description: Proposal module.
             Integrates Backbone features, DilateEncoder, FPN Decoder with CBAM, and Heads.
"""

import math
import torch
import torch.nn as nn
from backbone.modules import DilateEncoder


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes, ratio=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class Proposal(nn.Module):
    """
    The Heavy-Lifting Detector.
    Features:
    - DilateEncoder for C5/C4 context.
    - FPN-style Decoder (Deconv + Lateral).
    - CBAM Attention at each fusion stage.
    """

    def __init__(self, backbone, heads, head_conv=64):
        super(Proposal, self).__init__()
        self.backbone = backbone
        self.heads = heads

        # -----------------------------------------------------------------
        # 1. Context Enhancement (DilateEncoder)
        # Input: C4 (2048ch for ResNet50), Output: 2048ch
        # -----------------------------------------------------------------
        c4_ch = 2048  # ResNet50 Bottleneck expansion * 512
        self.dilate_encoder = DilateEncoder(c4_ch, c4_ch, dilation_list=[4, 8, 12, 16])

        # -----------------------------------------------------------------
        # 2. FPN Lateral Layers (Project ResNet features to decoding dims)
        # -----------------------------------------------------------------
        # C3 (1024) -> 256
        self.lat_conv3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # C2 (512) -> 128
        self.lat_conv2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # C1 (256) -> 64
        self.lat_conv1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # -----------------------------------------------------------------
        # 3. Deconv Layers (Upsampling)
        # -----------------------------------------------------------------
        # Stage 1: C4(2048) -> 256 (2x up)
        self.deconv1 = self._make_deconv_block(c4_ch, 256)
        self.cbam1 = CBAM(256)  # Attention after fusion

        # Stage 2: Fuse1(256) -> 128 (2x up)
        self.deconv2 = self._make_deconv_block(256, 128)
        self.cbam2 = CBAM(128)  # Attention after fusion

        # Stage 3: Fuse2(128) -> 64 (2x up)
        self.deconv3 = self._make_deconv_block(128, 64)
        self.cbam3 = CBAM(64)  # Attention after fusion

        # -----------------------------------------------------------------
        # 4. Detection Heads
        # -----------------------------------------------------------------
        self.head_hm = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, heads['hm'], kernel_size=1)
        )
        self.head_reg = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, heads['reg'], kernel_size=1)
        )
        self.head_ends = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, heads['ends'], kernel_size=1)
        )
        self.head_aux = nn.Sequential(
            nn.Conv2d(64, head_conv, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, heads['aux'], kernel_size=1)
        )

        self.init_weights()

        # Explicitly initialize heatmap bias
        init_prob = 0.1
        bias_value = -math.log((1. - init_prob) / init_prob)
        nn.init.constant_(self.head_hm[-1].bias, bias_value)

    @staticmethod
    def _make_deconv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2,
                               padding=1, output_padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. Backbone Extraction (Multi-scale)
        # c1: 256, c2: 512, c3: 1024, c4: 2048
        c1, c2, c3, c4 = self.backbone(x)

        # 2. Context Enhancement
        c4 = self.dilate_encoder(c4)

        # 3. FPN Decoding + Attention
        # Up 1: C4 -> 256 + C3_lat
        up1 = self.deconv1(c4)
        fuse1 = up1 + self.lat_conv3(c3)
        feat1 = self.cbam1(fuse1)

        # Up 2: Feat1 -> 128 + C2_lat
        up2 = self.deconv2(feat1)
        fuse2 = up2 + self.lat_conv2(c2)
        feat2 = self.cbam2(fuse2)

        # Up 3: Feat2 -> 64 + C1_lat
        up3 = self.deconv3(feat2)
        fuse3 = up3 + self.lat_conv1(c1)
        feats = self.cbam3(fuse3)  # Final 64ch feature map

        # 4. Heads
        out_hm = torch.sigmoid(self.head_hm(feats))
        out_reg = self.head_reg(feats)
        out_ends = self.head_ends(feats)
        out_aux = self.head_aux(feats)

        return feats, out_hm, out_reg, out_ends, out_aux
