"""
File: models/evolution.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2026-01-22
Description: Iterative deformation/evolution module using replicate padding for open curves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        # Use replicate padding to handle open curve boundaries correctly
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, padding=1, padding_mode='replicate', bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, padding_mode='replicate', bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class Evolution(nn.Module):
    def __init__(self, in_channels=64, contour_len=128):
        super(Evolution, self).__init__()
        self.contour_len = contour_len

        # Input projection
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )

        # GCN Backbone
        self.gcn_stack = nn.Sequential(
            BasicBlock(in_channels, in_channels),
            BasicBlock(in_channels, in_channels),
            BasicBlock(in_channels, in_channels),
            BasicBlock(in_channels, in_channels)
        )

        self.fusion = nn.Conv1d(in_channels * 2, in_channels, 1)
        self.offset_head = nn.Conv1d(in_channels, 2, kernel_size=1)

    def forward(self, feature_map, init_coords, iteration=1):
        B, N, L, _ = init_coords.shape
        H, W = feature_map.shape[-2:]
        C = feature_map.shape[1]

        curr_coords = init_coords.clone()

        for i in range(iteration):
            # Normalize coordinates to [-1, 1]
            grid = curr_coords.clone()
            grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
            grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0

            flat_grid = grid.view(B, N * L, 1, 2)

            # Sampling
            features = F.grid_sample(feature_map, flat_grid, align_corners=True, padding_mode='border')
            features = features.view(B, C, N, L).permute(0, 2, 1, 3).reshape(B * N, C, L)

            # Processing
            x = self.init_conv(features)
            global_feat = torch.max(x, dim=2, keepdim=True)[0].expand_as(x)
            local_feat = self.gcn_stack(x)

            fused = torch.cat([local_feat, global_feat], dim=1)
            fused = self.fusion(fused)
            offsets = self.offset_head(fused).permute(0, 2, 1).view(B, N, L, 2)

            curr_coords = curr_coords + offsets

        return curr_coords
