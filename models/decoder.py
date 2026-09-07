"""
File: models/decoder.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2026-01-22
Description: Utility to decode heatmap/regression outputs into initial topological structure.
"""

import torch
import torch.nn.functional as F


def decode_initial_contours(hm, reg, ends, K=30, contour_len=128):
    """
    Decodes proposal outputs into initial lines.
    hm: already passed through sigmoid.
    """
    b, c, h, w = hm.shape

    # NMS
    hmax = F.max_pool2d(hm, (3, 3), 1, 1)
    hm = hm * (hmax == hm).float()

    # Top-K
    scores, inds = torch.topk(hm.view(b, -1), K)
    ys = torch.div(inds, w, rounding_mode='trunc').float()
    xs = (inds % w).float()
    inds_flat = inds.view(b, K)

    # Gather helper
    batch_inds = torch.arange(b, device=hm.device).view(-1, 1).expand(-1, K)

    # Reg
    reg_trans = reg.permute(0, 2, 3, 1).contiguous().view(b, -1, 2)
    reg_k = reg_trans[batch_inds, inds_flat]

    # Ends
    ends_trans = ends.permute(0, 2, 3, 1).contiguous().view(b, -1, 4)
    ends_k = ends_trans[batch_inds, inds_flat]

    # Keypoints
    centers = torch.stack([xs, ys], dim=2) + reg_k

    stretch_factor = 1

    p1 = centers + (ends_k[..., 0:2] * stretch_factor)
    p2 = centers + (ends_k[..., 2:4] * stretch_factor)

    # Prevents dead points in evolution grid_sample due to out-of-bounds.
    p1[..., 0].clamp_(0, w - 1)
    p1[..., 1].clamp_(0, h - 1)
    p2[..., 0].clamp_(0, w - 1)
    p2[..., 1].clamp_(0, h - 1)
    centers[..., 0].clamp_(0, w - 1)
    centers[..., 1].clamp_(0, h - 1)

    # Interpolation
    num_pts = contour_len // 2
    steps1 = torch.linspace(0, 1, num_pts, device=hm.device).view(1, 1, num_pts, 1)
    seg1 = p1.unsqueeze(2) * (1 - steps1) + centers.unsqueeze(2) * steps1

    steps2 = torch.linspace(0, 1, contour_len - num_pts + 1, device=hm.device).view(1, 1, -1, 1)
    steps2 = steps2[..., 1:, :]
    seg2 = centers.unsqueeze(2) * (1 - steps2) + p2.unsqueeze(2) * steps2

    init_contours = torch.cat([seg1, seg2], dim=2)

    return init_contours, scores
