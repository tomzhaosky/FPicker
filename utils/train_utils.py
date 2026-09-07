"""
File: utils/train_utils.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2025-12-28
Description: Utility functions for training, including EMA and Hungarian Matching.
"""

import copy
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


class ModelEMA(object):
    def __init__(self, model, decay=0.9999):
        self.ema = model.module if hasattr(model, 'module') else model
        self.ema = copy.deepcopy(self.ema)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].to(device=esd[k].device)
                esd[k].copy_(esd[k] * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])


def match_and_align(pred_contours, gt_contours, gt_masks, threshold=15.0):
    """
    Hungarian matching with bidirectional point-order alignment for filaments.
    """
    B = pred_contours.shape[0]
    aligned_inputs = gt_contours.clone()

    # Detach to CPU for Hungarian algorithm
    p_np = pred_contours.detach().cpu().numpy()
    g_np = gt_contours.detach().cpu().numpy()
    m_np = gt_masks.detach().cpu().numpy()

    for b in range(B):
        valid_gt_idx = np.where(m_np[b] > 0.5)[0]
        if len(valid_gt_idx) == 0:
            continue

        cur_g = g_np[b][valid_gt_idx]
        cur_p = p_np[b]

        # Compute cost matrix based on Euclidean distance of centroids
        g_centers = cur_g.mean(axis=1)
        p_centers = cur_p.mean(axis=1)
        dists = np.linalg.norm(g_centers[:, None, :] - p_centers[None, :, :], axis=2)

        row_ind, col_ind = linear_sum_assignment(dists)

        for r, c in zip(row_ind, col_ind):
            if dists[r, c] < threshold:
                gt_absolute_idx = valid_gt_idx[r]

                p_curve = pred_contours[b, c]
                g_curve = gt_contours[b, gt_absolute_idx]

                # Bi-directional distance check to handle point-order ambiguity
                dist_normal = torch.mean(torch.norm(p_curve - g_curve, dim=1))
                dist_flip = torch.mean(torch.norm(p_curve.flip(dims=[0]) - g_curve, dim=1))

                # Align prediction direction with ground truth to stabilize gradients
                if dist_flip < dist_normal:
                    aligned_inputs[b, gt_absolute_idx] = p_curve.flip(dims=[0])
                else:
                    aligned_inputs[b, gt_absolute_idx] = p_curve

    return aligned_inputs
