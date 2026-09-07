"""
File: utils/loss.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2025-12-22
Description: Defines multi-task loss functions including Modified Focal Loss
and contour-evolution fitting and uniformity losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModifiedFocalLoss(nn.Module):
    def __init__(self):
        super(ModifiedFocalLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, gt):
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()
        neg_weights = torch.pow(1 - gt, 4)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.sum()
        if num_pos == 0:
            return -neg_loss.sum() / (pred.shape[-2] * pred.shape[-1])
        return -(pos_loss.sum() + neg_loss.sum()) / num_pos


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()
        self.eps = 1e-4

    def forward(self, pred, target, mask):
        pred = pred.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        mask = mask.unsqueeze(3).expand_as(pred)
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        return loss / (mask.sum() + self.eps)


class EvolutionLoss(nn.Module):
    def __init__(self, lambda_hm=20.0, lambda_evolution=1.0, lambda_aux=0.1, lambda_ends=0.5, df_max=50.0):
        super(EvolutionLoss, self).__init__()
        self.focal_loss = ModifiedFocalLoss()
        self.reg_l1 = RegL1Loss()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss()

        self.lambda_hm = lambda_hm
        self.lambda_evolution = lambda_evolution
        self.lambda_aux = lambda_aux
        self.lambda_ends = lambda_ends
        self.eps = 1e-6
        self.df_max = df_max

    def forward(self, outputs, targets):
        hm_p, reg_p, ends_p, aux_p, contour_p = outputs['hm'], outputs['reg'], outputs['ends'], outputs['aux'], outputs[
            'contours']
        hm_g, reg_g, ends_g, ind_m, df_g, af_g, af_m, contour_g, contour_m = targets

        # 1. Detection Heads
        loss_hm = self.focal_loss(hm_p, hm_g)
        loss_reg = self.reg_l1(reg_p, reg_g, ind_m)
        loss_ends = self.reg_l1(ends_p, ends_g, ind_m)

        # 2. Auxiliary Geometric Heads
        near_mask = (df_g < 20.0).float()
        loss_df = (torch.abs(aux_p[:, 0:1] / self.df_max - df_g / self.df_max)
                   * near_mask).sum() / (near_mask.sum() + self.eps)

        diff_af = torch.abs(aux_p[:, 1:3] - af_g)
        loss_af = (diff_af * near_mask).sum() / (near_mask.sum() * 2.0 + self.eps)
        loss_aux = loss_df + loss_af

        # 3. Contour Evolution
        loss_evolution = torch.tensor(0.0, device=hm_p.device)
        if contour_p is not None:
            pt_mask = contour_m.view(*contour_p.shape[:2], 1, 1).expand_as(contour_p)
            loss_fit = (self.smooth_l1(contour_p, contour_g) * pt_mask).sum() / (pt_mask.sum() + self.eps)

            diffs = contour_p[:, :, 1:] - contour_p[:, :, :-1]
            dists = torch.norm(diffs, dim=-1)
            mean_dist = torch.mean(dists, dim=2, keepdim=True)
            uni_mask = contour_m.view(contour_p.shape[0], contour_p.shape[1], 1)
            loss_uni = (torch.abs(dists - mean_dist) * uni_mask).sum() / (uni_mask.sum() * dists.shape[2] + self.eps)
            loss_evolution = loss_fit + 0.5 * loss_uni

        # 4. Total Loss Calculation
        total_loss = (self.lambda_hm * loss_hm +
                      loss_reg +
                      self.lambda_ends * loss_ends +
                      self.lambda_aux * loss_aux +
                      self.lambda_evolution * loss_evolution
                      )

        return total_loss, {
            'hm': loss_hm.item(), 'reg': loss_reg.item(), 'ends': loss_ends.item(),
            'aux': loss_aux.item(), 'evolution': loss_evolution.item()
        }
