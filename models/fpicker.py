"""
File: models/fpicker.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2026-01-22
Description: Top-level Meta-Architecture connecting Proposal, Decoder, and Evolution.
"""

import torch
import torch.nn as nn
from .proposal import Proposal
from .evolution import Evolution
from .decoder import decode_initial_contours
from utils.train_utils import match_and_align


class FPicker(nn.Module):
    def __init__(self, backbone, heads, head_conv=64, contour_len=128):
        super(FPicker, self).__init__()

        # 1. Proposal
        self.proposal = Proposal(backbone, heads, head_conv)

        # 2. Evolution
        self.evolution = Evolution(in_channels=64, contour_len=contour_len)

        self.contour_len = contour_len

    def forward(self, imgs, gt_contours=None, contour_masks=None, p_gt=0.0, mode='train'):
        """
        Args:
            imgs: [B, 3, H, W]
            gt_contours: [B, N, L, 2] (optional, for training)
            contour_masks: [B, N] (optional, for training)
            p_gt: float, probability to use Teacher Forcing
            mode: 'train' or 'test'
        """

        # 1. Run Proposal (Backbone + Neck + Heads)
        feats, out_hm, out_reg, out_ends, out_aux = self.proposal(imgs)

        # 2. Initial Contour Generation
        init_contours = None
        scores = None  # Initialize scores to None

        if mode == 'train':
            # Training logic with Bridge Strategy
            if torch.rand(1).item() < p_gt and gt_contours is not None:
                # [Branch A] Teacher Forcing: Use noisy GT
                noise = torch.randn_like(gt_contours) * 0.5
                init_contours = gt_contours + noise
            else:
                # [Branch B] Auto-Regression: Decode and Match

                # K needs to cover enough candidates to match GT
                K_dynamic = max(30, gt_contours.size(1) if gt_contours is not None else 30)
                pred_init, _ = decode_initial_contours(out_hm, out_reg, out_ends, K=K_dynamic,
                                                       contour_len=self.contour_len)

                if gt_contours is not None and contour_masks is not None:
                    # Match predictions to GT slots for loss calculation
                    init_contours = match_and_align(pred_init, gt_contours, contour_masks)
                else:
                    init_contours = pred_init

        else:
            # Inference logic (val/test/demo)
            # Scores are generated here
            init_contours, scores = decode_initial_contours(out_hm, out_reg, out_ends, K=300,
                                                            contour_len=self.contour_len)

        # 3. Run Evolution
        # Only run if we have initial contours
        if init_contours is not None and init_contours.shape[1] > 0:
            pred_refined = self.evolution(feats, init_contours, iteration=3)
        else:
            pred_refined = None

        return {
            'hm': out_hm,
            'reg': out_reg,
            'ends': out_ends,
            'aux': out_aux,
            'contours': pred_refined,
            'scores': scores
        }
