"""
File: data/cryo_fiber.py
Author: Tingyin Zhao @ SGroup, EE, Tsinghua
Date: 2025-12-23
Description: Defines the CryoFiberDataset class for robust loading, augmentation, and ground truth target generation.
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config.net_config import model_cfg
from .create_gt import proposal_target_generator
from .augmentations import (
    Compose, ConvertFromInts, PhotometricDistort,
    Expand, RandomMirror, RandomRotate, RandomInvert,
    ToPercentCoords, Resize
)


class CryoFiberDataset(Dataset):
    def __init__(self, data_dir, split='train', max_fibers=30, contour_len=128):
        self.data_dir = data_dir
        self.split = split
        self.max_fibers = max_fibers
        self.contour_len = contour_len
        self.stride = model_cfg.get('down_ratio', 4)
        self.input_size = model_cfg.get('train_size', 512)

        # 1. Load Annotations
        ann_file = os.path.join(data_dir, 'annotations', f'instances_{split}2025.json')
        self.img_dir = os.path.join(data_dir, 'images', f'{split}2025')

        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        print(f"Loading annotations from {ann_file}...")
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        self.images = {img['id']: img for img in self.coco['images']}
        self.img_ids = list(self.images.keys())

        self.img_to_anns = {img_id: [] for img_id in self.img_ids}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id in self.img_to_anns:
                self.img_to_anns[img_id].append(ann)

        # 2. Define Augmentation Pipeline
        if split == 'train':
            self.augmentor = Compose([
                ConvertFromInts(),  # -> Float32
                PhotometricDistort(),  # Color Jitter (expects BGR)
                Expand(mean=(0.406 * 255, 0.456 * 255, 0.485 * 255)),  # Zoom out (Fill BGR mean)
                RandomMirror(),  # Horizontal Flip
                RandomInvert(),  # Vertical Flip
                RandomRotate(),  # 90 Degree Rotate
                ToPercentCoords(),  # -> [0, 1]
                Resize(self.input_size)  # -> input_size x input_size
            ])
        else:
            # Val/Test: Resize only
            self.augmentor = Compose([
                ConvertFromInts(),
                ToPercentCoords(),
                Resize(self.input_size)
            ])

        print(f"Loaded {len(self.img_ids)} images for split: {split}")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.images[img_id]
        file_name = img_info['file_name']
        img_path = os.path.join(self.img_dir, file_name)

        # Read Image (BGR)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")

        orig_h, orig_w = img.shape[:2]

        # Parse Points -> Pseudo Boxes [x, y, x, y]
        anns = self.img_to_anns[img_id]
        all_points = []
        point_labels = []

        for fiber_idx, ann in enumerate(anns):
            if 'line' not in ann:
                continue
            # Flatten lines to points
            pts = np.array(ann['line'], dtype=np.float32).reshape(-1, 2)
            for pt in pts:
                # Add padding to make valid boxes
                r = 3.0
                x1 = max(0, pt[0] - r)
                y1 = max(0, pt[1] - r)
                x2 = min(orig_w, pt[0] + r)
                y2 = min(orig_h, pt[1] + r)

                all_points.append([x1, y1, x2, y2])
                point_labels.append(fiber_idx)

        boxes = np.array(all_points, dtype=np.float32)

        labels = np.array(point_labels, dtype=np.int64)

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        # Apply Augmentation
        # img is BGR, boxes are relative [0, 1]
        img, boxes, labels = self.augmentor(img, boxes, labels)

        # Color Conversion BGR -> RGB & Normalization
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
        img /= 255.0
        img -= np.array([0.485, 0.456, 0.406], dtype=np.float32)  # RGB Mean
        img /= np.array([0.229, 0.224, 0.225], dtype=np.float32)  # RGB Std

        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]

        # Reconstruct Fibers from Points
        output_h, output_w = self.input_size // self.stride, self.input_size // self.stride
        gt_labels_reconstructed = []

        if len(boxes) > 0:
            unique_ids = np.unique(labels)
            for fid in unique_ids:
                mask = (labels == fid)

                # Recover coords
                current_boxes = boxes[mask]
                pts_norm_x = (current_boxes[:, 0] + current_boxes[:, 2]) / 2.0
                pts_norm_y = (current_boxes[:, 1] + current_boxes[:, 3]) / 2.0
                pts_norm = np.stack([pts_norm_x, pts_norm_y], axis=1)

                pts_pixel = pts_norm * self.input_size

                # Filter short fragments
                if len(pts_pixel) < 2:
                    continue


                gt_labels_reconstructed.append({'points': pts_pixel})

        # Generate Targets
        targets = proposal_target_generator(
            gt_labels_reconstructed,
            (output_w, output_h),
            num_classes=1,
            stride=self.stride,
            contour_len=self.contour_len,
            max_fibers=self.max_fibers
        )

        (hm, reg, ends, ind_masks, gt_df, gt_af, af_mask, gt_contours, contour_masks) = targets

        return {
            'image': img_tensor,
            'hm': torch.from_numpy(hm),
            'reg': torch.from_numpy(reg),
            'ends': torch.from_numpy(ends),
            'ind_masks': torch.from_numpy(ind_masks),
            'gt_df': torch.from_numpy(gt_df),
            'gt_af': torch.from_numpy(gt_af),
            'af_mask': torch.from_numpy(af_mask),
            'gt_contours': torch.from_numpy(gt_contours),
            'contour_masks': torch.from_numpy(contour_masks),
            'meta': {
                'img_id': img_id,
                'file_name': file_name,
                'orig_size': np.array([orig_w, orig_h], dtype=np.int32)
            }
        }
